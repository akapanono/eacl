# ERC 模型增强实现计划：SAS-NSG-EACL

## 0. 任务目标

请在当前 ERC 项目代码基础上实现一个增强版模型，不要从零重写整个项目。

当前已有模型大致是：

- 使用 prompt-based encoder 提取目标话语表示：
  - 输入类似：`For utterance: target_utterance speaker feels <mask>`
  - 使用 `<mask>` 位置向量作为目标话语表示 `h_i`
- 每个情绪类别有多个子锚点：
  - `activation`
  - `interaction`
  - `expression`
  - `context_shift`
- 使用 domain adapters 将样本和锚点映射到不同域空间
- 使用 domain gate 对不同域的预测结果加权融合
- 训练目标包含：
  - 交叉熵损失
  - 监督对比损失
  - 角度分离损失

现在需要在现有模型基础上增加三个模块：

1. `Neutral Decoupling`：中性情绪解耦模块
2. `Speaker State Guidance`：说话人状态引导模块
3. `Similar Anchor Separation`：相似锚点拉开模块

最终模型可以命名为：

```text
SAS-NSG-EACL
```

含义：

```text
SAS = Similar Anchor Separation
N = Neutral Decoupling
SG = Speaker State Guidance
EACL = Emotion-Anchored Contrastive Learning
```

---

## 1. 总体设计原则

### 1.1 不要破坏原模型主体

请保留当前模型已有结构：

- prompt-based encoder
- emotion anchors / subanchors
- domain adapters
- domain gate
- prototype matching
- supervised contrastive loss
- angle separation loss

新增模块应该以“可开关”的方式加入。

所有新增模块都需要通过 config 参数控制，例如：

```yaml
use_neutral_decoupling: true
use_speaker_state: true
use_similar_anchor_separation: true
use_hard_anchor_negative: true
```

如果这些参数为 `false`，模型应该退化为原来的基础模型。

---

## 2. 总体前向传播流程

目标流程如下：

```text
dialogue context + target utterance
        |
        v
Prompt Encoder
        |
        v
h_i = Encoder(x_i)[mask]
        |
        |--------------------------|
        |                          |
        v                          v
Neutral Branch              Speaker State Encoder
        |                          |
        v                          v
p_neu                    r_i = StateEncoder(state_text_i)
        |                          |
        |------------- Fuse --------
                      |
                      v
u_i = Fuse(h_i, r_i)
                      |
                      v
Domain Adapters
activation / interaction / expression / context_shift
                      |
                      v
Domain-wise anchor matching
                      |
                      v
Domain Gate
                      |
                      v
p_emo over non-neutral emotions
                      |
                      v
Final probability:
P(neutral) = p_neu
P(c != neutral) = (1 - p_neu) * p_emo(c)
```

---

## 3. 模块一：Neutral Decoupling 中性情绪解耦

### 3.1 设计动机

ERC 数据集中 `neutral` 类别通常占比较高，而且它并不是普通情绪，而是“无明显情绪 / 弱情绪 / 模糊情绪”的混合状态。

因此不要让 `neutral` 和 `happy`、`sad`、`angry` 等具体情绪共享完全相同的多域子锚点空间。

需要将任务拆成两步：

```text
第一步：判断当前样本是否为 neutral
第二步：如果不是 neutral，再判断具体非中性情绪类别
```

---

### 3.2 类别处理

假设原始类别为：

```python
labels = ["neutral", "happy", "sad", "angry", "excited", "frustrated"]
```

需要构造：

```python
neutral_label = "neutral"

non_neutral_labels = [
    "happy",
    "sad",
    "angry",
    "excited",
    "frustrated",
]
```

如果当前数据集没有 `neutral` 类，则自动关闭 neutral decoupling。

---

### 3.3 Neutral Branch

新增一个二分类分支：

```python
self.neutral_classifier = nn.Sequential(
    nn.Dropout(dropout),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_size, 1)
)
```

输入为融合后的表示 `u_i`，如果没有 speaker state，则输入 `h_i`：

```python
neutral_logit = self.neutral_classifier(u_i).squeeze(-1)
p_neu = torch.sigmoid(neutral_logit)
```

对应标签：

```python
neutral_target = (labels == neutral_id).float()
```

损失：

```python
L_neu = F.binary_cross_entropy_with_logits(
    neutral_logit,
    neutral_target
)
```

---

### 3.4 非中性情绪分支

原来的多域子锚点只用于非中性情绪类别。

如果 `use_neutral_decoupling = true`，则锚点张量应该从：

```python
anchors.shape = [num_labels, num_subanchors, hidden_size]
```

变成：

```python
anchors.shape = [num_non_neutral_labels, num_subanchors, hidden_size]
```

需要建立两个映射：

```python
label_id_to_non_neutral_id
non_neutral_id_to_label_id
```

训练时，如果样本是 neutral：

- 不参与非中性情绪分类损失
- 不参与非中性 anchor supervised contrastive loss
- 只参与 neutral branch loss

如果样本不是 neutral：

- 参与非中性情绪分类损失
- 参与 anchor matching
- 参与 supervised contrastive loss
- 参与 similar anchor separation / hard anchor negative

---

### 3.5 最终概率计算

非中性情绪分支输出：

```python
p_emo = softmax(non_neutral_logits, dim=-1)
```

最终概率：

```text
P(neutral) = p_neu
P(c != neutral) = (1 - p_neu) * p_emo(c)
```

实现时可以构造完整类别概率：

```python
final_probs = torch.zeros(batch_size, num_labels).to(device)

final_probs[:, neutral_id] = p_neu

for non_neu_idx, original_label_id in enumerate(non_neutral_id_to_label_id):
    final_probs[:, original_label_id] = (1 - p_neu) * p_emo[:, non_neu_idx]
```

最终分类损失可以使用：

```python
L_final = F.nll_loss(
    torch.log(final_probs + 1e-8),
    labels
)
```

注意避免 `log(0)`。

---

## 4. 模块二：Speaker State Guidance 说话人状态引导

### 4.1 设计动机

当前模型已有四个域：

```text
activation
interaction
expression
context_shift
```

但是这四个域目前主要依赖锚点模板和训练信号自动学习，没有额外显式信息。

新增 speaker state guidance，用离线生成的说话人状态文本增强模型，尤其增强：

- interaction 域
- context_shift 域
- expression 域

---

### 4.2 数据格式

请支持从数据文件中读取一个可选字段：

```json
{
  "utterance": "...",
  "speaker": "Speaker A",
  "context": ["..."],
  "label": "happy",
  "speaker_state": {
    "mental_state": "the speaker seems relieved but still uncertain",
    "interaction_relation": "the speaker is responding positively to the previous speaker",
    "expression_style": "the emotion is expressed indirectly",
    "context_shift": "the emotion becomes more positive than before"
  }
}
```

如果数据集中没有 `speaker_state` 字段，则：

- `use_speaker_state = false` 时正常运行
- `use_speaker_state = true` 但字段缺失时，使用空字符串或默认状态文本

默认状态文本：

```text
mental_state: unknown.
interaction_relation: unknown.
expression_style: unknown.
context_shift: unknown.
```

---

### 4.3 Speaker State 文本拼接

将四个字段拼接成一段文本：

```python
state_text = (
    "mental_state: {mental_state} "
    "interaction_relation: {interaction_relation} "
    "expression_style: {expression_style} "
    "context_shift: {context_shift}"
)
```

注意：

- 不要让 LLM 输出真实情绪标签
- 不要在 state text 中直接包含 `happy`、`sad` 等标签词
- 该字段只是辅助说明说话人状态，不是直接预测情绪

---

### 4.4 Speaker State Encoder

优先实现简单版本。

如果项目中已有同一个 encoder，可以复用它编码 `state_text`。

输入：

```python
state_input_ids
state_attention_mask
```

输出：

```python
r_i = StateEncoder(state_text_i)
```

可以使用 `[CLS]` 或 mean pooling。

建议实现 mean pooling：

```python
def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
```

得到：

```python
r_i.shape = [batch_size, hidden_size]
```

---

### 4.5 Speaker State Fusion

将 `h_i` 和 `r_i` 融合成 `u_i`。

新增模块：

```python
self.state_proj = nn.Linear(hidden_size, hidden_size)

self.state_gate = nn.Sequential(
    nn.Linear(hidden_size * 2, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.Sigmoid()
)

self.state_fusion_norm = nn.LayerNorm(hidden_size)
```

前向计算：

```python
e_i = self.state_proj(r_i)

alpha_i = self.state_gate(
    torch.cat([h_i, e_i], dim=-1)
)

u_i = self.state_fusion_norm(
    h_i + alpha_i * e_i
)
```

如果不使用 speaker state：

```python
u_i = h_i
```

---

### 4.6 State-guided Domain Gate

原来的 domain gate 可能是：

```python
g_i = softmax(G(h_i))
```

现在改成：

```python
gate_input = torch.cat([u_i, r_i, p_neu], dim=-1)
```

其中：

```python
p_neu.shape = [batch_size, 1]
```

如果不使用 neutral decoupling，则不拼接 `p_neu`。

如果不使用 speaker state，则不拼接 `r_i`。

建议写成可配置输入维度：

```python
gate_input_dim = hidden_size

if use_speaker_state:
    gate_input_dim += hidden_size

if use_neutral_decoupling:
    gate_input_dim += 1
```

然后：

```python
self.domain_gate = nn.Sequential(
    nn.Linear(gate_input_dim, hidden_size),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_size, num_subanchors)
)
```

计算：

```python
domain_weights = F.softmax(self.domain_gate(gate_input), dim=-1)
```

---

## 5. 模块三：Similar Anchor Separation 相似锚点拉开

### 5.1 设计动机

当前模型已经有 angle separation，但它通常是对所有类别中心施加统一分离约束。

新增 Similar Anchor Separation，专门处理容易混淆的相似情绪，例如：

```text
happy vs excited
sad vs frustrated
angry vs frustrated
neutral vs weak sadness
```

注意：

如果使用 neutral decoupling，则 neutral 不在非中性 anchor 空间中，所以 SAS 默认只处理非中性情绪之间的相似对。

---

### 5.2 不要拉开同一情绪内部子锚点

不要做：

```text
happy_activation 远离 happy_expression
```

因为同一情绪内部的四个子锚点本来就是互补的。

应该做：

```text
happy_activation 远离 excited_activation
happy_interaction 远离 excited_interaction
sad_context_shift 远离 frustrated_context_shift
```

即：

```text
只拉开不同类别之间、相同域上的相似锚点。
```

---

### 5.3 相似情绪对集合

先实现静态版本，后续再实现动态版本。

配置文件中增加：

```yaml
similar_emotion_pairs:
  - ["happy", "excited"]
  - ["sad", "frustrated"]
  - ["angry", "frustrated"]
```

代码中需要把 label name 转换成 anchor 内部 id。

如果某个标签不存在，则跳过该 pair，不要报错中断。

---

### 5.4 SAS Loss 公式

对于相似情绪对 `(c, d)`，每个域 `k` 都有对应锚点：

```text
a_{c,k}
a_{d,k}
```

先通过 domain adapter 映射：

```python
v_c_k = D_k(a_c_k)
v_d_k = D_k(a_d_k)
```

归一化：

```python
v_c_k = F.normalize(v_c_k, dim=-1)
v_d_k = F.normalize(v_d_k, dim=-1)
```

计算余弦相似度：

```python
sim = torch.sum(v_c_k * v_d_k, dim=-1)
```

margin loss：

```python
loss = relu(sim - margin) ** 2
```

完整实现：

```python
def similar_anchor_separation_loss(
    anchor_embeddings,
    domain_adapters,
    similar_pairs,
    margin=0.30
):
    """
    anchor_embeddings:
        Tensor, shape [num_non_neutral_labels, num_subanchors, hidden_size]

    domain_adapters:
        ModuleList, length = num_subanchors

    similar_pairs:
        List[Tuple[int, int]]
        label ids in non-neutral anchor space

    return:
        scalar loss
    """

    losses = []

    for c, d in similar_pairs:
        for k, adapter in enumerate(domain_adapters):
            a_c_k = anchor_embeddings[c, k]
            a_d_k = anchor_embeddings[d, k]

            v_c_k = adapter(a_c_k.unsqueeze(0)).squeeze(0)
            v_d_k = adapter(a_d_k.unsqueeze(0)).squeeze(0)

            v_c_k = F.normalize(v_c_k, dim=-1)
            v_d_k = F.normalize(v_d_k, dim=-1)

            sim = torch.sum(v_c_k * v_d_k, dim=-1)

            losses.append(F.relu(sim - margin).pow(2))

    if len(losses) == 0:
        return anchor_embeddings.new_tensor(0.0)

    return torch.stack(losses).mean()
```

---

### 5.5 Hard Anchor Negative Loss

除了直接拉开锚点，还要在样本训练中提高相似情绪负锚点的权重。

例如真实标签是 `happy`，那么 `excited` 应该是 hard negative。

定义：

```python
weight(y, c) = 1 + rho, if (y, c) in similar_pairs
weight(y, c) = 1, otherwise
```

建议：

```yaml
hard_negative_rho: 2.0
hard_negative_temperature: 0.07
```

实现逻辑：

1. 对每个非中性样本取表示 `z_i`
2. 对每个类别计算类别中心 anchor：

```python
class_anchor_c = mean_k(anchor[c, k])
```

3. 计算样本与所有类别中心的相似度：

```python
logits = z_i @ class_anchors.T / temperature
```

4. 对 hard negative 类别增强 logit 权重

更稳定的做法是直接在 denominator 中加权：

```text
pos = exp(logit[y])
denom = pos + sum_{c != y} weight(y, c) * exp(logit[c])
loss = -log(pos / denom)
```

伪代码：

```python
def hard_anchor_negative_loss(
    sample_repr,
    labels,
    class_anchors,
    similar_pair_set,
    temperature=0.07,
    rho=2.0
):
    """
    sample_repr:
        [num_non_neutral_samples, hidden_size]

    labels:
        [num_non_neutral_samples]
        labels in non-neutral label space

    class_anchors:
        [num_non_neutral_labels, hidden_size]

    similar_pair_set:
        set of tuple ids, e.g. {(0, 1), (1, 0)}

    return:
        scalar loss
    """

    sample_repr = F.normalize(sample_repr, dim=-1)
    class_anchors = F.normalize(class_anchors, dim=-1)

    logits = torch.matmul(sample_repr, class_anchors.T) / temperature
    exp_logits = torch.exp(logits)

    losses = []

    for i in range(sample_repr.size(0)):
        y = labels[i].item()

        pos = exp_logits[i, y]
        denom = pos.clone()

        for c in range(class_anchors.size(0)):
            if c == y:
                continue

            weight = 1.0
            if (y, c) in similar_pair_set:
                weight += rho

            denom = denom + weight * exp_logits[i, c]

        losses.append(-torch.log(pos / (denom + 1e-8)))

    if len(losses) == 0:
        return sample_repr.new_tensor(0.0)

    return torch.stack(losses).mean()
```

---

## 6. 总损失函数

最终训练损失：

```python
L_total = (
    L_final
    + lambda_neu * L_neu
    + lambda_supcon * L_supcon
    + lambda_angle * L_angle
    + lambda_sas * L_sas
    + lambda_hard * L_hard
)
```

配置建议：

```yaml
lambda_neu: 0.5
lambda_supcon: 1.0
lambda_angle: 0.05
lambda_sas: 0.02
lambda_hard: 0.05

sas_margin: 0.30
hard_negative_rho: 2.0
hard_negative_temperature: 0.07
```

如果某个模块关闭，则对应 loss 为 0。

---

## 7. 训练阶段建议

请支持直接端到端训练，但最好也支持分阶段训练。

### 7.1 阶段一：基础预热

先训练原模型主体：

```python
L = L_final + lambda_supcon * L_supcon + lambda_angle * L_angle
```

关闭：

```yaml
use_neutral_decoupling: false
use_speaker_state: false
use_similar_anchor_separation: false
use_hard_anchor_negative: false
```

训练 2 到 3 个 epoch。

---

### 7.2 阶段二：加入 neutral 和 speaker state

开启：

```yaml
use_neutral_decoupling: true
use_speaker_state: true
```

关闭：

```yaml
use_similar_anchor_separation: false
use_hard_anchor_negative: false
```

目标是让模型先学会：

```text
neutral / non-neutral 解耦
speaker state 融合
state-guided domain gate
```

---

### 7.3 阶段三：加入相似锚点拉开

开启：

```yaml
use_similar_anchor_separation: true
use_hard_anchor_negative: true
```

目标是进一步改善：

```text
happy vs excited
sad vs frustrated
angry vs frustrated
```

等相似情绪混淆。

---

## 8. 配置文件新增字段

请在 config 中增加以下字段。

```yaml
# Neutral Decoupling
use_neutral_decoupling: true
neutral_label_name: "neutral"
lambda_neu: 0.5

# Speaker State Guidance
use_speaker_state: true
speaker_state_encoder: "shared"  # shared / separate / none
speaker_state_pooling: "mean"    # mean / cls
use_state_in_domain_gate: true
use_state_fusion: true

# Similar Anchor Separation
use_similar_anchor_separation: true
similar_emotion_pairs:
  - ["happy", "excited"]
  - ["sad", "frustrated"]
  - ["angry", "frustrated"]
sas_margin: 0.30
lambda_sas: 0.02

# Hard Anchor Negative
use_hard_anchor_negative: true
hard_negative_rho: 2.0
hard_negative_temperature: 0.07
lambda_hard: 0.05
```

---

## 9. 代码实现建议

请优先搜索当前项目中以下关键词，找到对应文件后再修改：

```text
domain_gate
num_subanchors
anchor
subanchor
prototype
contrastive
supcon
angle_loss
forward
loss
```

建议新增或修改以下模块：

```text
models/
  enhanced_eacl.py 或在现有模型文件中增加类
losses/
  similar_anchor_separation.py
  hard_anchor_negative.py
data/
  speaker_state_dataset.py 或在现有 dataset 中增加 speaker_state 字段读取
configs/
  sas_nsg_eacl.yaml
```

如果项目当前没有这些目录，不要强行创建复杂结构，优先沿用原项目结构。

---

## 10. Forward 输出要求

模型 forward 最好返回一个 dict，方便训练和调试：

```python
return {
    "logits": final_logits,
    "probs": final_probs,
    "neutral_logit": neutral_logit,
    "neutral_prob": p_neu,
    "non_neutral_logits": non_neutral_logits,
    "domain_weights": domain_weights,
    "loss": total_loss,
    "loss_final": L_final,
    "loss_neu": L_neu,
    "loss_supcon": L_supcon,
    "loss_angle": L_angle,
    "loss_sas": L_sas,
    "loss_hard": L_hard,
}
```

如果某个模块关闭，对应字段可以为 `None` 或 0，但不要导致训练代码崩溃。

---

## 11. 推理逻辑

推理时：

```python
final_probs = model(... )["probs"]
pred = final_probs.argmax(dim=-1)
```

如果使用 neutral decoupling：

```python
if p_neu > threshold:
    pred = neutral_id
else:
    pred = argmax(non_neutral_probs)
```

但默认建议使用统一的 `final_probs.argmax()`，避免手动 threshold 带来额外调参。

---

## 12. 消融实验开关

请确保可以方便跑以下 ablation：

| 实验名 | Neutral | Speaker State | SAS | Hard Negative |
|---|---|---|---|---|
| Base | false | false | false | false |
| Base + N | true | false | false | false |
| Base + SG | false | true | false | false |
| Base + SAS | false | false | true | true |
| Base + N + SG | true | true | false | false |
| Base + N + SAS | true | false | true | true |
| Full | true | true | true | true |

---

## 13. 需要额外记录的指标

除了 overall accuracy / weighted-F1 / macro-F1，还需要记录：

```text
neutral F1
non-neutral macro-F1
happy-excited confusion count
sad-frustrated confusion count
angry-frustrated confusion count
average domain gate weight
SAS loss value
hard negative loss value
```

如果项目已有 confusion matrix，请在验证阶段输出：

```text
confusion_matrix.csv
```

并额外输出相似情绪对的混淆率：

```python
confusion_rate(c, d) = M[c, d] + M[d, c]
```

---

## 14. 实现优先级

请按以下顺序实现，避免一次性改太多导致难以调试。

### Step 1：Neutral Decoupling

先实现：

- neutral label 检测
- non-neutral label mapping
- neutral classifier
- final_probs 合成
- L_neu 和 L_final

确保不加 speaker state 和 SAS 时可以正常训练。

---

### Step 2：Speaker State Guidance

再实现：

- dataset 读取 speaker_state 字段
- state_text 拼接
- state encoder
- state fusion
- state-guided domain gate

确保没有 speaker_state 字段时不会崩溃。

---

### Step 3：Similar Anchor Separation

再实现：

- static similar_emotion_pairs
- label name 到 anchor id 的映射
- L_sas
- hard anchor negative loss

先不要实现动态 mining。

---

### Step 4：日志和消融

最后实现：

- loss 分项打印
- domain weights 打印
- neutral F1
- 相似情绪混淆统计
- ablation config

---

## 15. 最小可行版本

如果代码结构复杂，优先实现这个最小版本：

```text
Base model
+ Neutral Branch
+ Speaker State Fusion into Domain Gate
+ Static SAS Loss
```

也就是说：

1. neutral 单独二分类；
2. speaker_state 编码后拼接进 domain gate；
3. 手动指定 similar emotion pairs；
4. 不实现动态 mining；
5. 不改变原有训练入口太多。

---

## 16. 需要特别注意的问题

### 16.1 neutral 样本不要进入非中性 anchor loss

如果使用 neutral decoupling，neutral 样本只参与：

```text
L_neu
L_final
```

不要参与：

```text
L_sas
L_hard
non-neutral supervised contrastive loss
```

否则会破坏非中性情绪锚点空间。

---

### 16.2 同一情绪内部子锚点不要互相排斥

不要把：

```text
happy_activation
happy_interaction
happy_expression
happy_context_shift
```

互相拉开。

SAS 只处理：

```text
不同情绪类别之间的相同域锚点
```

例如：

```text
happy_activation vs excited_activation
sad_context_shift vs frustrated_context_shift
```

---

### 16.3 新模块必须可关闭

所有新增模块必须通过 config 关闭，并且关闭后模型行为尽量接近原模型。

---

### 16.4 先保证跑通，再追求复杂

不要一开始实现动态相似对挖掘。先使用静态 pairs：

```yaml
similar_emotion_pairs:
  - ["happy", "excited"]
  - ["sad", "frustrated"]
```

跑通后再考虑根据验证集 confusion matrix 自动更新。

---

## 17. 预期改进点

该增强模型主要希望改善以下问题：

1. `neutral` 类别占比过高导致模型偏向 neutral；
2. `happy` 和 `excited` 容易混淆；
3. `sad` 和 `frustrated` 容易混淆；
4. interaction / context_shift 域缺少显式说话人状态信息；
5. 原 angle loss 对所有类别一视同仁，不能专门优化高混淆情绪对。

---

## 18. 最终论文描述对应关系

实现完成后，方法部分可以写成三个新增模块：

```text
1. 中性情绪解耦模块
2. 说话人状态引导的多域聚合模块
3. 相似锚点分离模块
```

核心公式对应：

```text
u_i = Fuse(h_i, r_i)
```

```text
P(y = neutral | x_i) = p_i^neu
P(y = c | x_i) = (1 - p_i^neu) * Σ_k g_{i,k} p_k(c | x_i)
```

```text
L_sas = Σ_{(c,d)∈S} Σ_k max(0, cos(v_{c,k}, v_{d,k}) - m)^2
```

总损失：

```text
L_total =
L_final
+ λ_neu L_neu
+ λ_supcon L_supcon
+ λ_angle L_angle
+ λ_sas L_sas
+ λ_hard L_hard
```

---

## 19. 完成标准

完成后请确保：

- 代码可以正常训练；
- 原模型配置仍然可以运行；
- 新模型配置可以运行；
- loss 中每一项都有日志；
- 关闭新增模块后不会报错；
- 缺失 speaker_state 字段时不会报错；
- similar_emotion_pairs 中不存在的标签会自动跳过；
- 输出 final_probs 的 shape 为 `[batch_size, num_labels]`；
- neutral decoupling 下的 label mapping 正确；
- 至少提供一个 `sas_nsg_eacl.yaml` 示例配置。
