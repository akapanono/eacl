# ERC 模型止损版改进方案

## 0. 当前结论

当前模型不是“模块太少”，而是**模块偏多、约束偏强、训练目标之间可能互相冲突**。因此，现阶段不建议继续加入新的复杂模块，例如 speaker memory、adaptive fusion、LLM knowledge、更复杂 graph module 或更强 hard negative。

当前最重要的任务是：

> **先回退到稳定结构，恢复指标，再用最小消融定位到底是哪一个模块导致变差。**

本文档的目标不是提出一个更复杂的新模型，而是给出一套**可直接让 Codex 修改和跑实验的止损方案**。

---

## 1. 当前模型为什么可能越改越差

当前模型已经包含：

```text
PLM Prompt Encoder
+ Emotion Anchor / Prototype Head
+ Multi-domain Subanchors
+ Domain-gated Pooling
+ Neutral Decoupling
+ Speaker State Fusion
+ Classifier-Prototype Fusion
+ SupCon Loss
+ Angle Loss
+ Similar Anchor Separation Loss
+ Hard Anchor Negative Loss
+ Gate Entropy Loss
+ Dynamic Prototype Momentum Update
```

这些模块单独看都有理论依据，但同时开启时，会带来几个问题。

### 1.1 辅助损失过多，目标冲突

当前 loss 可能包含：

```text
L_total =
  L_CE
  + lambda_neu * L_neutral
  + lambda_supcon * L_SupCon
  + lambda_angle * L_Angle
  + lambda_sas * L_SAS
  + lambda_hard * L_Hard
  - lambda_gate_entropy * H_gate
```

其中 CE 负责主任务分类，SupCon 负责表示空间聚类，SAS 和 hard negative 负责拉开相似情感，gate entropy 负责约束 domain gate。它们并不总是同向优化。如果 test 越改越差，最可能不是 CE 出问题，而是辅助约束太强。

### 1.2 Hard Negative 可能过强

Hard negative 的理论目标是区分：

```text
happy vs excited
angry vs frustrated
sad vs frustrated
```

但 ERC 中这些类别本身就存在语义重叠和标注模糊。过早、过强地把它们推远，可能导致：

```text
train 上区分更强
valid/test 上泛化更差
```

因此，hard negative 不适合作为默认强开启模块。

### 1.3 Domain-gated Subanchor 容易过拟合

当前 domain-gated subanchor 会根据每个样本动态选择 activation、interaction、expression、context_shift 等 domain 权重。这个想法有创新性，但 IEMOCAP、MELD 等 ERC 数据集规模有限，domain gate 容易学到数据集偏差。

如果出现 train F1 上升但 valid/test F1 不稳定，domain-gated pooling 是重点怀疑对象之一。

### 1.4 Dynamic Prototype Update 可能破坏初始语义 anchor

Emotion anchor 的优势在于利用 label semantics 作为先验。如果训练过程中持续用 batch centroid 更新 prototype，则可能出现：

```text
初始 anchor 有语义
训练后 anchor 被小 batch 噪声拉偏
```

尤其是在 batch_size 较小、类别不均衡、non-neutral 样本少的情况下，prototype drift 可能明显影响 test 泛化。

### 1.5 Speaker State 可能是噪声

如果数据中没有真实的 mental_state、interaction_relation、expression_style、context_shift，而是统一填充 `unknown.`，那么 speaker state 分支并没有提供有效信息，反而可能引入额外噪声。

因此，现阶段建议先关闭 speaker state，等主干稳定后再考虑重新设计 speaker memory。

---

## 2. 止损原则

### 原则一：先关复杂模块，不再加新模块

优先关闭：

```text
hard negative
SAS
domain_gated
speaker_state
dynamic prototype update
adaptive fusion
```

### 原则二：保留稳定主干

保留：

```text
PLM Prompt Encoder
+ Classifier Head
+ Prototype Head
+ Fixed Classifier-Prototype Fusion
+ Neutral Decoupling
+ 轻量 SupCon
```

### 原则三：先恢复指标，再追求创新

如果 test F1 已经下降，不要继续加理论模块，而是先恢复到历史最好附近。

### 原则四：一次只改一个变量

不要同时改 loss、pooling、prototype update、fusion、speaker information。否则无法判断到底哪个模块有效。

---

## 3. 推荐止损版模型结构

建议先使用以下结构：

```text
Input Dialogue
  ↓
Prompt Encoder / PLM
  ↓
Target Utterance Representation h_i
  ↓
Classifier Head ──────────────┐
                              ├── Fixed Fusion → Final Prediction
Prototype Head  ──────────────┘
  ↓
Neutral Decoupling Branch
```

保留内容：

```text
1. PLM 编码器
2. 普通 classifier head
3. prototype head
4. neutral decoupling
5. fixed classifier-prototype fusion
6. 轻量 SupCon
```

关闭内容：

```text
1. hard negative
2. SAS
3. domain-gated pooling
4. speaker state
5. dynamic prototype update
6. gate entropy
7. adaptive fusion
```

---

## 4. 推荐配置

### 4.1 模型配置

```yaml
model:
  use_neutral_decoupling: true

  use_classifier_prototype_fusion: true
  fusion_alpha: 0.7

  prototype_pooling: logsumexp

  use_domain_gated: false
  use_speaker_state: false
  use_similar_anchor_separation: false
  use_hard_anchor_negative: false

  freeze_prototype_epochs: 999
  normalize_prototypes_after_update: true
```

说明：

- `fusion_alpha = 0.7` 表示更依赖 classifier head；
- prototype head 只作为辅助语义先验；
- `prototype_pooling = logsumexp` 比 domain-gated 更稳；
- `freeze_prototype_epochs = 999` 等价于训练中基本不更新 prototype；
- 如果代码支持，建议直接加 `--disable_prototype_update`。

### 4.2 损失配置

```yaml
loss:
  lambda_neu: 0.2
  lambda_supcon: 0.1
  lambda_angle: 0.0
  lambda_sas: 0.0
  lambda_hard: 0.0
  lambda_gate_entropy: 0.0
```

如果 SupCon 仍然不稳定，可以继续降到：

```yaml
lambda_supcon: 0.05
```

如果模型退化成普通分类器，再尝试：

```yaml
lambda_angle: 0.005
```

但不要一开始就开 SAS 和 hard negative。

### 4.3 训练配置

```yaml
training:
  lr: 3e-5
  ptmlr: 3e-6
  dropout: 0.3

  batch_size: 8
  gradient_accumulation_steps: 2
  effective_batch_size: 16

  max_grad_norm: 0.5
  early_stopping_patience: 5
  save_best_metric: valid_fscore
```

如果显存允许，优先使用：

```yaml
batch_size: 16
gradient_accumulation_steps: 1
```

SupCon 对 batch 内样本分布比较敏感，因此 effective batch size 不建议太小。

---

## 5. 推荐实验顺序

不要一次跑 22 组。先跑下面 6 组。

| 编号 | 设置 | 目的 |
|---|---|---|
| R0 | 历史最好版本 | 找回基准 |
| R1 | 当前版本关闭 hard negative | 判断 hard negative 是否伤害泛化 |
| R2 | R1 继续关闭 SAS | 判断相似情感分离是否过强 |
| R3 | R2 将 domain_gated 改成 logsumexp | 判断 domain gate 是否过拟合 |
| R4 | R3 冻结 prototype update | 判断动态 anchor 是否拉坏语义 |
| R5 | R4 将 fusion_alpha 改为 0.7 | 判断 classifier 主导是否更稳 |

推荐执行顺序：

```text
第一步：跑 R0，确认历史最好分数
第二步：跑 R1，只关闭 hard negative
第三步：如果 R1 提升，说明 hard negative 是主要问题
第四步：如果 R1 没提升，跑 R2，继续关闭 SAS
第五步：如果 R2 还不行，跑 R3，把 domain_gated 换成 logsumexp
第六步：如果 R3 仍不稳，跑 R4，冻结 prototype
第七步：最后跑 R5，让 classifier head 主导预测
```

---

## 6. 建议命令模板

下面命令需要根据你代码里的参数名微调。

### 6.1 R1：关闭 hard negative

```bash
python src/run.py \
  --dataset IEMOCAP \
  --use_neutral_decoupling \
  --use_classifier_prototype_fusion \
  --fusion_alpha 0.5 \
  --prototype_pooling domain_gated \
  --lambda_hard 0.0 \
  --lambda_sas 0.002 \
  --lambda_supcon 0.2 \
  --lambda_neu 0.2
```

### 6.2 R2：关闭 hard negative + SAS

```bash
python src/run.py \
  --dataset IEMOCAP \
  --use_neutral_decoupling \
  --use_classifier_prototype_fusion \
  --fusion_alpha 0.5 \
  --prototype_pooling domain_gated \
  --lambda_hard 0.0 \
  --lambda_sas 0.0
```

### 6.3 R3：改成 logsumexp pooling

```bash
python src/run.py \
  --dataset IEMOCAP \
  --use_neutral_decoupling \
  --use_classifier_prototype_fusion \
  --fusion_alpha 0.5 \
  --prototype_pooling logsumexp \
  --lambda_hard 0.0 \
  --lambda_sas 0.0 \
  --lambda_gate_entropy 0.0
```

### 6.4 R4：冻结 prototype update

```bash
python src/run.py \
  --dataset IEMOCAP \
  --use_neutral_decoupling \
  --use_classifier_prototype_fusion \
  --fusion_alpha 0.5 \
  --prototype_pooling logsumexp \
  --freeze_prototype_epochs 999 \
  --lambda_hard 0.0 \
  --lambda_sas 0.0 \
  --lambda_gate_entropy 0.0
```

### 6.5 R5：classifier 主导融合

```bash
python src/run.py \
  --dataset IEMOCAP \
  --use_neutral_decoupling \
  --use_classifier_prototype_fusion \
  --fusion_alpha 0.7 \
  --prototype_pooling logsumexp \
  --freeze_prototype_epochs 999 \
  --lambda_supcon 0.1 \
  --lambda_neu 0.2 \
  --lambda_angle 0.0 \
  --lambda_hard 0.0 \
  --lambda_sas 0.0 \
  --lambda_gate_entropy 0.0
```

---

## 7. 给 Codex 的代码修改清单

### 7.1 增加关闭 prototype update 的参数

在 `src/run.py` 增加：

```python
parser.add_argument("--disable_prototype_update", action="store_true")
```

在训练时更新 prototype 的位置加入：

```python
if not args.disable_prototype_update:
    model.update_prototypes(...)
```

如果原来没有统一函数，就在所有调用 momentum update 的地方加判断。

### 7.2 增加稳定版配置 preset

在实验队列中增加一个 preset：

```python
rescue_stable = {
    "prototype_pooling": "logsumexp",
    "use_classifier_prototype_fusion": True,
    "fusion_alpha": 0.7,
    "use_neutral_decoupling": True,
    "use_speaker_state": False,
    "use_similar_anchor_separation": False,
    "use_hard_anchor_negative": False,
    "disable_prototype_update": True,
    "lambda_supcon": 0.1,
    "lambda_neu": 0.2,
    "lambda_angle": 0.0,
    "lambda_sas": 0.0,
    "lambda_hard": 0.0,
    "lambda_gate_entropy": 0.0,
    "max_grad_norm": 0.5,
    "dropout": 0.3,
}
```

### 7.3 确保关闭模块时 loss 真正为 0

检查 `src/model/loss.py`：

```python
if not args.use_hard_anchor_negative or args.lambda_hard <= 0:
    hard_loss = torch.tensor(0.0, device=device)

if not args.use_similar_anchor_separation or args.lambda_sas <= 0:
    sas_loss = torch.tensor(0.0, device=device)

if args.lambda_gate_entropy <= 0:
    gate_entropy_loss = torch.tensor(0.0, device=device)
```

避免出现“命令行关闭了模块，但 loss 内部仍然计算”的情况。

### 7.4 日志中必须记录每个 loss 分量

训练日志中输出：

```text
ce_loss
neutral_loss
supcon_loss
angle_loss
sas_loss
hard_loss
gate_entropy
total_loss
```

如果某个 loss 关闭，应显示为：

```text
0.0000
```

这样才能确认实验真的关闭成功。

### 7.5 增加 prototype drift 统计

在训练开始保存初始 anchor：

```python
init_anchor = model.emo_anchor.detach().clone()
```

每个 epoch 后计算：

```python
drift = 1 - cosine_similarity(
    init_anchor.flatten(1),
    current_anchor.flatten(1),
    dim=-1
).mean()
```

记录：

```text
prototype_drift
```

如果 drift 持续变大且 valid/test 下降，说明 prototype update 可能有害。

---

## 8. 判断结果的方法

### 8.1 如果关闭 hard negative 后 test 上升

说明 hard negative 过强。后续可以只把 hard negative 写成“后续可探索方向”，或者改成课程式 hard negative，但暂时不要作为主模型默认模块。

### 8.2 如果关闭 SAS 后 test 上升

说明相似 anchor separation 过强。后续可以降低到：

```yaml
lambda_sas: 0.0005
```

而不是使用：

```yaml
lambda_sas: 0.002
```

### 8.3 如果 domain_gated 改 logsumexp 后 test 上升

说明 domain gate 过拟合。此时论文里不要强调 domain-gated 是核心创新，可以改成：

```text
multi-subanchor logsumexp aggregation
```

这个说法更稳。

### 8.4 如果冻结 prototype 后 test 上升

说明动态 prototype update 破坏了初始语义 anchor。此时建议主模型使用 fixed semantic anchor，只把 momentum update 放到消融实验里。

### 8.5 如果 fusion_alpha = 0.7 后 test 上升

说明 classifier head 比 prototype head 更可靠。此时可以把 prototype 定位成：

```text
semantic regularizer / auxiliary semantic prior
```

而不是让 prototype 主导预测。

---

## 9. 最终推荐主模型

如果 R5 效果最好，建议最终模型写成：

```text
PLM Prompt Encoder
+ Fixed Emotion Semantic Anchors
+ LogSumExp Prototype Aggregation
+ Neutral Decoupling
+ Lightweight Supervised Contrastive Learning
+ Classifier-dominant Prototype Fusion
```

对应名称可以是：

```text
Stable Emotion-Anchored Contrastive Learning for ERC
```

中文可称为：

```text
稳定情感锚定对比学习模型
```

---

## 10. 论文中可以这样解释

```text
在初步实验中，直接叠加多子锚点、domain-gated 聚合、hard negative 和动态 prototype 更新虽然增强了模型表达能力，但也引入了额外的不稳定因素。由于 ERC 数据集规模有限，且 happy/excited、angry/frustrated 等类别之间存在天然语义重叠，过强的相似负样本约束可能导致模型在训练集上形成过硬的分类边界，从而影响测试集泛化。因此，本文最终采用较为稳定的 logsumexp prototype aggregation，并将 emotion anchor 作为语义先验辅助分类器，而非完全替代数据驱动的分类头。同时，针对 neutral 类别占比高、语义边界弱的问题，模型保留 neutral decoupling，以降低 neutral 类别对其他情感类别的干扰。
```

---

## 11. 理论依据对应关系

| 当前改动 | 理论依据 | 采用方式 |
|---|---|---|
| emotion anchor | EACL | 保留，但固定或弱更新 |
| SupCon | SACL / SupCon | 保留，但降低权重 |
| neutral decoupling | Decoupled Neutral Emotion | 保留 |
| hard negative | EACL / SACL | 暂时关闭或后续课程式开启 |
| speaker modeling | CoMPM / DialogueGCN | 暂不加入，作为后续工作 |
| domain-gated subanchor | mixture-like idea | 先关闭，改用 logsumexp |

---

## 12. 参考文献

1. Yu, F., Guo, J., Wu, Z., & Dai, X. (2024). **Emotion-Anchored Contrastive Learning Framework for Emotion Recognition in Conversation**. Findings of NAACL 2024.  
   https://aclanthology.org/2024.findings-naacl.282/

2. Hu, D., Bao, Y., Wei, L., Zhou, W., & Hu, S. (2023). **Supervised Adversarial Contrastive Learning for Emotion Recognition in Conversations**. ACL 2023.  
   https://aclanthology.org/2023.acl-long.606/

3. Kang, Y., & Cho, Y.-S. (2024). **Improving Contrastive Learning in Emotion Recognition in Conversation via Data Augmentation and Decoupled Neutral Emotion**. EACL 2024.  
   https://aclanthology.org/2024.eacl-long.134/

4. Lee, J., & Lee, W. (2022). **CoMPM: Context Modeling with Speaker's Pre-trained Memory Tracking for Emotion Recognition in Conversation**. NAACL 2022.  
   https://aclanthology.org/2022.naacl-main.416/

5. Khosla, P., Teterwak, P., Wang, C., et al. (2020). **Supervised Contrastive Learning**. NeurIPS 2020.  
   https://arxiv.org/abs/2004.11362

---

## 13. 最终执行建议

现阶段只做一件事：

```text
不要继续加模块。
先用 R1-R5 找出是哪一个模块导致 test 下降。
```

优先怀疑顺序：

```text
hard negative
→ SAS
→ domain_gated
→ prototype momentum update
→ speaker_state
→ fusion alpha
```

最终目标是先恢复一个稳定版本：

```text
PLM + neutral decoupling + classifier-prototype fusion + light SupCon + fixed/logsumexp prototype
```

等这个版本 test 回升后，再考虑是否把 SAS 或 hard negative 以很小权重加回来。
