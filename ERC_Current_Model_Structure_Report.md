# ERC 当前模型结构报告

生成时间：2026-05-25  
当前代码版本：`a395917 回退` + 当前工作区域锚点更新  
当前状态：已回退到“大改动之前 / 改进1”附近的 EACL 结构，并重新设计了 emotion anchor 的域划分与标准锚点。

## 1. 总体结论

当前项目主体是 **Emotion-Anchored Contrastive Learning, EACL**。模型不是复杂的多模块融合结构，而是一个相对清晰的两阶段框架：

1. **阶段一：PLM 表征学习 + 情绪锚点对比学习**
   - 用预训练语言模型编码带上下文的对话轮次。
   - 取 prompt 中 `<mask>` 位置的 hidden state 作为当前 utterance 表征。
   - 一条分支经过线性分类头输出 emotion logits。
   - 另一条分支映射到 anchor/prototype 空间，与情绪锚点做 supervised contrastive learning。

2. **阶段二：Emotion Anchor Adaptation**
   - 阶段一结束后，提取 train/dev/test 的 mapped utterance embeddings。
   - 使用训练好的 emotion anchors 初始化一个轻量 `Classifier`。
   - 单独训练该 prototype classifier，进一步提升 anchor 的分类能力。

当前版本没有启用之前大改动中的 speaker memory、adaptive fusion、neutral-aware SupCon、hard negative、SAS/NSG 队列脚本等结构。

## 2. 数据输入与 Prompt 构造

相关文件：

- `src/dataset.py`
- `src/utils/data_process.py`

支持数据集：

- `IEMOCAP`：6 类
- `MELD`：7 类
- `EmoryNLP`：7 类

每个样本对应一个目标 utterance。数据处理逻辑如下：

1. 读取对话。
2. 对每轮构造：

   ```text
   speaker says: utterance_text
   ```

3. 对目标轮次之前的上下文进行截断拼接，最多受 `max_len` 和过去窗口约束。
4. 对目标 utterance 构造 prompt：

   ```text
   For utterance: {text} {speaker} feels <mask>
   ```

5. 最终输入是：

   ```text
   [历史上下文] + For utterance: ... feels <mask>
   ```

6. `DialogueDataset.__getitem__()` 返回：

   ```python
   input_ids, label
   ```

当前版本没有 speaker state / speaker memory 额外输入。

## 3. 主模型 CLModel

相关文件：

- `src/model/model.py`

核心类：

```python
class CLModel(nn.Module)
```

### 3.1 PLM 编码器

模型使用 HuggingFace `AutoModel`：

```python
self.f_context_encoder = AutoModel.from_pretrained(args.bert_path, local_files_only=True)
```

默认运行脚本中使用：

```bash
./pretrained/sup-simcse-roberta-large
```

编码后取 `<mask>` 位置 hidden state：

```python
mask_outputs = utterance_encoded[batch_index, mask_pos]
```

这个 `mask_outputs` 是当前 utterance 的主语义表征，维度通常为 1024。

### 3.2 分类头

分类头很轻：

```python
self.predictor = nn.Sequential(
    nn.Linear(self.dim, self.num_classes)
)
```

流程：

```text
mask_outputs
  -> dropout
  -> Linear(dim, num_classes)
  -> logits
```

如果不使用 nearest-neighbour prototype 推理，最终预测来自这个分类头。

### 3.3 Anchor 映射层

模型还有一个映射函数，把 PLM 表征映射到 prototype/anchor 空间：

```python
self.map_function = nn.Sequential(
    nn.Linear(self.dim, self.dim),
    nn.LayerNorm(self.dim),
    nn.ReLU(),
    nn.Linear(self.dim, args.mapping_lower_dim),
)
```

用途：

- 将 utterance 表征映射成 `mask_mapped_outputs`
- 将 emotion anchors 映射到同一空间
- 用于 supervised contrastive loss
- 用于 nearest-neighbour prototype 分类

## 4. Emotion Anchors / Prototypes

相关文件：

- `src/generate_anchors.py`
- `src/model/anchor_utils.py`
- `src/model/model.py`

### 4.1 Anchor 生成

运行入口：

```bash
python src/generate_anchors.py --bert_path <model_path> --num_subanchors <N>
```

每个 emotion 会有若干自然语言模板，例如：

```text
The speaker feels angry and tense.
The speaker feels sad and down.
```

这些模板经过同一个 PLM 编码，得到 emotion anchor。

保存格式：

```text
emo_anchors/{model_name}/{dataset_name}_emo_{num_subanchors}.pt
emo_anchors/{model_name}/{dataset_name}_emo.pt
```

其中：

- `{dataset_name}_emo_{num_subanchors}.pt` 保存多域 sub-anchor。
- `{dataset_name}_emo.pt` 保存标准锚点。

当前标准锚点不再由多个域锚点简单平均得到，而是由独立的 canonical emotion descriptions 生成。这样标准锚点更接近类别中心，域锚点则负责表达情绪在不同维度上的变化。

### 4.2 Anchor Tensor 形状

加载后统一为：

```text
[num_classes, num_subanchors, hidden_dim]
```

例如 IEMOCAP、`num_subanchors=5`：

```text
[6, 5, 1024]
```

### 4.3 Sub-anchor 聚合

当前支持：

```bash
--prototype_pooling max
--prototype_pooling logsumexp
--prototype_pooling entropy
--prototype_pooling domain_gated
```

默认是：

```bash
--prototype_pooling max
```

不同模式含义：

- `max`：每类多个 sub-anchor 取最大相似度。
- `logsumexp`：对多个 sub-anchor 做平滑聚合。
- `entropy`：根据 sub-anchor/domain 分布熵计算权重。
- `domain_gated`：使用 `domain_gate` 学习不同 domain/sub-anchor 的权重。

注意：`entropy` 和 `domain_gated` 要求：

```bash
--num_subanchors 5
```

当前 5 个域为：

```text
valence
arousal
dominance_control
social_appraisal
discourse_context
```

并且默认会关闭第二阶段训练，除非显式加：

```bash
--force_two_stage
```

## 5. Domain-aware 组件

当前代码保留了较早版本中的 domain-aware prototype 支持，但不是默认主路径。

相关结构：

```python
self.domain_adapters
self.domain_gate
```

`domain_adapters` 为每个 sub-anchor/domain 准备一个映射器。当前每个 emotion 有 5 个 domain anchors，分别对应 valence、arousal、dominance/control、social appraisal 和 discourse context。  
`domain_gate` 根据当前 utterance 表征生成 domain 权重。

当：

```bash
--prototype_pooling domain_gated
```

时，模型走 `domain_gated_scores()` 分支。

否则主流程仍是普通：

```text
mask_outputs -> map_function -> anchor similarity
```

## 6. Forward 输出

`CLModel.forward(sentences, return_mask_output=True)` 返回：

```python
feature, mask_mapped_outputs, mask_outputs, anchor_scores
```

含义：

- `feature`
  - 默认是分类头 logits。
  - 若 `prototype_pooling == entropy` 且启用 nearest-neighbour，则可能替换为 anchor scores。

- `mask_mapped_outputs`
  - utterance 在 anchor 空间中的表示。
  - 用于 SupCon loss 和二阶段缓存。

- `mask_outputs`
  - PLM 原始 `<mask>` hidden state。
  - 用于动态更新 anchors。

- `anchor_scores`
  - utterance 与 emotion anchors 的相似度分数。
  - 只有 `--use_nearest_neighbour` 时用于预测。

## 7. Loss 结构

相关文件：

- `src/model/loss.py`
- `src/trainer/trainer.py`

总损失在 trainer 中组合：

```python
loss = ce_loss * ce_loss_weight + (1 - ce_loss_weight) * cl_loss
```

默认：

```bash
--ce_loss_weight 0.1
```

也就是：

```text
0.1 * CrossEntropy + 0.9 * ContrastiveLoss
```

### 7.1 Cross Entropy

分类头输出 logits，与真实 emotion label 做 CE：

```python
nn.CrossEntropyLoss(ignore_index=-1)
```

可选类别均衡：

```bash
--class_balanced_ce
```

启用后根据训练集标签频率设置 class weights。

### 7.2 SupConLoss

对比学习输入包括：

```text
utterance mapped representations
emotion anchors
```

如果没有禁用 anchors：

```bash
--disable_emo_anchor
```

则会把 utterance representations 和 flattened anchors 拼接：

```python
concated_reps = torch.cat([reps, flat_anchor], dim=0)
concated_labels = torch.cat([labels, anchor_labels], dim=0)
```

正样本：

```text
label 相同的 utterance / anchor
```

负样本：

```text
label 不同的 utterance / anchor
```

相似度函数：

```python
(1 + cosine_similarity(x, y)) / 2 + eps
```

### 7.3 AngleLoss

为了让不同类的 class anchors 在空间里更分离，SupConLoss 中还加入了 anchor center 的角度损失：

```python
loss += args.angle_loss_weight * angleloss
```

默认：

```bash
--angle_loss_weight 1.0
```

运行脚本 `run.sh` 中设置为：

```bash
ang_weight=0.1
```

## 8. Anchor 动态更新

阶段一训练时，如果没有设置：

```bash
--disable_anchor_updates
```

则每个 batch 后会调用：

```python
model.update_anchors(raw_reps, label)
```

更新逻辑：

1. 取当前 batch 的 raw `<mask>` outputs。
2. 通过 `map_function` 映射。
3. 对每个类别，找该类别样本最接近的 sub-anchor。
4. 用 momentum 更新原始 anchor：

```python
anchor = momentum * anchor + (1 - momentum) * class_centroid
```

默认：

```bash
--prototype_momentum 0.9
```

## 9. 训练流程

相关文件：

- `src/run.py`
- `src/trainer/trainer.py`
- `run.sh`

### 9.1 阶段一

每个 epoch：

1. train
2. dev
3. test
4. 记录 weighted F1
5. 保存 best checkpoint

保存路径：

```text
saved_models/{dataset_name}/model_.pkl
```

早停可选：

```bash
--early_stop_patience
--early_stop_metric valid|test
```

### 9.2 阶段二

如果没有设置：

```bash
--disable_two_stage_training
```

则进入第二阶段：

1. 加载阶段一 best model。
2. 提取 train/dev/test 的 mapped embeddings。
3. 构造 `Classifier(args, anchors)`。
4. 用 anchors 作为可训练参数进行 10 epoch 分类器训练。

二阶段分类器本质是：

```python
class Classifier(nn.Module):
    self.weight = nn.Parameter(anchors)
```

预测时计算输入 embedding 和每个 anchor 的 cosine similarity，再按 sub-anchor 聚合。

## 10. 当前默认运行方式

当前 `run.sh`：

```bash
bash run.sh IEMOCAP ./pretrained/sup-simcse-roberta-large
```

脚本内部关键参数：

```bash
ce_loss_weight=0.1
tmp=0.1
ang_weight=0.1
stage_two_lr=1e-4
seed=1
num_subanchors=5
```

实际运行会先生成 anchors：

```bash
python src/generate_anchors.py --bert_path $model_path --num_subanchors 5
```

然后后台启动训练：

```bash
CUDA_VISIBLE_DEVICES=3 python src/run.py \
  --anchor_path "./emo_anchors/${dir_name}" \
  --bert_path $model_path \
  --dataset_name $dataset \
  --ce_loss_weight 0.1 \
  --temp 0.1 \
  --seed 1 \
  --angle_loss_weight 0.1 \
  --stage_two_lr 1e-4 \
  --num_subanchors 5 \
  --disable_training_progress_bar \
  --use_nearest_neighbour
```

## 11. 当前仍保留的主要开关

### 模型与训练

| 参数 | 作用 | 默认 |
|---|---|---|
| `--bert_path` | PLM 路径 | `./pretrained/sup-simcse-roberta-large` |
| `--dataset_name` | 数据集 | `IEMOCAP` |
| `--epochs` | 阶段一 epoch 数 | `8` |
| `--batch_size` | batch size | `8` |
| `--lr` | 非 PLM 参数学习率 | `4e-4` |
| `--ptmlr` | PLM 学习率 | `1e-5` |
| `--dropout` | dropout | `0.1` |
| `--max_grad_norm` | 梯度裁剪 | `5.0` |

### Loss

| 参数 | 作用 | 默认 |
|---|---|---|
| `--ce_loss_weight` | CE 与 CL 的混合比例 | `0.1` |
| `--angle_loss_weight` | anchor angle loss 权重 | `1.0` |
| `--temp` | contrastive / prototype 温度 | `0.5` |
| `--class_balanced_ce` | 是否启用类别均衡 CE | 关闭 |

### Prototype / Anchor

| 参数 | 作用 | 默认 |
|---|---|---|
| `--anchor_path` | anchor 文件目录 | 无 |
| `--num_subanchors` | 每类 sub-anchor 数 | `1` |
| `--prototype_pooling` | sub-anchor 聚合方式 | `max` |
| `--prototype_momentum` | 动态更新 momentum | `0.9` |
| `--disable_anchor_updates` | 禁止训练中更新 anchors | 关闭 |
| `--disable_emo_anchor` | SupCon 中不拼接 emotion anchors | 关闭 |
| `--use_nearest_neighbour` | 用 anchor scores 做预测 | 关闭 |

### 二阶段

| 参数 | 作用 | 默认 |
|---|---|---|
| `--disable_two_stage_training` | 禁用二阶段 | 关闭 |
| `--stage_two_lr` | 二阶段分类器学习率 | `1e-4` |
| `--force_two_stage` | entropy/domain_gated 下强制二阶段 | 关闭 |
| `--save_stage_two_cache` | 保存二阶段 embedding cache | 关闭 |

## 12. 与之前大改动版本的区别

当前版本没有以下结构：

- speaker memory
- speaker state encoder
- classifier-prototype adaptive fusion
- neutral-aware SupCon
- hard negative schedule
- SAS / NSG 实验队列
- rescue 实验集
- confusion matrix 自动导出
- 多组参数后台队列脚本

也就是说，当前结构更接近原始 EACL 主干：

```text
Prompted dialogue input
  -> PLM
  -> <mask> representation
  -> classifier logits
  -> mapped representation
  -> emotion-anchor contrastive learning
  -> optional nearest-neighbour anchor prediction
  -> optional stage-2 anchor adaptation
```

## 13. 当前结构的优点与风险

### 优点

- 结构简单，变量少，便于定位实验结果变化。
- Anchor 与 SupCon 是核心贡献，当前版本保留了主逻辑。
- 二阶段 anchor adaptation 可单独观察对结果的提升。
- `num_subanchors` 和 `prototype_pooling` 仍可做轻量实验。

### 风险

- 当前 prompt 只显式利用文本上下文，没有显式建模 speaker 状态。
- Anchor 动态更新可能带来漂移，尤其在少数类上。
- `use_nearest_neighbour` 会让预测依赖 anchor 空间质量，而不是分类头。
- `accumulation_step` 当前实现是按 `batch_id % accumulation_step == 0` 触发 optimizer step，严格来说第 0 个 batch 就会 step，一般不是标准梯度累积写法。
- `run.sh` 固定 `CUDA_VISIBLE_DEVICES=3`，如果机器 GPU 编号不同，需要手动改。

## 14. 建议后续实验从哪里开始

如果目标是恢复稳定结果，建议先只动少数参数：

1. 固定当前结构，不再加入新模块。
2. 先比较：

   ```bash
   --use_nearest_neighbour
   ```

   开与关的差别。

3. 比较：

   ```bash
   --disable_anchor_updates
   ```

   开与关的差别，判断 prototype 动态更新是否带来漂移。

4. 保持 `num_subanchors=5`，只比较：

   ```bash
   --prototype_pooling max
   --prototype_pooling logsumexp
   ```

5. 暂时不要重新加入 speaker memory / adaptive fusion / hard negative。
