# ERC 项目改动计划：聚类初始化域子锚点 + HMPEAE 风格超球面拉远

> 目标：在当前 EACL 主干上，新增一套可控、可消融的 **Cluster-initialized Hyperspherical Sub-anchor** 机制。  
> 核心改动：  
> 1. 子锚点不再只依赖情绪模板，而是根据训练集表示进行类别内聚类初始化。  
> 2. 参考 HMPEAE 的超球面多原型思想，对不同情感类别的子锚点进行 hard negative 拉远。  
> 3. 保留当前 EACL 的 CE + SupCon 主损失，不推翻原结构，只做轻量增强。

---

## 0. 当前项目背景

当前项目主体是 **Emotion-Anchored Contrastive Learning, EACL**，主流程大致是：

```text
Prompted dialogue input
    -> PLM encoder
    -> <mask> hidden state
    -> classifier logits
    -> map_function
    -> mapped utterance representation
    -> emotion-anchor supervised contrastive learning
    -> optional nearest-neighbour anchor prediction
    -> optional stage-2 anchor adaptation
```

当前已有结构：

```text
src/dataset.py
src/utils/data_process.py
src/model/model.py
src/model/loss.py
src/trainer/trainer.py
src/generate_anchors.py
src/run.py
run.sh
```

当前已有 anchor 相关机制：

```text
emotion anchors
num_subanchors
prototype_pooling = max / logsumexp / entropy / domain_gated
SupConLoss
AngleLoss
anchor dynamic update
stage-2 anchor adaptation
```

本次不要重新加入 speaker memory、adaptive fusion、hard negative queue、SAS/NSG 等大模块。  
本次只围绕 **domain sub-anchor** 做改动。

---

## 1. 总体改动目标

新增一个实验分支：

```text
Cluster-initialized Hyperspherical Sub-anchor ERC
```

中文描述：

```text
基于聚类初始化的超球面域子锚点对话情感识别模型
```

模型逻辑：

```text
先训练 baseline EACL
        ↓
提取训练集 mapped utterance embeddings
        ↓
按情感类别分别聚类
        ↓
每个类别得到 K 个 cluster center
        ↓
用 cluster center 初始化该情感类别的 K 个 domain sub-anchors
        ↓
继续训练时加入：
    1. 样本-子锚点紧致损失 L_pull
    2. 异类子锚点超球面分离损失 L_inter
    3. 可选：同类子锚点弱去塌缩损失 L_intra_div
```

---

## 2. 建议新增命令行参数

在 `src/run.py` 的参数解析中新增：

```python
parser.add_argument("--use_cluster_anchors", action="store_true",
                    help="Use cluster-initialized mapped-space sub-anchors.")

parser.add_argument("--cluster_anchor_path", type=str, default=None,
                    help="Path to cluster anchors saved by generate_cluster_anchors.py.")

parser.add_argument("--anchor_pull_weight", type=float, default=0.0,
                    help="Weight for sample-to-subanchor compactness loss.")

parser.add_argument("--hyp_inter_weight", type=float, default=0.0,
                    help="Weight for hyperspherical inter-class sub-anchor separation loss.")

parser.add_argument("--intra_div_weight", type=float, default=0.0,
                    help="Weight for optional intra-class sub-anchor diversity loss.")

parser.add_argument("--intra_same_upper", type=float, default=0.85,
                    help="Upper cosine similarity threshold for same-class sub-anchor diversity loss.")

parser.add_argument("--freeze_cluster_anchors", action="store_true",
                    help="Freeze cluster anchors during stage-1 training.")

parser.add_argument("--warmup_anchor_update_epochs", type=int, default=0,
                    help="Freeze anchor EMA update for first N epochs.")
```

第一版推荐先使用：

```bash
--use_cluster_anchors \
--cluster_anchor_path <path> \
--anchor_pull_weight 0.1 \
--hyp_inter_weight 0.05 \
--intra_div_weight 0.0 \
--disable_anchor_updates \
--prototype_pooling logsumexp \
--ce_loss_weight 0.3
```

---

## 3. 新增脚本：`src/generate_cluster_anchors.py`

### 3.1 目的

新增脚本用于生成基于训练集表示的聚类子锚点。

输入：

```text
baseline checkpoint
dataset_name
bert_path
anchor_path
num_subanchors
mapping_lower_dim
```

输出：

```text
cluster_anchors/{model_name}/{dataset_name}_cluster_{num_subanchors}.pt
```

保存内容建议：

```python
{
    "anchors": cluster_anchors,          # Tensor [num_classes, num_subanchors, mapping_lower_dim]
    "counts": cluster_counts,            # Tensor [num_classes, num_subanchors]
    "dataset_name": args.dataset_name,
    "num_subanchors": args.num_subanchors,
    "space": "mapped",
    "normalize": True
}
```

### 3.2 为什么使用 `mask_mapped_outputs`

不要用原始 `<mask>` hidden state 直接聚类，优先用当前模型的：

```python
mask_mapped_outputs
```

原因：

```text
1. 当前 SupCon 和 anchor similarity 本来就在 mapped space 中进行。
2. mapped space 维度更低，更适合聚类。
3. 后续 cluster anchors 可以直接参与余弦相似度计算。
```

### 3.3 脚本主要流程

伪代码：

```python
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

def extract_mapped_embeddings(model, train_loader, device):
    model.eval()
    all_reps = []
    all_labels = []

    with torch.no_grad():
        for sentences, labels in train_loader:
            sentences = sentences.to(device)
            labels = labels.to(device)

            feature, mask_mapped_outputs, mask_outputs, anchor_scores = model(
                sentences, return_mask_output=True
            )

            reps = F.normalize(mask_mapped_outputs, dim=-1)

            all_reps.append(reps.cpu())
            all_labels.append(labels.cpu())

    all_reps = torch.cat(all_reps, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_reps, all_labels


def build_cluster_anchors(all_reps, all_labels, num_classes, num_subanchors):
    anchors = []
    counts = []

    for c in range(num_classes):
        cls_reps = all_reps[all_labels == c]

        if len(cls_reps) == 0:
            raise ValueError(f"No samples found for class {c}")

        k = min(num_subanchors, len(cls_reps))

        # sklearn KMeans
        km = KMeans(n_clusters=k, random_state=0, n_init="auto")
        assign = km.fit_predict(cls_reps.numpy())
        centers = torch.tensor(km.cluster_centers_, dtype=torch.float)

        # 如果某类样本数小于 num_subanchors，则用已有 center 重复补齐
        if k < num_subanchors:
            repeat_num = num_subanchors - k
            pad = centers[:1].repeat(repeat_num, 1)
            centers = torch.cat([centers, pad], dim=0)

        centers = F.normalize(centers, dim=-1)
        anchors.append(centers)

        cls_counts = torch.zeros(num_subanchors, dtype=torch.long)
        for idx in range(k):
            cls_counts[idx] = int((assign == idx).sum())
        counts.append(cls_counts)

    anchors = torch.stack(anchors, dim=0)  # [C, K, D]
    counts = torch.stack(counts, dim=0)    # [C, K]
    return anchors, counts
```

### 3.4 可选：实现 spherical k-means

第一版可以先用普通 KMeans + normalize。  
如果效果一般，再改成 spherical k-means：

```text
每轮：
1. normalize samples
2. normalize centers
3. 根据 cosine similarity 分配样本
4. 每个簇中心取均值后 normalize
```

---

## 4. 修改 `src/model/model.py`

### 4.1 加载 cluster anchors

当前模型已经能加载 emotion anchors。新增逻辑：

```python
if args.use_cluster_anchors:
    obj = torch.load(args.cluster_anchor_path, map_location="cpu")

    if isinstance(obj, dict):
        cluster_anchors = obj["anchors"]
    else:
        cluster_anchors = obj

    # cluster anchors are already in mapped space
    self.cluster_anchors = nn.Parameter(
        cluster_anchors.float(),
        requires_grad=not args.freeze_cluster_anchors
    )
else:
    self.cluster_anchors = None
```

注意：

```text
cluster_anchors shape: [num_classes, num_subanchors, mapping_lower_dim]
```

它和原来的模板 anchor 不同，原来的模板 anchor 可能是：

```text
[num_classes, num_subanchors, hidden_dim]
```

所以 cluster anchors 不需要再经过 `map_function`。

### 4.2 统一获取当前训练使用的 anchor

建议在 `CLModel` 中加一个函数：

```python
def get_active_mapped_anchors(self):
    """
    Return anchors in mapped space.
    Shape: [num_classes, num_subanchors, mapping_lower_dim]
    """
    if self.cluster_anchors is not None:
        return F.normalize(self.cluster_anchors, dim=-1)

    # 否则走原来的 emotion anchor 映射逻辑
    # 注意根据当前代码实际 anchor 变量名修改
    mapped = self.map_function(self.emotion_anchors)
    return F.normalize(mapped, dim=-1)
```

如果原代码中的 emotion anchors 已经在 forward 里映射，则不要重复映射。  
Codex 修改时请先阅读当前 `src/model/model.py` 中 anchor 的实际变量名和映射逻辑，再最小化改动。

### 4.3 nearest-neighbour 预测使用 active anchors

如果启用了：

```bash
--use_nearest_neighbour
```

并且同时启用：

```bash
--use_cluster_anchors
```

则 anchor scores 应该基于 `cluster_anchors` 计算。

逻辑：

```python
reps = F.normalize(mask_mapped_outputs, dim=-1)
anchors = self.get_active_mapped_anchors()
scores = torch.einsum("bd,ckd->bck", reps, anchors)

if args.prototype_pooling == "max":
    anchor_scores = scores.max(dim=-1).values
elif args.prototype_pooling == "logsumexp":
    anchor_scores = torch.logsumexp(scores / temp, dim=-1) * temp
else:
    # 其他 pooling 先保持原逻辑或暂不支持 cluster anchors
```

第一版建议只支持：

```text
max
logsumexp
```

如果使用 `entropy` / `domain_gated` 且 `use_cluster_anchors=True`，可以先报错：

```python
raise NotImplementedError("cluster anchors currently support max/logsumexp only")
```

---

## 5. 修改 `src/model/loss.py`

新增三个 loss。

---

### 5.1 样本-子锚点紧致损失：`anchor_pull_loss`

作用：

```text
让样本表示靠近自己真实情感类别下最相似的 domain sub-anchor。
```

实现：

```python
import torch
import torch.nn.functional as F


def anchor_pull_loss(reps, labels, anchors):
    """
    reps:    [B, D]
    labels:  [B]
    anchors: [C, K, D]

    Return:
        scalar loss
    """
    reps = F.normalize(reps, dim=-1)
    anchors = F.normalize(anchors, dim=-1)

    cur_anchors = anchors[labels]  # [B, K, D]
    sim = torch.einsum("bd,bkd->bk", reps, cur_anchors)

    best_k = sim.argmax(dim=1)
    batch_idx = torch.arange(reps.size(0), device=reps.device)
    pos_anchor = cur_anchors[batch_idx, best_k]

    cos = F.cosine_similarity(reps, pos_anchor, dim=-1)
    loss = (1.0 - cos).mean()
    return loss
```

第一版先用 hard assignment。  
后续可扩展 soft assignment：

```python
q = torch.softmax(sim / temp, dim=-1)
loss = (q * (1.0 - sim)).sum(dim=-1).mean()
```

---

### 5.2 HMPEAE 风格异类子锚点拉远损失：`hyperspherical_inter_anchor_loss`

作用：

```text
参考 HMPEAE 的 L_inter，对每个子锚点寻找最接近的异类子锚点 hard negative，
并最小化它们的余弦相似度。
```

实现：

```python
def hyperspherical_inter_anchor_loss(anchors):
    """
    anchors: [C, K, D]

    Return:
        scalar loss
    """
    C, K, D = anchors.shape
    anchors = F.normalize(anchors, dim=-1)
    flat = anchors.reshape(C * K, D)

    sim = torch.matmul(flat, flat.t())  # [C*K, C*K]

    labels = torch.arange(C, device=anchors.device).repeat_interleave(K)
    diff_mask = labels[:, None] != labels[None, :]

    hardest_diff_sim = sim.masked_fill(~diff_mask, -1e4).max(dim=1).values

    return hardest_diff_sim.mean()
```

解释：

```text
如果某个 angry 子锚点和 frustrated 子锚点最接近，
这个损失会优先惩罚这对原型，使其在超球面空间中分离。
```

---

### 5.3 可选同类子锚点弱去塌缩损失：`intra_anchor_diversity_loss`

作用：

```text
防止同一情感类别内部的多个子锚点完全重合。
```

注意：

```text
不要强行把同类子锚点拉得很远。
只需要限制它们相似度不要过高。
```

实现：

```python
def intra_anchor_diversity_loss(anchors, same_upper=0.85):
    """
    anchors: [C, K, D]
    same_upper: if same-class sub-anchor cosine similarity > same_upper, penalize it.

    Return:
        scalar loss
    """
    C, K, D = anchors.shape

    if K <= 1:
        return anchors.new_tensor(0.0)

    anchors = F.normalize(anchors, dim=-1)
    flat = anchors.reshape(C * K, D)
    sim = torch.matmul(flat, flat.t())

    labels = torch.arange(C, device=anchors.device).repeat_interleave(K)
    same_mask = labels[:, None] == labels[None, :]

    eye = torch.eye(C * K, dtype=torch.bool, device=anchors.device)
    same_mask = same_mask & (~eye)

    same_sim = sim[same_mask]

    if same_sim.numel() == 0:
        return anchors.new_tensor(0.0)

    return F.relu(same_sim - same_upper).mean()
```

第一版可以不启用：

```bash
--intra_div_weight 0.0
```

等实验 3 稳定后再试：

```bash
--intra_div_weight 0.01
--intra_same_upper 0.85
```

---

## 6. 修改 `src/trainer/trainer.py`

### 6.1 在原 loss 后追加新 loss

当前原始 loss 类似：

```python
loss = ce_loss * args.ce_loss_weight + (1 - args.ce_loss_weight) * cl_loss
```

修改为：

```python
loss = ce_loss * args.ce_loss_weight + (1 - args.ce_loss_weight) * cl_loss

extra_loss_dict = {}

if args.use_cluster_anchors:
    anchors = model.get_active_mapped_anchors()
    reps = mask_mapped_outputs

    if args.anchor_pull_weight > 0:
        pull = anchor_pull_loss(reps, labels, anchors)
        loss = loss + args.anchor_pull_weight * pull
        extra_loss_dict["pull"] = float(pull.detach().cpu())

    if args.hyp_inter_weight > 0:
        inter = hyperspherical_inter_anchor_loss(anchors)
        loss = loss + args.hyp_inter_weight * inter
        extra_loss_dict["inter"] = float(inter.detach().cpu())

    if args.intra_div_weight > 0:
        intra = intra_anchor_diversity_loss(
            anchors,
            same_upper=args.intra_same_upper
        )
        loss = loss + args.intra_div_weight * intra
        extra_loss_dict["intra_div"] = float(intra.detach().cpu())
```

需要在文件顶部引入：

```python
from src.model.loss import (
    anchor_pull_loss,
    hyperspherical_inter_anchor_loss,
    intra_anchor_diversity_loss,
)
```

具体 import 路径根据当前项目实际写法调整。

### 6.2 日志记录

建议每个 epoch 统计：

```text
avg_ce_loss
avg_cl_loss
avg_pull_loss
avg_inter_loss
avg_intra_div_loss
```

至少打印：

```text
Epoch X | ce=... cl=... pull=... inter=... intra=...
```

---

## 7. 修复梯度累积逻辑

当前报告中提到，梯度累积可能按：

```python
batch_id % accumulation_step == 0
```

触发 optimizer step，这会导致第 0 个 batch 就 step，不是标准写法。

建议统一修成：

```python
loss = loss / args.accumulation_step
loss.backward()

if (batch_id + 1) % args.accumulation_step == 0 or (batch_id + 1) == len(train_loader):
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

如果当前项目没有启用 accumulation，可以先保持兼容，但不要破坏原训练。

---

## 8. Anchor 动态更新策略

第一版实验建议直接禁用：

```bash
--disable_anchor_updates
```

原因：

```text
cluster anchors 是从训练集整体表示聚类得到的，
如果一开始就用 batch EMA 更新，容易被少数 batch 噪声带偏。
```

如果后续需要启用动态更新，建议加 warmup：

```python
if epoch < args.warmup_anchor_update_epochs:
    skip update_anchors()
else:
    update_anchors()
```

并且只用高置信样本更新：

```python
prob = torch.softmax(logits, dim=-1)
conf = prob.max(dim=-1).values
mask = conf > 0.7
```

这个可以作为后续增强，不作为第一版必须内容。

---

## 9. 实验顺序

不要一次性加完所有功能。按下面顺序跑。

---

### 实验 0：当前 baseline

目的：确认当前回退版本稳定。

```bash
bash run.sh IEMOCAP ./pretrained/sup-simcse-roberta-large
```

记录：

```text
valid weighted F1
test weighted F1
best epoch
是否出现 nan
```

---

### 实验 1：只换聚类子锚点，不加新 loss

先训练 baseline 并保存 checkpoint，然后生成 cluster anchors。

示例：

```bash
python src/generate_cluster_anchors.py \
  --bert_path ./pretrained/sup-simcse-roberta-large \
  --dataset_name IEMOCAP \
  --checkpoint_path saved_models/IEMOCAP/model_.pkl \
  --anchor_path ./emo_anchors/sup-simcse-roberta-large \
  --num_subanchors 3 \
  --output_dir ./cluster_anchors/sup-simcse-roberta-large
```

然后训练：

```bash
CUDA_VISIBLE_DEVICES=3 python src/run.py \
  --bert_path ./pretrained/sup-simcse-roberta-large \
  --dataset_name IEMOCAP \
  --anchor_path ./emo_anchors/sup-simcse-roberta-large \
  --use_cluster_anchors \
  --cluster_anchor_path ./cluster_anchors/sup-simcse-roberta-large/IEMOCAP_cluster_3.pt \
  --num_subanchors 3 \
  --prototype_pooling logsumexp \
  --ce_loss_weight 0.3 \
  --disable_anchor_updates \
  --use_nearest_neighbour
```

目的：

```text
观察聚类初始化本身是否优于模板初始化。
```

---

### 实验 2：聚类子锚点 + L_pull

```bash
CUDA_VISIBLE_DEVICES=3 python src/run.py \
  --bert_path ./pretrained/sup-simcse-roberta-large \
  --dataset_name IEMOCAP \
  --anchor_path ./emo_anchors/sup-simcse-roberta-large \
  --use_cluster_anchors \
  --cluster_anchor_path ./cluster_anchors/sup-simcse-roberta-large/IEMOCAP_cluster_3.pt \
  --num_subanchors 3 \
  --prototype_pooling logsumexp \
  --ce_loss_weight 0.3 \
  --anchor_pull_weight 0.1 \
  --hyp_inter_weight 0.0 \
  --disable_anchor_updates \
  --use_nearest_neighbour
```

目的：

```text
观察样本向本类最近子锚点聚集是否提升结果。
```

---

### 实验 3：聚类子锚点 + L_pull + HMPEAE-style L_inter

这是主实验。

```bash
CUDA_VISIBLE_DEVICES=3 python src/run.py \
  --bert_path ./pretrained/sup-simcse-roberta-large \
  --dataset_name IEMOCAP \
  --anchor_path ./emo_anchors/sup-simcse-roberta-large \
  --use_cluster_anchors \
  --cluster_anchor_path ./cluster_anchors/sup-simcse-roberta-large/IEMOCAP_cluster_3.pt \
  --num_subanchors 3 \
  --prototype_pooling logsumexp \
  --ce_loss_weight 0.3 \
  --anchor_pull_weight 0.1 \
  --hyp_inter_weight 0.05 \
  --disable_anchor_updates \
  --use_nearest_neighbour
```

目的：

```text
验证参考 HMPEAE 的异类原型 hard negative 拉远是否有效。
```

---

### 实验 4：加入弱同类去塌缩

```bash
CUDA_VISIBLE_DEVICES=3 python src/run.py \
  --bert_path ./pretrained/sup-simcse-roberta-large \
  --dataset_name IEMOCAP \
  --anchor_path ./emo_anchors/sup-simcse-roberta-large \
  --use_cluster_anchors \
  --cluster_anchor_path ./cluster_anchors/sup-simcse-roberta-large/IEMOCAP_cluster_3.pt \
  --num_subanchors 3 \
  --prototype_pooling logsumexp \
  --ce_loss_weight 0.3 \
  --anchor_pull_weight 0.1 \
  --hyp_inter_weight 0.05 \
  --intra_div_weight 0.01 \
  --intra_same_upper 0.85 \
  --disable_anchor_updates \
  --use_nearest_neighbour
```

目的：

```text
防止同类子锚点塌缩到一起。
```

---

## 10. 需要输出的调试信息

请 Codex 帮忙在训练日志中增加以下信息。

### 10.1 子锚点相似度统计

每个 epoch 打印一次：

```text
avg_same_class_anchor_cos
max_same_class_anchor_cos
avg_diff_class_anchor_cos
max_diff_class_anchor_cos
```

其中：

```text
max_diff_class_anchor_cos 越低，说明异类子锚点拉得越开。
max_same_class_anchor_cos 太高，说明同类子锚点可能塌缩。
```

### 10.2 子锚点分配数量

在 `anchor_pull_loss` 中可以统计每个样本选择了哪个子锚点。  
每个 epoch 输出：

```text
class 0 subanchor counts: [xx, xx, xx]
class 1 subanchor counts: [xx, xx, xx]
...
```

目的：

```text
判断是否所有样本都挤到一个子锚点。
```

第一版可以先不实现完整统计，但建议预留函数：

```python
def compute_subanchor_assignment_counts(reps, labels, anchors):
    ...
```

---

## 11. 可接受的第一版完成标准

第一版不要求一次性达到最优结果，只要求功能闭环。

必须满足：

```text
1. generate_cluster_anchors.py 可以正常生成 [C, K, D] 的 cluster anchors。
2. use_cluster_anchors=True 时，模型可以正常加载 cluster anchors。
3. cluster anchors 在 mapped space 中使用，不重复经过 map_function。
4. anchor_pull_loss 可以正常反向传播。
5. hyperspherical_inter_anchor_loss 可以正常反向传播。
6. 不出现 nan / inf。
7. 原 baseline 路径不受影响，即不加新参数时仍能按原逻辑运行。
8. max/logsumexp pooling 至少有一个能和 cluster anchors 正常配合。
```

建议额外满足：

```text
1. 训练日志显示 pull/inter loss。
2. 能输出 anchor 相似度统计。
3. 能记录每组实验的 best valid / test weighted F1。
```

---

## 12. 代码实现注意事项

### 12.1 不要破坏原逻辑

新增参数默认都应该是关闭状态：

```text
use_cluster_anchors = False
anchor_pull_weight = 0.0
hyp_inter_weight = 0.0
intra_div_weight = 0.0
```

这样旧命令不受影响。

### 12.2 注意维度

模板 anchors 通常是：

```text
[C, K, hidden_dim]
```

cluster anchors 是：

```text
[C, K, mapping_lower_dim]
```

不要把 cluster anchors 再传入 `map_function`。

### 12.3 注意 normalize

所有用于 cosine similarity 的表示都要：

```python
F.normalize(x, dim=-1)
```

包括：

```text
mask_mapped_outputs
cluster_anchors
mapped emotion anchors
```

### 12.4 注意 label 类型

loss 中的 labels 必须是：

```python
labels.long()
```

并且不包含 ignore label：

```text
-1
```

如果存在 `-1`，需要 mask 掉。

### 12.5 注意 device

加载 cluster anchors 后要进入模型参数或 buffer，确保和模型同 device。  
如果使用 `nn.Parameter`，`.to(device)` 会自动转移。

### 12.6 注意 sklearn 依赖

如果项目环境没有 sklearn，需要新增依赖：

```bash
pip install scikit-learn
```

也可以改成纯 PyTorch spherical k-means，避免新依赖。

---

## 13. 论文写法参考

可以把这部分写成方法章节中的一个小节：

```text
3.X Cluster-initialized Domain Sub-anchor Learning
```

示例表述：

```text
为缓解对话情感识别中同一情感类别内部表达差异较大、不同情感类别边界模糊的问题，本文提出一种基于聚类初始化的超球面域子锚点学习方法。首先，利用基础编码器提取训练集中目标话语的映射表示，并在每个情感类别内部进行聚类，以聚类中心初始化该类别的多个域子锚点。随后，将话语表示和域子锚点归一化到超球面空间中，通过样本—子锚点紧致性损失增强类内子域聚合。进一步地，参考超球面多原型方法中的类间分离思想，本文对每个子锚点选取最相近的异类子锚点作为 hard negative，并最小化其余弦相似度，从而扩大不同情感类别子锚点之间的决策间隔。
```

消融实验可以设计为：

```text
w/o Cluster Init
w/o Pull Loss
w/o Hyperspherical Inter-anchor Loss
w/o Intra-domain Diversity
```

---

## 14. 最终建议优先实现顺序

请 Codex 按这个顺序改：

```text
1. 新增 loss 函数：
   - anchor_pull_loss
   - hyperspherical_inter_anchor_loss
   - intra_anchor_diversity_loss

2. 新增 CLModel.get_active_mapped_anchors()

3. 修改 forward / nearest-neighbour 分支，使 cluster anchors 可用于 max/logsumexp pooling

4. 修改 trainer，把新 loss 接入原 loss

5. 新增 generate_cluster_anchors.py

6. 新增 run.py 参数

7. 修复梯度累积逻辑

8. 增加训练日志中的 pull/inter/intra loss

9. 跑实验 1、2、3
```

第一版最小可用实验命令：

```bash
python src/generate_cluster_anchors.py \
  --bert_path ./pretrained/sup-simcse-roberta-large \
  --dataset_name IEMOCAP \
  --checkpoint_path saved_models/IEMOCAP/model_.pkl \
  --anchor_path ./emo_anchors/sup-simcse-roberta-large \
  --num_subanchors 3 \
  --output_dir ./cluster_anchors/sup-simcse-roberta-large

CUDA_VISIBLE_DEVICES=3 python src/run.py \
  --bert_path ./pretrained/sup-simcse-roberta-large \
  --dataset_name IEMOCAP \
  --anchor_path ./emo_anchors/sup-simcse-roberta-large \
  --use_cluster_anchors \
  --cluster_anchor_path ./cluster_anchors/sup-simcse-roberta-large/IEMOCAP_cluster_3.pt \
  --num_subanchors 3 \
  --prototype_pooling logsumexp \
  --ce_loss_weight 0.3 \
  --anchor_pull_weight 0.1 \
  --hyp_inter_weight 0.05 \
  --disable_anchor_updates \
  --use_nearest_neighbour
```
