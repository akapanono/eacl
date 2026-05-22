# ERC 模型 NaN / 指数爆炸改进方案

## 1. 问题判断

你现在的现象是：

```text
第 1～2 个 epoch 效果很好；
继续训练后效果越来越差；
后期 loss 或指标出现 inf / NaN。
```

这通常不是普通的过拟合，而是训练过程出现了数值不稳定。结合当前模型结构，最可疑的位置是：

1. `Hard Anchor Negative Loss` 中手动使用 `exp(logits)`；
2. `Similar Anchor Separation Loss` 没有严格归一化；
3. `Neutral Decoupling` 中对 `final_probs` 直接取 `log`；
4. 某些 batch 没有非中性样本，导致空 tensor 求 mean；
5. 锚点动量更新导致 anchor norm 变大或语义漂移；
6. speaker state 融合后没有 LayerNorm，导致 domain gate 输入尺度失控。

本文件给 Codex 的目标是：在不推翻现有模型结构的前提下，修复训练不稳定问题。

---

## 2. 总体修改原则

### 2.1 先稳定，再提升

不要一开始就把所有增强模块开到最大。先保证模型能完整训练完，再逐步增加模块权重。

### 2.2 禁止直接使用高风险公式

尽量避免：

```python
torch.exp(logits)
torch.log(probs)
empty_tensor.mean()
```

优先使用：

```python
torch.logsumexp()
F.cross_entropy()
F.binary_cross_entropy_with_logits()
torch.clamp()
```

### 2.3 所有新增模块都要可关闭

配置文件中需要保留：

```yaml
use_neutral_decoupling: true
use_speaker_state: true
use_similar_anchor_separation: true
use_hard_anchor_negative: true
```

关闭后模型应当退化为原基础模型。

---

## 3. 修改一：Hard Negative Loss 改成 logsumexp 稳定版

### 3.1 当前风险

如果当前 hard negative loss 类似：

```python
logits = sim / temperature
exp_logits = torch.exp(logits)
```

当 temperature 很小时，`exp(logits)` 很容易变成 `inf`，最终导致 NaN。

### 3.2 替换为稳定版本

请将 hard negative loss 替换为：

```python
import torch
import torch.nn.functional as F


def hard_anchor_negative_loss_stable(
    sample_repr,
    labels,
    class_anchors,
    similar_pair_set,
    temperature=0.1,
    rho=1.0,
):
    if sample_repr is None or labels is None:
        return class_anchors.new_tensor(0.0)

    if sample_repr.size(0) == 0:
        return class_anchors.new_tensor(0.0)

    sample_repr = F.normalize(sample_repr, dim=-1)
    class_anchors = F.normalize(class_anchors, dim=-1)

    logits = torch.matmul(sample_repr, class_anchors.T) / temperature

    weight = torch.ones_like(logits)

    for i in range(labels.size(0)):
        y = int(labels[i].item())
        for c in range(class_anchors.size(0)):
            if c == y:
                continue
            if (y, c) in similar_pair_set:
                weight[i, c] = 1.0 + rho

    weighted_logits = logits + torch.log(weight.clamp(min=1e-8))
    log_den = torch.logsumexp(weighted_logits, dim=-1)

    index = torch.arange(labels.size(0), device=labels.device)
    log_pos = logits[index, labels]

    loss = (log_den - log_pos).mean()

    if not torch.isfinite(loss):
        raise ValueError("hard_anchor_negative_loss_stable produced NaN or Inf.")

    return loss
```

### 3.3 推荐参数

先使用保守参数：

```yaml
hard_negative_temperature: 0.1
hard_negative_rho: 1.0
lambda_hard: 0.01
```

如果还不稳定，继续降：

```yaml
hard_negative_temperature: 0.2
hard_negative_rho: 0.5
lambda_hard: 0.005
```

---

## 4. 修改二：SAS Loss 必须 normalize + clamp

### 4.1 当前风险

如果 SAS 中直接写：

```python
sim = torch.sum(v_c_k * v_d_k)
```

但没有执行 `F.normalize`，这个 sim 是普通点积，不是 cosine，数值可能无限增大。

### 4.2 替换为稳定版本

```python
import torch
import torch.nn.functional as F


def similar_anchor_separation_loss_stable(
    anchor_embeddings,
    domain_adapters,
    similar_pairs,
    margin=0.30,
):
    if similar_pairs is None or len(similar_pairs) == 0:
        return anchor_embeddings.new_tensor(0.0)

    losses = []

    for c, d in similar_pairs:
        if c >= anchor_embeddings.size(0) or d >= anchor_embeddings.size(0):
            continue

        for k, adapter in enumerate(domain_adapters):
            a_c_k = anchor_embeddings[c, k]
            a_d_k = anchor_embeddings[d, k]

            v_c_k = adapter(a_c_k.unsqueeze(0)).squeeze(0)
            v_d_k = adapter(a_d_k.unsqueeze(0)).squeeze(0)

            v_c_k = F.normalize(v_c_k, dim=-1)
            v_d_k = F.normalize(v_d_k, dim=-1)

            sim = torch.sum(v_c_k * v_d_k, dim=-1)
            sim = torch.clamp(sim, min=-1.0, max=1.0)

            losses.append(F.relu(sim - margin).pow(2))

    if len(losses) == 0:
        return anchor_embeddings.new_tensor(0.0)

    loss = torch.stack(losses).mean()

    if not torch.isfinite(loss):
        raise ValueError("similar_anchor_separation_loss_stable produced NaN or Inf.")

    return loss
```

### 4.3 推荐参数

```yaml
sas_margin: 0.30
lambda_sas: 0.005
```

如果依然不稳定：

```yaml
lambda_sas: 0.002
```

---

## 5. 修改三：Neutral Decoupling 不要直接 log(final_probs)

### 5.1 当前风险

如果训练时写：

```python
L_final = F.nll_loss(torch.log(final_probs), labels)
```

那么只要概率里出现 0，就会得到：

```text
log(0) = -inf
```

后面很容易变成 NaN。

### 5.2 推荐做法

训练时不要对合成概率直接取 log。训练阶段拆成两个稳定损失：

1. neutral / non-neutral：`BCEWithLogits`
2. 非中性情绪分类：`CrossEntropy`

### 5.3 稳定实现

```python
import torch
import torch.nn.functional as F


def neutral_decoupling_loss_stable(
    neutral_logit,
    non_neutral_logits,
    labels,
    neutral_id,
    label_id_to_non_neutral_id,
):
    device = labels.device

    neutral_target = (labels == neutral_id).float()

    L_neu = F.binary_cross_entropy_with_logits(
        neutral_logit,
        neutral_target
    )

    non_neutral_mask = labels != neutral_id

    if non_neutral_mask.sum() > 0:
        original_labels = labels[non_neutral_mask]

        mapped_labels = [
            label_id_to_non_neutral_id[int(y)]
            for y in original_labels.tolist()
        ]

        mapped_labels = torch.tensor(
            mapped_labels,
            dtype=torch.long,
            device=device
        )

        L_emo = F.cross_entropy(
            non_neutral_logits[non_neutral_mask],
            mapped_labels
        )
    else:
        L_emo = neutral_logit.new_tensor(0.0)

    L_task = L_neu + L_emo

    if not torch.isfinite(L_task):
        raise ValueError("neutral_decoupling_loss_stable produced NaN or Inf.")

    return L_task, L_neu, L_emo
```

### 5.4 推理阶段再合成 final_probs

```python
def compose_final_probs(
    neutral_logit,
    non_neutral_logits,
    num_labels,
    neutral_id,
    non_neutral_id_to_label_id,
):
    p_neu = torch.sigmoid(neutral_logit)
    p_emo = F.softmax(non_neutral_logits, dim=-1)

    batch_size = neutral_logit.size(0)
    final_probs = torch.zeros(batch_size, num_labels, device=neutral_logit.device)

    final_probs[:, neutral_id] = p_neu

    for non_neu_idx, original_label_id in enumerate(non_neutral_id_to_label_id):
        final_probs[:, original_label_id] = (1.0 - p_neu) * p_emo[:, non_neu_idx]

    final_probs = torch.clamp(final_probs, min=1e-8, max=1.0)
    final_probs = final_probs / final_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    return final_probs
```

---

## 6. 修改四：处理空 batch

### 6.1 问题

使用 neutral decoupling 后，某些 batch 可能全是 neutral：

```python
non_neutral_mask.sum() == 0
```

此时如果继续计算非中性 SupCon、hard negative 或分类 loss，就可能 NaN。

### 6.2 修改要求

所有非中性分支都要加判断：

```python
if non_neutral_mask.sum() == 0:
    L_emo = h_i.new_tensor(0.0)
    L_supcon = h_i.new_tensor(0.0)
    L_hard = h_i.new_tensor(0.0)
else:
    # normal computation
```

SupCon 中也要处理没有正样本的情况：

```python
positive_count = positive_mask.sum(dim=1)
valid = positive_count > 0

if valid.sum() == 0:
    return features.new_tensor(0.0)
```

---

## 7. 修改五：稳定锚点动量更新

### 7.1 当前风险

锚点训练后期可能被不断拉偏，造成 anchor norm 增大和语义空间崩坏。

### 7.2 修改要求

1. 前几个 epoch 冻结锚点；
2. 更新时使用 `torch.no_grad()`；
3. 更新后 normalize；
4. momentum 设置更保守。

### 7.3 推荐配置

```yaml
prototype_momentum: 0.99
freeze_prototype_epochs: 2
normalize_prototypes_after_update: true
```

如果仍然不稳定：

```yaml
prototype_momentum: 0.995
freeze_prototype_epochs: 3
```

### 7.4 示例代码

```python
@torch.no_grad()
def update_prototypes_stable(
    anchors,
    batch_h,
    batch_labels,
    batch_assignments,
    momentum=0.99,
    current_epoch=0,
    freeze_epochs=2,
    normalize=True,
):
    if current_epoch < freeze_epochs:
        return anchors

    for c in range(anchors.size(0)):
        for k in range(anchors.size(1)):
            mask = (batch_labels == c) & (batch_assignments == k)

            if mask.sum() == 0:
                continue

            batch_mean = batch_h[mask].mean(dim=0)

            new_anchor = momentum * anchors[c, k] + (1.0 - momentum) * batch_mean

            if normalize:
                new_anchor = F.normalize(new_anchor, dim=-1)

            anchors[c, k].copy_(new_anchor)

    return anchors
```

---

## 8. 修改六：Speaker State Fusion 加 LayerNorm

### 8.1 当前风险

不要直接：

```python
u_i = h_i + r_i
```

也不要直接：

```python
gate_input = torch.cat([h_i, r_i], dim=-1)
```

这样容易导致输入尺度变大。

### 8.2 推荐实现

```python
import torch
import torch.nn as nn


class SpeakerStateFusion(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()

        self.state_proj = nn.Linear(hidden_size, hidden_size)

        self.state_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, h_i, r_i):
        if r_i is None:
            return h_i

        e_i = self.state_proj(r_i)

        alpha_i = self.state_gate(
            torch.cat([h_i, e_i], dim=-1)
        )

        u_i = self.norm(h_i + alpha_i * e_i)

        return u_i
```

---

## 9. 修改七：加入梯度裁剪和 NaN 检查

### 9.1 梯度裁剪

训练循环中加入：

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
scheduler.step()
optimizer.zero_grad()
```

如果仍然不稳定：

```yaml
max_grad_norm: 0.5
```

### 9.2 Loss 检查

```python
def check_finite_loss(loss_dict):
    for name, value in loss_dict.items():
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            if not torch.isfinite(value).all():
                print(f"[NaN/Inf Detected] {name}: {value}")
                raise ValueError(f"{name} is NaN or Inf.")
```

使用方式：

```python
loss_dict = {
    "loss": loss,
    "loss_task": L_task,
    "loss_neu": L_neu,
    "loss_emo": L_emo,
    "loss_supcon": L_supcon,
    "loss_angle": L_angle,
    "loss_sas": L_sas,
    "loss_hard": L_hard,
}

check_finite_loss(loss_dict)
```

### 9.3 参数检查

```python
def check_model_parameters(model):
    for name, param in model.named_parameters():
        if param is None:
            continue
        if not torch.isfinite(param).all():
            raise ValueError(f"Parameter {name} contains NaN or Inf.")
```

---

## 10. 推荐稳定配置

请新增配置文件：

```text
configs/stable_sas_nsg_eacl.yaml
```

内容如下：

```yaml
# learning rate
learning_rate_encoder: 1e-5
learning_rate_new_modules: 5e-5
warmup_ratio: 0.1
max_grad_norm: 1.0

# temperature
temperature: 0.1

# loss weights
lambda_neu: 0.2
lambda_supcon: 0.2
lambda_angle: 0.01
lambda_sas: 0.005
lambda_hard: 0.01

# SAS
sas_margin: 0.30

# hard negative
hard_negative_rho: 1.0
hard_negative_temperature: 0.1

# prototypes
prototype_momentum: 0.99
freeze_prototype_epochs: 2
normalize_prototypes_after_update: true

# modules
use_neutral_decoupling: true
use_speaker_state: true
use_similar_anchor_separation: true
use_hard_anchor_negative: true

# training
early_stopping_patience: 5
```

如果仍然 NaN，使用更保守配置：

```yaml
lambda_sas: 0.002
lambda_hard: 0.005
hard_negative_temperature: 0.2
hard_negative_rho: 0.5
max_grad_norm: 0.5
prototype_momentum: 0.995
freeze_prototype_epochs: 3
```

---

## 11. 排查顺序

### Step 1：只跑 Base

```yaml
use_neutral_decoupling: false
use_speaker_state: false
use_similar_anchor_separation: false
use_hard_anchor_negative: false
```

如果 Base 也 NaN，问题在原模型。

重点查：

```text
SupCon loss
angle loss
prototype update
learning rate
```

---

### Step 2：只开 Neutral

```yaml
use_neutral_decoupling: true
use_speaker_state: false
use_similar_anchor_separation: false
use_hard_anchor_negative: false
```

如果 NaN，重点查：

```text
是否直接 log(final_probs)
non_neutral_mask 是否为空
label mapping 是否错误
BCE 是否重复 sigmoid
```

---

### Step 3：打开 Speaker State

```yaml
use_neutral_decoupling: true
use_speaker_state: true
use_similar_anchor_separation: false
use_hard_anchor_negative: false
```

如果 NaN，重点查：

```text
r_i 是否有 NaN
state fusion 是否有 LayerNorm
domain gate 输入是否过大
domain_weights 是否极端 one-hot
```

---

### Step 4：打开 SAS，不开 Hard

```yaml
use_neutral_decoupling: true
use_speaker_state: true
use_similar_anchor_separation: true
use_hard_anchor_negative: false
```

如果 NaN，重点查：

```text
SAS 是否 normalize
SAS sim 是否 clamp
lambda_sas 是否太大
anchor norm 是否变大
```

---

### Step 5：最后打开 Hard Negative

```yaml
use_neutral_decoupling: true
use_speaker_state: true
use_similar_anchor_separation: true
use_hard_anchor_negative: true
```

如果只有这一步 NaN，基本就是：

```text
hard loss 指数爆炸
temperature 太小
rho 太大
lambda_hard 太大
```

---

## 12. 必须记录的训练日志

每隔 50 或 100 step 输出：

```python
print("loss:", loss.item())
print("loss_task:", L_task.item())
print("loss_neu:", L_neu.item())
print("loss_emo:", L_emo.item())
print("loss_supcon:", L_supcon.item())
print("loss_angle:", L_angle.item())
print("loss_sas:", L_sas.item())
print("loss_hard:", L_hard.item())
```

同时输出：

```python
print("max neutral_logit:", neutral_logit.abs().max().item())
print("max non_neutral_logits:", non_neutral_logits.abs().max().item())
print("anchor norm mean:", anchors.norm(dim=-1).mean().item())
print("anchor norm max:", anchors.norm(dim=-1).max().item())
print("domain weight min:", domain_weights.min().item())
print("domain weight max:", domain_weights.max().item())
```

---

## 13. 问题定位规则

### 13.1 hard loss 突然 inf

处理：

```yaml
hard_negative_temperature: 0.2
hard_negative_rho: 0.5
lambda_hard: 0.005
```

并确认代码已使用 `logsumexp`。

---

### 13.2 anchor norm 持续增大

处理：

```yaml
prototype_momentum: 0.995
freeze_prototype_epochs: 3
normalize_prototypes_after_update: true
```

---

### 13.3 domain_weights 极端 one-hot

如果出现：

```text
domain weight min: 0.0000
domain weight max: 0.9999
```

说明 domain gate 过度偏向单一域。

处理：

1. 降低 domain gate 学习率；
2. gate 输入前加 LayerNorm；
3. 增加 dropout；
4. 可加入 gate entropy regularization。

可选：

```python
gate_entropy = -torch.sum(
    domain_weights * torch.log(domain_weights + 1e-8),
    dim=-1
).mean()

L_total = L_total - lambda_gate_entropy * gate_entropy
```

推荐：

```yaml
lambda_gate_entropy: 0.001
```

---

### 13.4 全部预测 neutral

说明 neutral branch 太强。

处理：

```yaml
lambda_neu: 0.1
```

或者对 BCE 使用 class weight。

---

## 14. 总损失推荐写法

建议使用：

```python
L_task = L_emo + lambda_neu * L_neu

L_total = (
    L_task
    + lambda_supcon * L_supcon
    + lambda_angle * L_angle
    + lambda_sas * L_sas
    + lambda_hard * L_hard
)
```

注意：

如果 `L_task` 已经包含 `lambda_neu * L_neu`，总损失中不要再次加 `L_neu`。

---

## 15. Codex 修改任务清单

请 Codex 按以下顺序改：

1. 替换 hard negative loss 为 `logsumexp` 稳定版；
2. 替换 SAS loss，确保 normalize + clamp；
3. 修改 neutral decoupling loss，不再直接 `log(final_probs)`；
4. 所有 non-neutral 分支增加 empty batch 判断；
5. 修改 prototype update：冻结前几轮、no_grad、更新后 normalize；
6. speaker state fusion 增加 projection、gate 和 LayerNorm；
7. 训练循环中加入 gradient clipping；
8. 加入 loss / parameter finite check；
9. 新增 `stable_sas_nsg_eacl.yaml`；
10. 分阶段打开模块进行排查。

---

## 16. 最推荐的短期修复方案

为了最快恢复稳定训练，先做这个最小修复版本：

```text
1. hard negative loss 改为 logsumexp；
2. neutral loss 改为 BCEWithLogits + CrossEntropy；
3. prototype update 后 normalize；
4. 加 gradient clipping；
5. lambda_hard 降到 0.005；
6. lambda_sas 降到 0.002；
7. freeze_prototype_epochs = 2。
```

如果该版本稳定，再逐步恢复：

```yaml
lambda_sas: 0.005
lambda_hard: 0.01
hard_negative_temperature: 0.1
hard_negative_rho: 1.0
```

---

## 17. 预期结果

修改后应达到：

```text
1. 训练不再出现 NaN；
2. hard loss 不再突然 inf；
3. anchor norm 保持稳定；
4. logits 不会持续增大到几十或几百；
5. domain gate 不会过早完全 one-hot；
6. 训练后期即使下降，也应是正常过拟合，而不是数值爆炸；
7. 可以通过消融实验确认问题来自哪个增强模块。
```
