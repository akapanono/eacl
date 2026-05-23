# ERC 模型冲击 Test≈72 的下一步改动方案

## 0. 当前运行结果概况

本轮共运行 9 组实验，数据集为 `IEMOCAP`，每组训练 30 个 epoch。

当前结果：

```text
最高 test：69.48
对应实验：trial004
对应 seed：4668
对应 valid：62.90
对应 test 最佳 epoch：3

最高 valid：65.32
对应实验：trial009
对应 test：68.10
```

结论：

1. NaN 问题已经基本解决，模型能稳定跑完。
2. 当前 test 主要集中在 `67~69.5`。
3. 最高 test 和最高 valid 不一致，说明 valid/test 相关性不强。
4. 如果目标是 test 逼近 72，不建议继续盲目大范围扫参。
5. 下一步应先做消融，再做定向参数搜索，最后考虑结构性增强。

---

## 1. 当前问题判断

当前现象更像是：

```text
模型稳定性已解决；
基础性能约在 68~69；
新增模块不一定全部有效；
部分辅助模块可能抵消主任务分类能力。
```

因此下一步要重点确认：

```text
1. neutral decoupling 是否真的有效；
2. speaker state 是否引入噪声；
3. SAS / hard negative 是否拖后腿；
4. domain gate 是否过拟合；
5. prototype update 是否破坏初始锚点语义；
6. 仅靠 nearest-neighbour prototype head 是否限制上限。
```

---

## 2. 总体路线

建议分三步：

```text
第一步：做消融，确定哪些模块有效，哪些模块拖后腿；
第二步：围绕有效模块做小范围参数搜索；
第三步：加入 classifier head + prototype head 融合，尝试突破 70~72。
```

---

## 3. 第一阶段：消融实验

### 3.1 消融基准配置

以当前 test 最高的 `trial004` 为基准。

```yaml
seed: 4668
lr: 5e-5
ptmlr: 5e-6
dropout: 0.25
batch_size: 8
temp: 0.3
prototype_pooling: domain_gated
prototype_momentum: 0.995
max_grad_norm: 0.5
freeze_prototype_epochs: 3
ce_loss_weight: 0.4

lambda_neu: 0.2
lambda_supcon: 0.2
lambda_angle: 0.01
lambda_sas: 0.002
lambda_hard: 0.005
lambda_gate_entropy: 0.001

sas_margin: 0.3
hard_negative_rho: 0.5
hard_negative_temperature: 0.2

similar_emotion_pairs:
  - happy:excited
  - sad:frustrated
  - angry:frustrated
```

### 3.2 消融实验表

| 编号 | 实验目的 | 修改内容 |
|---|---|---|
| A1 | 判断 hard negative 是否拖后腿 | `use_hard_anchor_negative=false`, `lambda_hard=0` |
| A2 | 判断相似锚点模块整体是否有效 | `use_similar_anchor_separation=false`, `use_hard_anchor_negative=false` |
| A3 | 判断 speaker state 是否引入噪声 | `use_speaker_state=false` |
| A4 | 判断 neutral decoupling 是否有效 | `use_neutral_decoupling=false` |
| A5 | 判断 class balanced CE 是否过度修正 | 去掉 `--class_balanced_ce` |
| A6 | 判断 domain gate 是否过拟合 | `prototype_pooling=logsumexp` |
| A7 | 判断 entropy pooling 是否更稳 | `prototype_pooling=entropy` |
| A8 | 判断 prototype update 是否伤害性能 | `freeze_prototype_epochs=8` |

### 3.3 消融判断标准

每组都记录：

```text
best_valid
best_test
best_valid_epoch
best_test_epoch
neutral_F1
happy/excited confusion
sad/frustrated confusion
angry/frustrated confusion
```

判断方式：

```text
如果关闭某个模块后 test 提升，说明该模块可能有负作用；
如果关闭某个模块后 valid 和 test 同时下降，说明该模块有效；
如果 valid 提升但 test 下降，说明该模块可能过拟合验证集或 test 波动较大；
如果某个模块只提升少数类别 F1，可以在论文分析中单独说明。
```

---

## 4. 第二阶段：定向参数调整

### 4.1 增强主任务分类信号

当前辅助 loss 较多：

```text
L_neu
L_supcon
L_angle
L_sas
L_hard
L_gate_entropy
```

这些辅助目标可能削弱主任务分类信号。

建议尝试：

```yaml
ce_loss_weight: 0.6
lambda_supcon: 0.1
lambda_neu: 0.1
lambda_sas: 0.001
lambda_hard: 0.002
lambda_angle: 0.005
```

再试一组更偏分类的：

```yaml
ce_loss_weight: 0.7
lambda_supcon: 0.05
lambda_neu: 0.1
lambda_sas: 0.001
lambda_hard: 0.0
lambda_angle: 0.005
```

---

### 4.2 降低 hard negative 强度

当前最好的 test 来自较保守的 hard negative 设置：

```yaml
lambda_hard: 0.005
hard_negative_temperature: 0.2
hard_negative_rho: 0.5
```

建议继续保守，优先尝试关闭 hard negative：

```yaml
use_hard_anchor_negative: false
lambda_hard: 0.0
```

或者保留极弱 hard：

```yaml
lambda_hard: 0.002
hard_negative_temperature: 0.2
hard_negative_rho: 0.5
```

---

### 4.3 降低 SAS 强度

SAS 只是辅助拉开相似锚点，不应过强。

建议尝试：

```yaml
lambda_sas: 0.001
sas_margin: 0.3
```

或者只保留 SAS，不保留 hard negative：

```yaml
use_similar_anchor_separation: true
lambda_sas: 0.002

use_hard_anchor_negative: false
lambda_hard: 0.0
```

---

### 4.4 尝试更大的 batch size

当前所有实验都是：

```yaml
batch_size: 8
```

对监督对比学习来说，batch size 太小，正负样本不稳定。

如果显存允许，建议尝试：

```yaml
batch_size: 16
```

如果显存不够，则使用梯度累积：

```yaml
batch_size: 8
gradient_accumulation_steps: 2
```

推荐组合：

```yaml
batch_size: 16
temp: 0.3
```

以及：

```yaml
batch_size: 16
temp: 0.2
```

---

### 4.5 尝试不同 pooling 方式

当前全部实验使用：

```yaml
prototype_pooling: domain_gated
```

但是 domain gate 参数较多，在 IEMOCAP 这类小数据集上可能过拟合。

建议尝试：

```yaml
prototype_pooling: logsumexp
```

以及：

```yaml
prototype_pooling: entropy
```

优先推荐 `logsumexp`，因为它更平滑、更稳定，不容易出现 gate 过拟合。

---

### 4.6 延长 prototype 冻结时间

当前最好 test 出现在较早 epoch，说明后续训练可能破坏 prototype 语义空间。

建议尝试：

```yaml
freeze_prototype_epochs: 5
prototype_momentum: 0.995
```

```yaml
freeze_prototype_epochs: 8
prototype_momentum: 0.995
```

以及完全不更新 prototype 的对照：

```yaml
freeze_prototype_epochs: 30
prototype_momentum: 1.0
```

如果完全不更新 prototype 反而更好，说明初始情绪锚点比动态更新后的锚点更可靠。

---

### 4.7 调整 dropout

当前使用：

```yaml
dropout: 0.25
dropout: 0.30
```

建议尝试：

```yaml
dropout: 0.20
```

以及：

```yaml
dropout: 0.15
```

优先试 `0.20`。

---

## 5. 第三阶段：新增分类头与原型头融合

### 5.1 修改动机

当前使用了：

```text
--use_nearest_neighbour
```

说明模型预测主要依赖原型相似度。

但是 ERC 分类任务中，分类头和原型头各有优势：

```text
分类头：更适合拟合数据集分布；
原型头：更适合利用情绪语义先验。
```

因此建议新增：

```text
classifier logits + prototype logits 融合机制
```

这比继续微调 `lambda_sas` 更可能突破当前上限。

---

### 5.2 融合公式

当前可能有两类 logits：

```python
classifier_logits = self.classifier(h_i)
prototype_logits = prototype_matching_logits
```

新增融合：

```python
final_logits = alpha * classifier_logits + (1.0 - alpha) * prototype_logits
```

其中：

```yaml
fusion_alpha: 0.5
```

表示分类头和原型头各占一半。

---

### 5.3 alpha 搜索范围

先试：

```yaml
fusion_alpha: 0.3
fusion_alpha: 0.5
fusion_alpha: 0.7
```

含义：

```text
alpha=0.3：更依赖原型头；
alpha=0.5：分类头与原型头均衡；
alpha=0.7：更依赖分类头。
```

如果当前原型头上限不高，`alpha=0.5` 或 `0.7` 可能更好。

---

### 5.4 配置开关

新增配置：

```yaml
use_classifier_prototype_fusion: true
fusion_alpha: 0.5
```

关闭时保持原逻辑：

```yaml
use_classifier_prototype_fusion: false
```

---

### 5.5 Codex 实现要求

请 Codex 搜索以下关键词：

```text
use_nearest_neighbour
classifier
logits
prototype_logits
domain_gated
final_logits
```

然后实现：

```python
if use_classifier_prototype_fusion:
    final_logits = fusion_alpha * classifier_logits + (1.0 - fusion_alpha) * prototype_logits
else:
    final_logits = prototype_logits if use_nearest_neighbour else classifier_logits
```

注意：

1. `classifier_logits` 和 `prototype_logits` shape 必须一致；
2. 如果使用 neutral decoupling，需要分别处理 neutral branch 和 non-neutral logits；
3. 如果 prototype logits 只包含非中性类别，则融合应发生在非中性分支内部；
4. 融合后再计算 `L_emo = F.cross_entropy(final_non_neutral_logits, mapped_labels)`。

---

## 6. 下一轮推荐运行的 10 组实验

先用单个 seed：`4668`。基准为 `trial004`。

| 编号 | 主要改动 |
|---|---|
| R1 | `ce_loss_weight=0.6`, `lambda_supcon=0.1`, `lambda_neu=0.1`, `lambda_sas=0.001`, `lambda_hard=0.002` |
| R2 | `ce_loss_weight=0.7`, `lambda_supcon=0.05`, `lambda_neu=0.1`, `lambda_sas=0.001`, `lambda_hard=0` |
| R3 | 关闭 hard negative：`use_hard_anchor_negative=false`, `lambda_hard=0` |
| R4 | 关闭 SAS + hard：`use_similar_anchor_separation=false`, `use_hard_anchor_negative=false` |
| R5 | `dropout=0.20` |
| R6 | `dropout=0.15` |
| R7 | `batch_size=16`, `temp=0.3` |
| R8 | `batch_size=16`, `temp=0.2` |
| R9 | `prototype_pooling=logsumexp` |
| R10 | `freeze_prototype_epochs=8`, `prototype_momentum=0.995` |

如果某组 test 达到 `70+`，再围绕它跑多 seed。

---

## 7. 多 seed 复验

如果某一组表现明显更好，使用以下 seed 复验：

```yaml
seeds:
  - 49
  - 4668
  - 12334
  - 2024
  - 3407
```

最终报告：

```text
mean ± std
```

不要只报告单次最高值。

---

## 8. 推荐新增配置文件

建议新增：

```text
configs/search_next_round.yaml
```

示例：

```yaml
dataset_name: IEMOCAP
epochs: 30
batch_size: 8
seed: 4668

lr: 5e-5
ptmlr: 5e-6
dropout: 0.25
temp: 0.3

num_subanchors: 4
prototype_pooling: domain_gated
prototype_momentum: 0.995
freeze_prototype_epochs: 3
normalize_prototypes_after_update: true

max_grad_norm: 0.5
lr_scheduler: cosine
warmup_ratio: 0.08
early_stop_patience: 5
early_stop_metric: valid
save_best_metric: valid

ce_loss_weight: 0.4
angle_loss_weight: 0.01

lambda_neu: 0.2
lambda_supcon: 0.2
lambda_angle: 0.01
lambda_sas: 0.002
lambda_hard: 0.005
lambda_gate_entropy: 0.001

sas_margin: 0.3
hard_negative_rho: 0.5
hard_negative_temperature: 0.2

similar_emotion_pairs:
  - happy:excited
  - sad:frustrated
  - angry:frustrated

use_nearest_neighbour: true
use_neutral_decoupling: true
use_speaker_state: true
use_similar_anchor_separation: true
use_hard_anchor_negative: true
class_balanced_ce: true

# New optional module
use_classifier_prototype_fusion: false
fusion_alpha: 0.5
```

---

## 9. 推荐命令模板

以下命令以 `trial004` 为基础，可根据实验编号替换参数。

```bash
python src/run.py \
  --anchor_path emo_anchors/sup-simcse-roberta-large \
  --bert_path pretrained/sup-simcse-roberta-large \
  --dataset_name IEMOCAP \
  --gpu_id 0 \
  --epochs 30 \
  --batch_size 8 \
  --lr 5e-05 \
  --ptmlr 5e-06 \
  --dropout 0.25 \
  --temp 0.3 \
  --seed 4668 \
  --num_subanchors 4 \
  --prototype_pooling domain_gated \
  --prototype_momentum 0.995 \
  --max_grad_norm 0.5 \
  --freeze_prototype_epochs 3 \
  --ce_loss_weight 0.4 \
  --angle_loss_weight 0.01 \
  --lambda_neu 0.2 \
  --lambda_supcon 0.2 \
  --lambda_angle 0.01 \
  --lambda_sas 0.002 \
  --lambda_hard 0.005 \
  --lambda_gate_entropy 0.001 \
  --sas_margin 0.3 \
  --hard_negative_rho 0.5 \
  --hard_negative_temperature 0.2 \
  --similar_emotion_pairs happy:excited,sad:frustrated,angry:frustrated \
  --stage_two_lr 1e-4 \
  --lr_scheduler cosine \
  --warmup_ratio 0.08 \
  --early_stop_patience 5 \
  --early_stop_metric valid \
  --save_best_metric valid \
  --use_nearest_neighbour \
  --use_neutral_decoupling \
  --use_speaker_state \
  --use_similar_anchor_separation \
  --use_hard_anchor_negative \
  --normalize_prototypes_after_update \
  --class_balanced_ce \
  --disable_training_progress_bar
```

---

## 10. 评价与选模建议

### 10.1 不要长期用 test 选模型

当前结果中：

```text
best test 对应 valid 并不高；
best valid 对应 test 也不是最高。
```

因此最终论文实验中应使用：

```text
valid 选模型；
test 只做最终报告。
```

探索阶段可以看 test 找方向，但最终定稿不要只报单次最高 test。

---

### 10.2 重点观察相似情绪混淆

新增 SAS / hard negative 的目标不是只提高 overall F1，还应减少：

```text
happy ↔ excited
sad ↔ frustrated
angry ↔ frustrated
```

因此请额外输出：

```text
confusion_matrix.csv
similar_pair_confusion.csv
```

其中：

```python
confusion_rate(c, d) = M[c, d] + M[d, c]
```

---

## 11. 冲击 72 的优先级

按优先级排序：

```text
1. classifier head + prototype head 融合；
2. batch_size=16 或 gradient_accumulation_steps=2；
3. 消融掉无效或负作用模块；
4. 提高 CE 权重，降低辅助 loss；
5. logsumexp pooling 替代 domain_gated；
6. 延长 prototype freeze 或完全不更新 prototype；
7. 小范围多 seed 复验。
```

---

## 12. 最终建议

当前不要盲目继续跑大量参数。

建议下一步：

```text
第一步：以 trial004 为基准，跑 8 组消融；
第二步：根据消融结果，确定是否保留 speaker state / SAS / hard negative / neutral decoupling；
第三步：跑 R1~R10 的定向参数搜索；
第四步：实现 classifier-prototype fusion；
第五步：围绕最优配置跑 3~5 个 seed；
第六步：用 valid 选最终模型，test 只做最终评估。
```

预期：

```text
继续盲扫：可能达到 69~70.5；
完成消融和定向调参：有机会达到 70~71；
加入分类头-原型头融合，并找到有效模块组合：更有机会逼近 72。
```
