# YOLOv11 训练指标详解

本文档基于 `runs/detect/train4/results.csv` 中的参数，解释每个指标的含义、相互关系和实际意义。

---

## 1. results.csv 列名总览

| 列名 | 类别 | 含义 |
|------|------|------|
| `epoch` | 基础 | 当前训练轮次 |
| `time` | 基础 | 从训练开始到当前 epoch 结束的累计时间（秒） |
| `train/box_loss` | 训练损失 | 边界框回归损失 |
| `train/cls_loss` | 训练损失 | 分类损失 |
| `train/dfl_loss` | 训练损失 | 分布焦点损失 |
| `metrics/precision(B)` | 验证指标 | 精确率 |
| `metrics/recall(B)` | 验证指标 | 召回率 |
| `metrics/mAP50(B)` | 验证指标 | IoU=0.50 下的平均精度均值 |
| `metrics/mAP50-95(B)` | 验证指标 | IoU=0.50:0.95 下的平均精度均值 |
| `val/box_loss` | 验证损失 | 验证集上的边界框损失 |
| `val/cls_loss` | 验证损失 | 验证集上的分类损失 |
| `val/dfl_loss` | 验证损失 | 验证集上的分布焦点损失 |
| `lr/pg0` | 学习率 | 参数组 0（backbone 权重）的学习率 |
| `lr/pg1` | 学习率 | 参数组 1（其他权重）的学习率 |
| `lr/pg2` | 学习率 | 参数组 2（bias 参数）的学习率 |

> 所有列名中的 `(B)` 表示 Bounding Box 检测任务（区别于分割、关键点等任务）。

---

## 2. 训练损失（越低越好）

### train/box_loss — 边界框回归损失

衡量预测框与真实框的位置和大小偏差。模型"框画得准不准"就看这个。

### train/cls_loss — 分类损失

衡量预测类别与真实类别的差距。模型"认得准不准"就看这个。比如把 `car` 认成 `truck` 会导致 cls_loss 上升。

### train/dfl_loss — 分布焦点损失（Distribution Focal Loss）

YOLOv11 特有的辅助损失。传统方法直接回归框的坐标值，而 DFL 将框的边界位置建模为一个概率分布，能更精确地回归框的边缘。可以理解为 box_loss 的"精细化补充"。

### 判断方法

- 三个 loss 都应该随 epoch 稳步下降。
- 若 train loss 下降但 val loss 不降或上升 → 过拟合。

---

## 3. 验证指标（越高越好）

### 先理解 IoU（Intersection over Union，交并比）

IoU 衡量预测框和真实框的重叠程度：

```
IoU = 两个框的交集面积 / 两个框的并集面积
```

- IoU = 1.0 → 完美重合
- IoU = 0.5 → 约一半重叠
- IoU = 0.0 → 完全没有重叠

在评估时需要设定一个 IoU 阈值来判断检测是否"匹配成功"：
- **IoU ≥ 阈值** 且类别正确 → True Positive（TP，正确检测）
- **IoU < 阈值** 或类别错误 → False Positive（FP，误检）
- 没有被任何预测框匹配到的真实目标 → False Negative（FN，漏检）

### metrics/precision(B) — 精确率

```
Precision = TP / (TP + FP) = 正确预测数 / 总预测数
```

**含义**：模型预测出来的所有框中，有多少是对的？

**举例**：模型在一张图上预测了 10 个框，其中 7 个与真实目标匹配，3 个是误检。
→ Precision = 7/10 = 0.70

**Precision 高** = 模型很少"乱报"，但 **不关心有没有漏检**。

### metrics/recall(B) — 召回率

```
Recall = TP / (TP + FN) = 正确预测数 / 真实目标总数
```

**含义**：所有真实目标中，有多少被正确检测到了？

**举例**：图中有 20 个真实目标，模型正确检测到了 8 个。
→ Recall = 8/20 = 0.40

**Recall 高** = 模型很少漏检，但 **不关心是否有误检**。

### Precision 和 Recall 的关系

两者是跷跷板关系：
- **提高置信度阈值**（只保留高置信度预测）→ Precision↑ Recall↓
- **降低置信度阈值**（放出更多预测）→ Recall↑ Precision↓

不可能同时让两者都达到 1.0。

### results.csv 中的 Precision 和 Recall 用的是什么置信度？

**不是固定的 conf=0.5 或 0.25**，而是在 F1 最大的最佳平衡点处的值（见下文 F1 部分）。

---

### metrics/mAP50(B) — mAP@IoU=0.50

**IoU=0.50 的含义**：只要预测框和真实框的重叠面积达到 50% 以上，就算检测成功。这是一个比较宽松的标准。

**mAP 的计算过程**：

1. **对每个类别画 Precision-Recall 曲线**：把置信度阈值从高到低滑动，每个阈值对应一个 (Precision, Recall) 点，连成曲线。
2. **计算曲线下面积 = AP**（Average Precision）：这个面积综合了所有置信度阈值下的表现。
3. **对所有类别取平均 = mAP**（mean AP）：

```
mAP50 = (AP_pedestrian + AP_rider + AP_car + ... + AP_traffic_sign) / 10
```

**mAP50 是评估检测模型最常用的指标**，因为它：
- 综合考虑了 Precision 和 Recall（通过 P-R 曲线积分）
- 不受单一置信度阈值影响
- 对每个类别独立评估后取平均，不受类别数量不平衡的影响

### metrics/mAP50-95(B) — mAP@IoU=0.50:0.95

在 IoU 从 0.50 到 0.95（步长 0.05，共 10 个阈值）下分别计算 mAP，然后取平均。

这是 **最严格的精度指标**，也是 COCO 数据集的官方主指标。IoU=0.95 要求预测框和真实框几乎完美重合，所以 mAP50-95 的数值通常远低于 mAP50。

### Precision vs mAP50 的区别

| | Precision | mAP50 |
|---|---|---|
| **看什么** | 在某个阈值下，预测的准不准 | 在所有阈值下的综合表现 |
| **考虑漏检吗** | 不考虑 | 考虑（通过 Recall 纳入惩罚） |
| **跨类别吗** | 所有类别合在一起算 | 先对每个类别单独算 AP，再平均 |
| **受阈值影响吗** | 是 | 否（对所有阈值积分） |
| **适合做什么** | 看当前设置下误检多不多 | **模型整体性能的标准评价指标** |

**一般来说，mAP50 和 mAP50-95 是评估检测模型最重要的两个指标。**

---

## 4. F1 Score — 精确率与召回率的平衡

### 定义

F1 是 Precision 和 Recall 的调和平均数：

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**只有 Precision 和 Recall 都高时，F1 才会高。**

### 为什么需要 F1

| 场景 | Precision | Recall | F1 | 问题 |
|------|-----------|--------|-----|------|
| 只输出最有把握的 1 个框 | 1.0 | 0.01 | 0.02 | 漏检严重 |
| 把整张图都框上 | 0.01 | 1.0 | 0.02 | 误检严重 |
| 表现均衡 | 0.67 | 0.40 | **0.50** | 相对平衡 |

### F1 曲线（BoxF1_curve.png）

- **横轴**：置信度阈值（0 → 1）
- **纵轴**：F1 值
- **曲线左侧**（低阈值）：预测框太多 → Recall 高但 Precision 低 → F1 不高
- **曲线右侧**（高阈值）：只保留高置信度框 → Precision 高但 Recall 低 → F1 不高
- **曲线顶点**：F1 最大 → Precision 和 Recall 达到最佳平衡

**results.csv 中报告的 Precision 和 Recall 就是在这个 F1 最大的点处取的值。**

---

## 5. 验证损失（越低越好）

| 列名 | 含义 |
|------|------|
| `val/box_loss` | 验证集上的边界框损失 |
| `val/cls_loss` | 验证集上的分类损失 |
| `val/dfl_loss` | 验证集上的分布焦点损失 |

与训练损失对应。对比 train loss 和 val loss 可判断是否过拟合：
- train loss↓ + val loss↓ → 正常学习
- train loss↓ + val loss↑ → 过拟合（模型记住了训练数据，泛化能力变差）
- train loss↓ + val loss 平稳 → 接近收敛

---

## 6. 学习率

| 列名 | 含义 |
|------|------|
| `lr/pg0` | 参数组 0（backbone 权重）的学习率 |
| `lr/pg1` | 参数组 1（其他权重）的学习率 |
| `lr/pg2` | 参数组 2（bias 参数）的学习率 |

学习率控制每次梯度更新的步长。YOLOv11 默认策略：
- **Warmup 阶段**（前 3 个 epoch）：学习率从很小的值逐步升到初始学习率（lr0=0.01）
- **衰减阶段**（epoch 4 到结束）：学习率按余弦退火策略逐步降到 lr0 × lrf = 0.01 × 0.01 = 0.0001

学习率太高 → 训练不稳定、loss 震荡；太低 → 收敛太慢、容易陷入局部最优。

---

## 7. 其他可视化文件说明

训练完成后 `runs/detect/trainN/` 下会生成以下图表：

| 文件 | 含义 |
|------|------|
| `results.png` | 所有指标随 epoch 的变化曲线 |
| `BoxF1_curve.png` | F1 vs 置信度阈值曲线 |
| `BoxP_curve.png` | Precision vs 置信度阈值曲线 |
| `BoxR_curve.png` | Recall vs 置信度阈值曲线 |
| `BoxPR_curve.png` | Precision-Recall 曲线（曲线下面积 = mAP） |
| `confusion_matrix.png` | 混淆矩阵（各类别的预测 vs 真实分布） |
| `labels.jpg` | 数据集标签分布统计 |
| `train_batchN.jpg` | 训练 batch 示例（含增强后的图片和标注） |
| `val_batchN_labels.jpg` | 验证集真实标注 |
| `val_batchN_pred.jpg` | 验证集模型预测结果 |
