# YOLOv11n 预处理与后处理规范

## 预处理

- 输入图片尺寸: `640 × 640`
- 保持长宽比: 是（letterbox）
- 缩放方法: 双线性插值（`cv2.INTER_LINEAR`）
- Letterbox 填充值: `114`（灰色填充）
- 颜色顺序: `RGB`（原始图片为 BGR，需转换）
- 输入数据类型: `float32`（INT8 部署时由输入层 QuantizeLinear 完成 float32 → int8）
- 归一化公式: `pixel_value / 255.0`（无 mean/std 标准化）
- 输入张量布局: `NCHW`（`[1, 3, 640, 640]`）
- Batch 大小: `1`

### 预处理流程（伪代码）

```
1. 读取图片 (BGR, HWC)
2. 计算缩放比例 r = min(640/H, 640/W)
3. 缩放到新尺寸 (int(W*r), int(H*r))
4. 使用 114 填充至 640×640（居中填充）
5. BGR → RGB
6. /255.0 归一化到 [0, 1]
7. HWC → CHW
8. 扩展 batch 维度 → [1, 3, 640, 640]
```

### Gemmini 侧预处理说明

- Gemmini 侧输入应为 INT8 张量
- 预处理需在 RISC-V CPU 端完成（读取图片、缩放、填充、归一化）
- 归一化后的 float32 值通过输入层 scale/zero_point 转为 int8
- 输入层 scale = `0.007874` ≈ `1/127`，zero_point = `0`（对称量化）
- 转换公式: `input_int8 = round(input_fp32 / input_scale)`

## 类别映射

- 类别来源文件: `ultralytics/cfg/datasets/bdd100k.yaml`
- 类别数量: `10`
- 类别顺序:

| ID | 类别名 |
|---|---|
| 0 | pedestrian |
| 1 | rider |
| 2 | car |
| 3 | truck |
| 4 | bus |
| 5 | train |
| 6 | motorcycle |
| 7 | bicycle |
| 8 | traffic light |
| 9 | traffic sign |

## 原始输出张量

| 张量名 | Shape | Dtype | 含义 |
|---|---|---|---|
| output0 | `[1, 14, 8400]` | float32 | bbox(4) + num_classes(10) 的检测结果，共 8400 个候选框 |

### 输出张量结构解析

- `14 = 4 (bbox) + 10 (class scores)`
- `8400 = 80×80 + 40×40 + 20×20 = 6400 + 1600 + 400`
- 三个检测头对应三个特征图尺度（stride 8, 16, 32）

## 解码规则

- 边界框参数化: anchor-free（无锚框）
- Stride 列表: `[8, 16, 32]`
- 特征图尺寸: `[80×80, 40×40, 20×20]`
- 每个特征图点预测 1 个框

### 边界框解码

YOLOv11 使用 Distribution Focal Loss (DFL) 进行边界框回归：
- 原始输出前 4 个通道为 `dist_left, dist_top, dist_right, dist_bottom`
- 经过 DFL decode（Softmax + 期望值计算）后得到距离值
- 最终 bbox = `[cx - left*stride, cy - top*stride, cx + right*stride, cy + bottom*stride]`

### 类别分数

- Sigmoid 激活: 是（类别分数需经 sigmoid 转换为概率）
- 分数计算: `score = sigmoid(raw_class_score)`
- 无 objectness score（anchor-free 设计）

## 阈值

- 置信度阈值: `0.25`（默认）
- IoU 阈值: `0.7`（默认）
- 最大检测数: `300`

## NMS

- NMS 类型: 标准 NMS
- 类别: per-class（每类独立做 NMS）
- 输出格式: `[x1, y1, x2, y2, confidence, class_id]`

### NMS 流程

```
1. 过滤置信度 < threshold 的候选框
2. 对每个类别独立执行 NMS
3. 按 IoU 阈值抑制重叠框
4. 限制最终输出不超过 max_det
5. 将 letterbox 坐标映射回原始图片坐标
```

## Gemmini 侧说明

### 可在 Gemmini 侧保持一致的部分

- 卷积计算（Conv2d、1x1 Conv、Depthwise Conv）→ Gemmini 加速
- MatMul 运算 → Gemmini 加速
- Add/ResAdd → Gemmini 加速
- 量化/反量化参数与 params.h 中的值完全一致

### 必须在 C 中实现的部分

- 预处理（图片加载、缩放、填充、归一化）
- SiLU 激活: `x * sigmoid(x)`
- Sigmoid 激活
- Upsample（最近邻上采样，2x）
- Concat（通道拼接）
- Split（通道分割）
- MaxPool（3×3，stride=2，pad=1）
- Reshape/Transpose/Permute
- Softmax（DFL 解码）
- 检测头后处理（DFL 解码、bbox 构建）
- NMS
- 坐标映射（letterbox → 原始图片）

### shen_yolov11n.c 中必须匹配的假设

- 输入尺寸: `640 × 640`
- 输入通道顺序: `RGB`
- 输入归一化: `/255.0`
- 量化参数: 使用 params.h 中的 scale/zero_point
- 类别数: `10`
- 输出: 8400 个候选框
- Stride: `[8, 16, 32]`
