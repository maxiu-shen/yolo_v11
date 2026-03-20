# YOLOv11n 模型结构摘要

## 基本信息

- 模型名称: `YOLOv11n`
- 模型来源: `Ultralytics 8.3.185 (weights/yolo11n.pt pretrained + BDD100K finetune)`
- 数据集: `BDD100K`（10 类目标检测）
- 输入尺寸: `640 × 640`
- 训练实验: `train4`（100 epochs, batch=16, imgsz=640）
- 导出日期: `2026-03-20`

## 模型规模

- 层数: `100 (fused)`
- 参数量: `2,584,102`
- GFLOPs: `6.3`
- FP32 权重大小: `5.2 MB (best.pt)`
- FP32 ONNX: `10.1 MB`
- INT8 ONNX: `4.3 MB` (v2: 检测头保持 FP32)
- ONNX 节点数: `320`
- ONNX Conv 层数: `88`
- ONNX MatMul 层数: `2`

## 精度摘要

### FP32 验证结果（BDD100K val, 10000 images）

| 指标 | 值 |
|---|---|
| Precision | 0.662 |
| Recall | 0.410 |
| mAP50 | 0.449 |
| mAP50-95 | 0.253 |

### INT8 验证结果（BDD100K val, 10000 images, GPU）

| 指标 | 值 | vs FP32 |
|---|---|---|
| Precision | 0.654 | -1.15% |
| Recall | 0.380 | -7.42% |
| mAP50 | 0.418 | -6.82% |
| mAP50-95 | 0.233 | -7.92% |

量化策略: PTQ v2 (per-channel 权重, MinMax, 500 张校准, 检测头 FP32)

## 预处理摘要

- 缩放方式: letterbox（保持长宽比，114 填充）
- 颜色顺序: RGB
- 归一化: /255.0
- 输入张量布局: NCHW `[1, 3, 640, 640]`

## 后处理摘要

- 输出张量: `[1, 14, 8400]`（4 bbox + 10 class）
- 解码规则: DFL anchor-free
- 置信度阈值: 0.25
- NMS IoU 阈值: 0.7
- 类别来源: `bdd100k.yaml`（10 类）

## 算子统计

| 算子类型 | 数量 | Gemmini/CPU |
|---|---:|---|
| Conv | 88 | Gemmini |
| MatMul | 2 | Gemmini |
| Add | 16 | Gemmini |
| Sigmoid | 78 | CPU |
| Mul | 79 | CPU |
| Concat | 23 | CPU |
| Split | 11 | CPU |
| Reshape | 8 | CPU |
| MaxPool | 3 | CPU |
| Transpose | 3 | CPU |
| Resize | 2 | CPU |
| Softmax | 2 | CPU |
| Slice | 2 | CPU |
| Sub | 2 | CPU |
| Div | 1 | CPU |

## 层映射摘要（关键层）

| 序号 | 层名 | 类型 | 权重 Shape | Kernel | Stride | Gemmini/CPU | 说明 |
|---|---|---|---|---|---|---|---|
| 0 | conv_0 | Conv2d | [16,3,3,3] | 3×3 | 2 | Gemmini | Backbone stem |
| 1 | conv_1 | Conv2d | [32,16,3,3] | 3×3 | 2 | Gemmini | Backbone stem |
| 2 | conv_2 | Conv2d | [32,32,1,1] | 1×1 | 1 | Gemmini | Pointwise |
| 5 | conv_5 | Conv2d | [64,48,1,1] | 1×1 | 1 | Gemmini | C2f bottleneck merge |
| 6 | conv_6 | Conv2d | [64,64,3,3] | 3×3 | 2 | Gemmini | Downsample |
| 10 | conv_10 | Conv2d | [128,96,1,1] | 1×1 | 1 | Gemmini | C2f merge |
| 11 | conv_11 | Conv2d | [128,128,3,3] | 3×3 | 2 | Gemmini | Downsample |
| 20 | conv_20 | Conv2d | [128,192,1,1] | 1×1 | 1 | Gemmini | C2PSA merge |
| 21 | conv_21 | Conv2d | [256,128,3,3] | 3×3 | 2 | Gemmini | Downsample |
| 30 | conv_30 | Conv2d | [256,384,1,1] | 1×1 | 1 | Gemmini | C2PSA merge |
| 31 | conv_31 | Conv2d | [128,256,1,1] | 1×1 | 1 | Gemmini | SPPF reduce |
| 32 | conv_32 | Conv2d | [256,512,1,1] | 1×1 | 1 | Gemmini | SPPF expand |
| 59 | conv_59 | Conv2d | [10,64,1,1] | 1×1 | 1 | Gemmini | Det head P3 cls |
| 73 | conv_73 | Conv2d | [10,64,1,1] | 1×1 | 1 | Gemmini | Det head P4 cls |
| 86 | conv_86 | Conv2d | [10,64,1,1] | 1×1 | 1 | Gemmini | Det head P5 cls |
| 87 | conv_87 | Conv2d | [1,16,1,1] | 1×1 | 1 | Gemmini | DFL proj |

## Gemmini 映射说明

### 预计映射到 `tiled_conv_auto` 的层

- 所有 3×3 Conv（standard_conv）: stride=1 和 stride=2 均可映射
- 所有 1×1 Conv（pointwise_conv）: 可映射为特殊 case 的 tiled_conv
- 总计 88 个 Conv 层

### 预计映射到 `tiled_matmul_auto` 的层

- matmul_0, matmul_1: 用于 PSA attention 中的 QK^T 和 attention * V
- 总计 2 个 MatMul 层

### 预计留在 CPU/软件的层

- SiLU (Sigmoid + Mul): 78 个 Sigmoid + 79 个 Mul
- Concat: 23 个通道拼接
- Split: 11 个通道分割
- MaxPool: 3 个（SPPF 模块中的 5×5 池化由 3×3 MaxPool 堆叠）
- Resize: 2 个上采样（Neck 中的 2x nearest 上采样）
- Softmax: 2 个（PSA attention 和 DFL 解码）
- Reshape/Transpose: 用于张量形状变换
- 检测头后处理和 NMS

### 不确定算子

- Add (16 个): 残差连接，Gemmini 可用 `tiled_resadd_auto` 加速，但需确认是否适用于所有 Add 场景

## 导出产物

- ONNX 路径: `exports/onnx/train4_best_fp32.onnx`, `exports/onnx/train4_best_int8.onnx`
- params.h 路径: `exports/params/train4_params.h`
- 层参数 JSON: `exports/params/train4_layer_params.json`
- 量化参数表: `exports/quant/train4_quant_params.json`, `exports/quant/train4_quant_params.csv`
- 预处理/后处理规范: `exports/reports/train4_preprocess_postprocess.md`
- FP32 验证报告: `exports/reports/train4_fp32_val_report_2026-03-18.md`
- INT8 验证报告: `exports/reports/train4_int8_val_report.md`（验证完成后生成）

## 已知风险

- `train` 类在验证集中仅 15 个实例，该类别指标统计意义较弱
- INT8 量化 v2 使用 per-channel 权重 + per-tensor 激活对称量化
- 检测头 (model.23) 保持 FP32 以避免分类通道量化归零问题
- DFL 解码中的 Softmax 和后续 MatMul 保持 FP32
- SPPF 模块中 MaxPool 的级联可能在 Gemmini 侧需要特殊处理
