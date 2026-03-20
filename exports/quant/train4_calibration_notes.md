# Train4 INT8 PTQ 校准说明 (v2)

## 版本历史

| 版本 | 日期 | 变更 |
|---|---|---|
| v1 | 2026-03-18 | per-tensor symmetric, 200 张校准, 全模型量化 → 分类输出全零 |
| v2 | 2026-03-20 | per-channel 权重, 500 张校准, 排除检测头 model.23 → 精度正常 |

## 校准方法

- 工具: `ONNX Runtime 1.15.1` (`onnxruntime.quantization.quantize_static`)
- 量化格式: `QDQ` (QuantizeLinear / DequantizeLinear)
- 校准算法: `MinMax`
- 权重类型: `INT8` (symmetric, **per-channel**)
- 激活类型: `INT8` (symmetric, per-tensor)
- Per-channel: `是`（权重 per-channel, 激活 per-tensor）

## 校准数据

- 来源: `datasets/BDD100K/images/train`
- 抽样数量: `500 张`
- 抽样方式: 随机抽样 (seed=42)
- 预处理: 与推理一致（letterbox 640×640, RGB, /255.0, NCHW float32）

## 量化配置

```python
quantize_static(
    model_input="exports/onnx/train4_best_fp32.onnx",
    model_output="exports/onnx/train4_best_int8.onnx",
    calibration_data_reader=calib_reader,
    quant_format=QuantFormat.QDQ,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8,
    calibrate_method=CalibrationMethod.MinMax,
    per_channel=True,
    nodes_to_exclude=exclude_nodes,  # 排除 model.23 检测头节点
    extra_options={
        "ActivationSymmetric": True,
        "WeightSymmetric": True,
    },
)
```

## 检测头排除说明

### 问题

v1 中对全模型量化导致分类通道输出全部为零：
- 输出张量 `[1, 14, 8400]`：行 0-3 为 bbox (值域 0~637)，行 4-13 为 cls (值域 0~0.86)
- 检测头 Concat 将 bbox 和 cls 合并为单一张量
- per-tensor 量化 scale 被 bbox 大值主导 (scale ≈ 637/127 ≈ 5.0)
- cls 值 (max=0.86) 远小于一个量化步长，全部归零

### 解决方案

排除检测头 `model.23` 下所有节点不做量化（保持 FP32 计算）：
- cv2.0/cv2.1/cv2.2 (bbox 回归 Conv)
- cv3.0/cv3.1/cv3.2 (分类 Conv)
- DFL (Distribution Focal Loss 解码)
- 后处理 (Concat/Split/Sigmoid/Sub/Add/Div/Mul)
- 排除节点总数: ~70 个

### 影响

- 检测头占模型整体计算量比例很小，FP32 保留对推理性能影响可忽略
- Backbone/Neck 的 88 个 Conv + 2 个 MatMul 中的大部分仍被量化
- INT8 ONNX 大小: 4.3 MB (vs FP32 10.1 MB, 压缩比 2.38x)
- 对 Gemmini 部署：backbone/neck 层仍可使用 INT8 加速，检测头层使用 CPU FP32 或单独量化

## 量化结果

- FP32 ONNX: `10.1 MB`
- INT8 ONNX: `4.3 MB`
- 压缩比: `2.38x`
- QuantizeLinear 节点数: `248`
- DequantizeLinear 节点数: `375`
- 量化计算层: `90`（88 Conv + 2 MatMul）

## INT8 验证精度

| 指标 | FP32 | INT8 | 差值 | 下降% |
|---|---|---|---|---|
| Precision | 0.662 | 0.654 | -0.008 | -1.15% |
| Recall | 0.410 | 0.380 | -0.030 | -7.42% |
| mAP50 | 0.449 | 0.418 | -0.031 | -6.82% |
| mAP50-95 | 0.253 | 0.233 | -0.020 | -7.92% |

## 对称量化说明

本次 PTQ 使用对称量化（symmetric），即：
- `zero_point = 0`（所有层）
- `scale` 由 MinMax 校准确定
- 量化公式: `x_int8 = round(x_fp32 / scale)`
- 反量化公式: `x_fp32 = x_int8 * scale`

对称量化的优势：
- 零点为 0，Gemmini 硬件上无需额外偏移计算
- 简化 requant 逻辑
- 与 Gemmini 的 `tiled_conv_auto` / `tiled_matmul_auto` 更兼容

## 面向 Gemmini 的注意事项

- Backbone/Neck 层使用 INT8 per-channel 权重量化参数（每个输出通道一个 scale）
- 检测头 (model.23) 在 ONNX 中保持 FP32，Gemmini 部署时需单独处理
- 建议 Gemmini 侧对检测头层使用更精细的自定义量化或 CPU 软件实现
- 量化参数表中仍包含所有 90 个 Conv/MatMul 层的参数信息

## 已知问题

- ONNX Runtime 1.15.1 的 CPUExecutionProvider 运行 INT8 QDQ 模型存在访问违规 (0xC0000005) bug
- Entropy 校准方法在 ORT 1.15.1 上同样会触发 0xC0000005 崩溃
- GPU 验证 (device=0) 稳定运行，推荐使用 GPU 进行 INT8 验证
- `train` 类在验证集中仅 15 个实例，该类别 mAP=0 不具统计意义

## 脚本

- PTQ 脚本: `tools/shen_int8_ptq.py`
- 量化参数导出: `tools/shen_export_quant.py`
- INT8 验证: `tools/shen_int8_val.py`
- 输出诊断: `tools/shen_int8_debug.py`
