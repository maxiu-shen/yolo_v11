# YOLOv11n Gemmini 部署指南

## 1. 核心结论

**全部 88 个 Conv + 2 个 MatMul 都可以在 Gemmini 上以 INT8 加速。** 检测头在 ONNX Runtime 验证时保持 FP32 是因为 ORT 的输出张量 Concat 混合了 bbox/cls 不同量纲。在 Gemmini 裸机 C 中每层独立计算，不存在此问题。

## 2. 整体架构

```
[输入图像 640×640×3]
       │ (预处理: letterbox, RGB, /255, INT8量化)
       ▼
┌──────────────────────────┐
│  Backbone + Neck (INT8)  │  ← Gemmini tiled_conv_auto / tiled_matmul_auto
│  63 Conv + 2 MatMul      │  ← 量化参数来自 train4_quant_params.json
│  (已在 INT8 ONNX 中量化)  │
└──────────┬───────────────┘
           │ INT8 激活输出 (3 个 scale: P3/P4/P5)
           ▼
┌──────────────────────────┐
│  检测头 Conv (INT8)      │  ← Gemmini tiled_conv_auto
│  25 Conv                 │  ← 量化参数来自 train4_detect_head_quant.json
│  cv2 (bbox): 9 Conv      │
│  cv3 (cls):  15 Conv     │
│  DFL:        1 Conv      │
└──────────┬───────────────┘
           │ INT8 或 FP32 激活输出 (反量化后)
           ▼
┌──────────────────────────┐
│  后处理 (CPU/RISC-V)     │
│  - Concat 3 尺度         │
│  - Split bbox/cls        │
│  - Sigmoid (cls)         │
│  - Softmax + DFL (bbox)  │
│  - Box 解码              │
│  - NMS                   │
└──────────┬───────────────┘
           ▼
      [检测结果]
```

## 3. 量化参数来源

| 模型区域 | Conv 数 | 量化参数来源 | 文件 |
|---|---|---|---|
| Backbone + Neck | 63 Conv + 2 MatMul | INT8 QDQ ONNX (ORT 量化) | `train4_quant_params.json` |
| 检测头 cv2/cv3/DFL | 25 Conv | 独立校准 (FP32 模型 + 200 图) | `train4_detect_head_quant.json` |

两套参数的量化方法一致：
- 权重: INT8 对称 per-channel (`scale = max(abs(w[oc])) / 127`)
- 激活: INT8 对称 per-tensor (`scale = max(abs(activation)) / 127`)
- zero_point = 0 (全部对称)

## 4. 检测头详细结构

### 4.1 cv2 分支 (bbox 回归, 3 个 scale × 3 Conv)

每个 scale 的 cv2 由 3 个 Conv 组成：
```
输入(来自 Neck) → 3×3 Conv+SiLU → 3×3 Conv+SiLU → 1×1 Conv(无激活) → bbox DFL 原始输出 [64ch]
```

| Scale | 层 | 权重 Shape | Kernel | 说明 |
|---|---|---|---|---|
| P3 (80×80) | cv2.0.0 | [64,64,3,3] | 3×3 | + SiLU |
| | cv2.0.1 | [64,64,3,3] | 3×3 | + SiLU |
| | cv2.0.2 | [64,64,1,1] | 1×1 | 无激活，输出 64ch DFL |
| P4 (40×40) | cv2.1.0 | [64,128,3,3] | 3×3 | + SiLU |
| | cv2.1.1 | [64,64,3,3] | 3×3 | + SiLU |
| | cv2.1.2 | [64,64,1,1] | 1×1 | 无激活 |
| P5 (20×20) | cv2.2.0 | [64,256,3,3] | 3×3 | + SiLU |
| | cv2.2.1 | [64,64,3,3] | 3×3 | + SiLU |
| | cv2.2.2 | [64,64,1,1] | 1×1 | 无激活 |

### 4.2 cv3 分支 (分类, 3 个 scale × 5 Conv)

每个 scale 的 cv3 由 5 个 Conv 组成（含 depthwise）：
```
输入 → DW 3×3 Conv+SiLU → 1×1 Conv+SiLU → DW 3×3 Conv+SiLU → 1×1 Conv+SiLU → 1×1 Conv(无激活) → cls 原始输出 [10ch]
```

| Scale | 层 | 权重 Shape | Kernel | 说明 |
|---|---|---|---|---|
| P3 | cv3.0.0.0 | [64,1,3,3] | 3×3 DW | group=64 |
| | cv3.0.0.1 | [64,64,1,1] | 1×1 | |
| | cv3.0.1.0 | [64,1,3,3] | 3×3 DW | group=64 |
| | cv3.0.1.1 | [64,64,1,1] | 1×1 | |
| | cv3.0.2 | [10,64,1,1] | 1×1 | 最终分类 |
| P4 | cv3.1.0.0 | [128,1,3,3] | 3×3 DW | group=128 |
| | cv3.1.0.1 | [64,128,1,1] | 1×1 | |
| | cv3.1.1.0 | [64,1,3,3] | 3×3 DW | group=64 |
| | cv3.1.1.1 | [64,64,1,1] | 1×1 | |
| | cv3.1.2 | [10,64,1,1] | 1×1 | 最终分类 |
| P5 | cv3.2.0.0 | [256,1,3,3] | 3×3 DW | group=256 |
| | cv3.2.0.1 | [64,256,1,1] | 1×1 | |
| | cv3.2.1.0 | [64,1,3,3] | 3×3 DW | group=64 |
| | cv3.2.1.1 | [64,64,1,1] | 1×1 | |
| | cv3.2.2 | [10,64,1,1] | 1×1 | 最终分类 |

### 4.3 DFL Conv

```
Softmax 输出 → 1×1 Conv → DFL 解码值
```

| 层 | 权重 Shape | 说明 |
|---|---|---|
| dfl/conv | [1,16,1,1] | 固定权重 [0,1,2,...,15]，无 bias |

### 4.4 SiLU 激活函数处理

SiLU(x) = x × sigmoid(x)，在 ONNX 中分解为 Sigmoid + Mul。

Gemmini 处理方式：
- Conv 输出 INT8 → 反量化为 FP32 → SiLU → 量化为 INT8 → 下一层 Conv 输入
- 或使用查找表 (LUT) 直接在 INT8 域实现 SiLU 近似

## 5. Gemmini 侧执行流程

### 5.1 Backbone + Neck (conv_0 ~ conv_48)

```c
// 量化参数来自 train4_params.h (已有 INT8 权重 + per-channel scales)
for (int i = 0; i < 49; i++) {  // backbone+neck layers
    tiled_conv_auto(
        ...,
        YOLOV11N_LAYERS[i].weight,
        YOLOV11N_LAYERS[i].bias,
        ...
    );
    // 中间非线性: SiLU / MaxPool / Concat / Split / Resize / Add
    // → CPU 处理
}
```

### 5.2 检测头 Conv (conv_49 ~ conv_87 + DFL)

```c
// 量化参数来自 train4_detect_head_quant.json
// 权重来自 train4_params.h 中的 CONV_49_WEIGHT 等 (NAIVE_SCALE 量化)
// 但 Gemmini 部署时应使用 detect_head_quant.json 中的 per-channel scales

// P3 cv2 分支
tiled_conv_auto(neck_p3_out, cv2_0_0_weight, cv2_0_0_bias, ...);
silu(cv2_0_0_out);
tiled_conv_auto(cv2_0_0_out, cv2_0_1_weight, cv2_0_1_bias, ...);
silu(cv2_0_1_out);
tiled_conv_auto(cv2_0_1_out, cv2_0_2_weight, cv2_0_2_bias, ...);
// cv2_0_2_out → 64ch DFL 原始输出

// P3 cv3 分支
tiled_conv_auto(neck_p3_out, cv3_0_0_0_weight, ...);  // DW conv
silu(cv3_0_0_0_out);
tiled_conv_auto(cv3_0_0_0_out, cv3_0_0_1_weight, ...);
silu(cv3_0_0_1_out);
// ... 省略中间层 ...
tiled_conv_auto(cv3_0_1_1_out, cv3_0_2_weight, ...);
// cv3_0_2_out → 10ch 分类 logits

// 对 P4, P5 重复上述流程
```

### 5.3 后处理 (CPU)

```c
// 1. Concat: 合并 3 个 scale 的输出
//    bbox_raw: [64, 8400] = cat([64,6400], [64,1600], [64,400])
//    cls_raw:  [10, 8400] = cat([10,6400], [10,1600], [10,400])

// 2. DFL 解码 bbox
for (int i = 0; i < 8400; i++) {
    // 对每个 anchor: 4 组 × 16 bin → softmax → 加权求和 → xywh
    for (int j = 0; j < 4; j++) {
        softmax(bbox_raw[j*16:(j+1)*16, i], 16);
        // DFL conv: 与 [0,1,...,15] 加权求和
        decoded_box[j][i] = dot_product(softmax_out, dfl_weights);
    }
}

// 3. Sigmoid 分类
for (int i = 0; i < 8400 * 10; i++) {
    cls_scores[i] = 1.0f / (1.0f + expf(-cls_raw[i]));
}

// 4. Box 解码: DFL 值 → 实际坐标 (考虑 stride 和 anchor 偏移)
// 5. NMS: 非极大值抑制
```

## 6. Depthwise Conv 注意事项

cv3 分支的 DW Conv (group = in_channels) 在 Gemmini 上的处理：

- `tiled_conv_auto` 支持 group 参数
- DW Conv 等价于 group=in_channels 的分组卷积
- 权重 shape 为 [C, 1, 3, 3]，每个通道独立卷积
- 如果 Gemmini 不直接支持 DW Conv，可以用 CPU 循环实现

## 7. 量化数据流示意

```
             Backbone/Neck              Detect Head
             ───────────               ───────────
输入:         FP32 → Q(INT8)            INT8 (来自上层)
Conv 权重:    INT8 (per-channel)        INT8 (per-channel)
Conv 计算:    INT32 累加                 INT32 累加
Conv 输出:    requant → INT8            requant → INT8
SiLU:        INT8→FP32→SiLU→INT8       INT8→FP32→SiLU→INT8
最终输出:     INT8 → 传给检测头          INT8 → 反量化→FP32 → CPU 后处理
```

## 8. 交付文件对应关系

| Gemmini 部署阶段 | 文件 | 说明 |
|---|---|---|
| Backbone/Neck Conv 权重+scale | `train4_params.h` (conv_0~conv_48) | INT8 权重数组 + per-channel WEIGHT_SCALES + REQUANT_SCALES |
| Backbone/Neck 层结构 | `train4_layer_params.json` | kernel/stride/pad/group 等 |
| 检测头 Conv 权重 | `train4_params.h` (conv_49~conv_87) | NAIVE_SCALE 量化的 INT8 权重 |
| 检测头量化参数 | `train4_detect_head_quant.json` | 独立校准的 per-channel weight scale + activation scale |
| 类别映射 | `classes.txt` | 10 类名称 |
| 预处理规范 | `train4_preprocess_postprocess.md` | letterbox, RGB, /255, NCHW |
| 后处理规范 | `train4_preprocess_postprocess.md` | DFL, Sigmoid, NMS |

## 9. 关于 params.h 中检测头权重的说明

`params.h` 中 conv_49~conv_87 的 INT8 权重使用 `NAIVE_SCALE` 量化：
- `NAIVE_SCALE = max(abs(w)) / 127` (per-tensor, 单个标量)
- 这是一个粗略的全层 scale

`train4_detect_head_quant.json` 提供了更精确的参数：
- 权重: per-channel scales (每个输出通道独立 scale)
- 激活: per-layer scales (基于 200 张校准图)

**Gemmini 部署时应优先使用 `train4_detect_head_quant.json` 中的 per-channel scales** 来重新量化检测头权重，以获得更好的精度。

## 10. 已知限制

- `train` 类 (火车) 在 BDD100K val 中仅 15 个实例，该类指标不具统计意义
- DFL 的 Softmax 在 INT8 域精度可能不足，建议 DFL 解码全程在 CPU FP32 执行
- PSA 注意力的 2 个 MatMul 在 Gemmini 上通过 `tiled_matmul_auto` 执行
- Resize (最近邻上采样 2x) 建议在 CPU 实现
