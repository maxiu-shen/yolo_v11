# Train4 主线流程执行总结

## 1. 流程全景

按照 AGENTS.md 定义的标准主线流程：

```
BDD100K → train/finetune → FP32 val → INT8 PTQ → INT8 val → ONNX export → params.h export → quant table export
```

实际执行中各步骤的顺序做了调整——先导出 FP32 ONNX 再做 PTQ（因为 ONNX Runtime 量化需要 FP32 ONNX 作为输入），完整执行链路为：

```
train4 训练完成 (已有 best.pt)
  → FP32 验证 (yolo val CLI)
  → FP32 ONNX 导出 (yolo export CLI)
  → INT8 PTQ (ONNX Runtime quantize_static)
  → INT8 验证 (Ultralytics YOLO.val() + GPU)
  → params.h 导出 (自写脚本)
  → 量化参数表导出 (自写脚本)
  → 检测头独立量化参数导出 (自写脚本)
  → 预处理/后处理规范编写
  → 模型结构摘要编写
  → Gemmini 部署指南编写
  → 最终交叉验证
```

---

## 2. 各步骤详细说明

### 2.1 FP32 验证

- **执行时间**: 2026-03-18
- **命令**: `yolo task=detect mode=val model=runs/detect/train4/weights/best.pt data=bdd100k.yaml device=0`
- **结果**: P=0.662, R=0.410, mAP50=0.449, mAP50-95=0.253
- **产物**: `exports/reports/train4_fp32_val_report_2026-03-18.md`

### 2.2 FP32 ONNX 导出

- **命令**: `yolo task=detect mode=export model=best.pt format=onnx imgsz=640`
- **输入**: `runs/detect/train4/weights/best.pt` (5.2 MB)
- **输出**: `exports/onnx/train4_best_fp32.onnx` (10.1 MB)
- **规格**: 输入 [1,3,640,640]，输出 [1,14,8400]，320 个节点，88 Conv + 2 MatMul

### 2.3 INT8 PTQ 量化

- **脚本**: `tools/shen_int8_ptq.py`
- **输入**: FP32 ONNX + BDD100K 训练集 500 张校准图
- **工具**: ONNX Runtime 1.15.1 `quantize_static`
- **关键参数**: QDQ 格式、per-channel 权重、MinMax 校准、对称量化
- **输出**: `exports/onnx/train4_best_int8.onnx` (4.3 MB, 压缩比 2.38x)

**遇到的问题与解决**:

| 版本 | 问题 | 解决方案 |
|---|---|---|
| v1 (per-tensor, 200 图, 全模型量化) | 分类通道输出全部为零 | 诊断发现输出 Concat 混合 bbox/cls 导致 cls 被量化归零 |
| v2 (per-channel, 500 图, 排除检测头) | 无 | 排除 model.23 的所有节点，backbone/neck 正常量化 |

### 2.4 INT8 验证

- **脚本**: `tools/shen_int8_val.py`
- **方法**: Ultralytics `YOLO.val()` 加载 INT8 ONNX，device=0 (GPU)
- **耗时**: ~1063 秒 (约 18 分钟，~9.4 it/s)
- **结果**: P=0.654, R=0.380, mAP50=0.418, mAP50-95=0.233
- **精度变化**: mAP50 下降 6.82%, mAP50-95 下降 7.92%（正常范围）
- **产物**: `exports/reports/train4_int8_val_report.md`

**遇到的问题与解决**:

| 方案 | 方式 | 问题 |
|---|---|---|
| yolo CLI + conda run + CPU | ~15 it/s | exit code -1，~500-1055 图后崩溃 |
| yolo CLI + 直接执行 + CPU | ~15 it/s | 0xC0000005 访问违规，ORT 1.15.1 CPU INT8 bug |
| 自写脚本 + conda run + CPU | ~15 it/s | 同上，~100-340 图后崩溃 |
| **自写脚本 + 直接 python.exe + GPU** | **~9.4 it/s** | **稳定完成全部 10000 张** |

结论：ORT 1.15.1 的 CPUExecutionProvider 运行 INT8 QDQ 模型存在已知 bug，GPU 方案稳定。

### 2.5 params.h 导出

- **脚本**: `tools/shen_export_params.py`
- **输入**: INT8 QDQ ONNX 模型
- **输出**:
  - `exports/params/train4_params.h` (11.2 MB) — C 头文件
  - `exports/params/train4_layer_params.json` (334 KB) — 层参数 JSON
  - `exports/params/classes.txt` — 10 类名称
- **内容**:
  - 90 层全部权重（65 层 INT8 + 25 层 FP32 检测头 naive 量化）
  - Per-channel WEIGHT_SCALES / REQUANT_SCALES 浮点数组（backbone/neck）
  - FP32 检测头的 NAIVE_SCALE + BIAS_FP32 数组
  - LayerParams 结构体汇总表（含 quantized 标志）

**v2 修复**:
- 原始版本把 per-channel scale 截断为单个标量 → 修复为完整数组
- 原始版本用错误的默认 scale 1.0 转换 FP32 检测头 → 修复为 naive 对称量化
- 增加了 `quantized` 字段区分量化层和 FP32 层

### 2.6 量化参数表导出

- **脚本**: `tools/shen_export_quant.py`
- **输入**: INT8 QDQ ONNX 模型
- **输出**:
  - `exports/quant/train4_quant_params.json` (889 KB) — 完整量化参数
  - `exports/quant/train4_quant_params.csv` (10 KB) — 快速查看表
  - `exports/quant/train4_calibration_notes.md` (4.1 KB) — 校准说明
- **内容**: 90 层的 input/weight/output scale + zero_point + requant_scale + Gemmini 映射

**v2 修复**:
- JSON 元数据更正 (per_channel=True, num_images=500, detect_head_excluded=True)
- 每层增加 `quantized` 布尔标志
- Per-channel requant_scale 改为数组
- CSV 增加 weight_granularity 和 weight_scale_count 列

### 2.7 检测头独立量化参数导出

- **脚本**: `tools/shen_export_detect_head_quant.py`
- **输入**: FP32 ONNX + 200 张校准图
- **方法**: 在 FP32 模型中插入中间输出节点，收集检测头各 Conv 的输入/输出激活统计，计算 per-layer 激活 scale + per-channel 权重 scale
- **输出**: `exports/quant/train4_detect_head_quant.json` (177 KB)
- **目的**: 检测头在 ONNX Runtime 中保持 FP32 是 ORT 的限制，但在 Gemmini 裸机 C 中每层独立计算，可以使用这些参数以 INT8 运行

### 2.8 文档编写

| 文档 | 路径 | 内容 |
|---|---|---|
| 预处理/后处理规范 | `exports/reports/train4_preprocess_postprocess.md` | letterbox、RGB、/255、NCHW、DFL 解码、Sigmoid、NMS |
| 模型结构摘要 | `exports/reports/train4_model_summary.md` | 模型规模、精度、算子统计、Gemmini/CPU 映射 |
| Gemmini 部署指南 | `exports/reports/train4_gemmini_deploy_guide.md` | 全模型 INT8 执行架构、检测头部署方案、代码框架 |

### 2.9 最终交叉验证

- **脚本**: `tools/shen_final_check.py`
- **检查内容**: 文件完整性、ONNX shape 一致性、层数匹配、per-channel scale 正确性、JSON/CSV/params.h 交叉比对、INT8 推理输出数值验证、文档内容一致性
- **结果**: 0 错误，0 警告

---

## 3. 产物清单

### 3.1 模型文件

| 文件 | 大小 | 说明 |
|---|---|---|
| `exports/onnx/train4_best_fp32.onnx` | 10.1 MB | FP32 ONNX 模型 |
| `exports/onnx/train4_best_int8.onnx` | 4.3 MB | INT8 QDQ ONNX (backbone/neck 量化, 检测头 FP32) |

### 3.2 参数文件

| 文件 | 大小 | 说明 |
|---|---|---|
| `exports/params/train4_params.h` | 11.2 MB | Gemmini C 头文件，90 层全部权重 + 量化参数 + LayerParams 表 |
| `exports/params/train4_layer_params.json` | 334 KB | 层参数 JSON（不含权重数据） |
| `exports/params/classes.txt` | 83 B | 10 类名称映射 |

### 3.3 量化参数文件

| 文件 | 大小 | 说明 |
|---|---|---|
| `exports/quant/train4_quant_params.json` | 889 KB | 90 层完整量化参数 (backbone/neck 来自 ORT, 检测头标记 FP32) |
| `exports/quant/train4_quant_params.csv` | 10 KB | 量化参数快速查看表 |
| `exports/quant/train4_detect_head_quant.json` | 177 KB | 检测头 25 Conv 独立量化参数 (Gemmini 全 INT8 部署用) |
| `exports/quant/train4_calibration_notes.md` | 4.1 KB | 校准方法说明 (v2) |

### 3.4 报告文档

| 文件 | 说明 |
|---|---|
| `exports/reports/train4_fp32_val_report_2026-03-18.md` | FP32 验证结果 (baseline) |
| `exports/reports/train4_int8_val_report.md` | INT8 验证结果 + FP32 对比 |
| `exports/reports/train4_model_summary.md` | 模型架构、精度、算子统计、Gemmini 映射 |
| `exports/reports/train4_preprocess_postprocess.md` | 预处理/后处理完整规范 |
| `exports/reports/train4_gemmini_deploy_guide.md` | Gemmini 部署完整指南 |

### 3.5 工具脚本

| 脚本 | 作用 |
|---|---|
| `tools/shen_int8_ptq.py` | INT8 PTQ 量化 (v2: per-channel, 排除检测头) |
| `tools/shen_int8_val.py` | INT8 验证 (Ultralytics YOLO.val, GPU) |
| `tools/shen_int8_debug.py` | INT8 输出诊断 (对比 FP32 vs INT8 原始输出值) |
| `tools/shen_export_params.py` | params.h 导出 (v2: per-channel scale 数组 + FP32 检测头处理) |
| `tools/shen_export_quant.py` | 量化参数表导出 (v2: 含量化状态标记) |
| `tools/shen_export_detect_head_quant.py` | 检测头独立量化参数导出 |
| `tools/shen_analyze_onnx_output.py` | ONNX 图结构分析 (追踪输出节点) |
| `tools/shen_detect_head_analysis.py` | 检测头结构与激活范围分析 |
| `tools/shen_final_check.py` | 最终交叉验证 (7 项检查) |
| `tools/shen_verify_exports.py` | 导出产物快速验证 |

---

## 4. 关键精度数据

### FP32 vs INT8 对比

| 指标 | FP32 | INT8 | 差值 | 下降% |
|---|---|---|---|---|
| Precision | 0.662 | 0.654 | -0.008 | -1.15% |
| Recall | 0.410 | 0.380 | -0.030 | -7.42% |
| mAP50 | 0.449 | 0.418 | -0.031 | -6.82% |
| mAP50-95 | 0.253 | 0.233 | -0.020 | -7.92% |

### INT8 分类别结果

| 类别 | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| pedestrian | 0.613 | 0.436 | 0.487 | 0.228 |
| rider | 0.576 | 0.291 | 0.315 | 0.149 |
| car | 0.666 | 0.698 | 0.726 | 0.443 |
| truck | 0.577 | 0.522 | 0.542 | 0.390 |
| bus | 0.598 | 0.474 | 0.530 | 0.399 |
| train | 1.000 | 0.000 | 0.000 | 0.000 |
| motorcycle | 0.714 | 0.171 | 0.257 | 0.123 |
| bicycle | 0.483 | 0.275 | 0.295 | 0.145 |
| traffic light | 0.638 | 0.456 | 0.496 | 0.180 |
| traffic sign | 0.680 | 0.472 | 0.537 | 0.273 |

---

## 5. 遇到的主要技术问题

### 问题 1: INT8 模型分类输出全零

- **现象**: v1 量化后 INT8 模型的分类通道 (后 10 行) 全部为 0.000000
- **根因**: 输出张量 Concat 合并 bbox (0~637) 和 cls (0~0.86)，per-tensor scale 被 bbox 主导，cls 值低于量化步长全部归零
- **解决**: 排除检测头 model.23 的所有节点不做量化，backbone/neck 正常量化

### 问题 2: CPU 推理崩溃 (0xC0000005)

- **现象**: ORT 1.15.1 CPUExecutionProvider 运行 INT8 QDQ 模型时随机崩溃
- **尝试**: conda run / 直接 python / yolo CLI，均在 500-1300 图后崩溃
- **解决**: 使用 GPU (CUDAExecutionProvider) 推理，稳定完成全部 10000 张

### 问题 3: per-channel scale 导出截断

- **现象**: v1 导出脚本只取 `scale.flat[0]`，丢失其他通道的 scale
- **解决**: v2 修复为完整存储 per-channel scale 数组 + 对应的 requant_scale 数组

---

## 6. 面向 Gemmini 交付物总结

Windows 工作区向 Linux/Gemmini 工作区交付的完整产物：

```
exports/
├── onnx/
│   ├── train4_best_fp32.onnx              ← FP32 参考模型
│   └── train4_best_int8.onnx              ← INT8 模型 (backbone/neck 量化)
├── params/
│   ├── train4_params.h                    ← 全部 90 层 C 头文件 (权重+scale+结构)
│   ├── train4_layer_params.json           ← 层参数索引
│   └── classes.txt                        ← 类别映射
├── quant/
│   ├── train4_quant_params.json           ← backbone/neck 量化参数
│   ├── train4_quant_params.csv            ← 快速查看
│   ├── train4_detect_head_quant.json      ← 检测头独立量化参数 (Gemmini 全 INT8 用)
│   └── train4_calibration_notes.md        ← 校准方法说明
└── reports/
    ├── train4_fp32_val_report_2026-03-18.md
    ├── train4_int8_val_report.md
    ├── train4_model_summary.md
    ├── train4_preprocess_postprocess.md
    └── train4_gemmini_deploy_guide.md     ← Gemmini 部署完整指南
```

全部满足 AGENTS.md 中的完成判定条件。
