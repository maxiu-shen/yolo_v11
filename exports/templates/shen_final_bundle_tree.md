# Windows Final Bundle Tree

本文档定义 Windows 侧 `yolo_v11` 工作区在完成 `BDD100K` 训练/微调、`INT8 PTQ`、验证与导出后，建议同步回 Linux/Gemmini 工作区的最终交付目录结构。

建议将该交付包放在：

`yolo_v11/exports/final_bundle/`

## 推荐目录树

```text
final_bundle/
├─ model/
│  ├─ best.pt
│  └─ best.onnx
├─ params/
│  ├─ params.h
│  ├─ layer_params.json
│  └─ weights_manifest.txt
├─ quant/
│  ├─ quant_params.json
│  ├─ quant_params.csv
│  └─ calibration_notes.md
├─ meta/
│  ├─ classes.txt
│  ├─ preprocess.md
│  ├─ postprocess.md
│  ├─ model_summary.md
│  └─ export_log.md
├─ tools/
│  ├─ shen_export_params.py
│  ├─ shen_export_quant.py
│  └─ shen_verify_bundle.py
└─ checks/
   ├─ fp32_val_metrics.txt
   ├─ int8_val_metrics.txt
   └─ bundle_checklist.md
```

## 各目录说明

### `model/`
- `best.pt`
  Windows 侧训练或微调得到的最佳权重，保留用于追溯与重新导出。
- `best.onnx`
  提供给 Linux/Gemmini 侧做结构核对、算子映射与导出一致性检查。

### `params/`
- `params.h`
  后续 `software/gemmini-rocc-tests/bareMetalC/shen_yolov11n.c` 直接消费的核心导出文件。
- `layer_params.json`
  记录每层结构参数、卷积配置、shape 和导出元信息，方便人工排查与自动脚本核对。
- `weights_manifest.txt`
  记录权重来源、导出脚本版本和关键文件摘要。

### `quant/`
- `quant_params.json`
  量化参数主文件，建议作为机器可读的标准版本。
- `quant_params.csv`
  便于快速人工查看和表格分析。
- `calibration_notes.md`
  记录 PTQ 校准数据集、方法、异常层和特殊处理说明。

### `meta/`
- `classes.txt`
  类别顺序与名称，确保 Windows 侧与 Linux/Gemmini 侧一致。
- `preprocess.md`
  输入尺寸、letterbox、颜色顺序、归一化方式、张量布局等说明。
- `postprocess.md`
  输出张量语义、decode 逻辑、阈值、NMS 规则等说明。
- `model_summary.md`
  模型结构摘要、层映射摘要、Gemmini 或 CPU 预计执行划分。
- `export_log.md`
  导出时间、命令、脚本版本、已知问题与注意事项。

### `tools/`
- `shen_export_params.py`
  导出 `params.h` 的脚本。
- `shen_export_quant.py`
  导出量化参数表的脚本。
- `shen_verify_bundle.py`
  检查交付包完整性和字段一致性的脚本。

> 如果这些脚本仍在迭代中，也建议把实际使用版本一并放进交付包，保证后续可复现。

### `checks/`
- `fp32_val_metrics.txt`
  记录 FP32 验证指标。
- `int8_val_metrics.txt`
  记录 INT8 验证指标。
- `bundle_checklist.md`
  交付前勾选清单，确认 `ONNX`、`params.h`、量化参数、预处理/后处理说明均齐全。

## 最低必需文件

如果只保留最小交付集合，至少应包含：

```text
final_bundle/
├─ model/best.onnx
├─ params/params.h
├─ quant/quant_params.json
├─ meta/classes.txt
├─ meta/preprocess.md
├─ meta/postprocess.md
└─ meta/model_summary.md
```

## 同步到 Linux/Gemmini 工作区后的用途

- `best.onnx`
  用于核对网络结构、层顺序和导出一致性。
- `params.h`
  作为 `shen_yolov11n.c` 的主要参数输入。
- `quant_params.json`
  用于确认每层 `scale`、`zero-point`、输出重定标关系。
- `classes.txt`
  保证类别顺序与检测结果解释一致。
- `preprocess.md` / `postprocess.md`
  确保 Linux/Gemmini 侧的输入输出语义与 Windows 侧验证一致。
- `model_summary.md`
  帮助快速完成 Gemmini/CPU 算子映射与代码实现。

## 交付建议

- 优先同步整个 `final_bundle/`，不要只拷贝 `best.onnx`
- 若 `params.h` 或量化参数表发生变化，必须同时更新 `model_summary.md` 和 `export_log.md`
- 若使用自定义导出脚本，建议把脚本版本和命令行参数一起写入 `export_log.md`
- 在 Linux/Gemmini 侧开始写 `shen_yolov11n.c` 前，先检查 `final_bundle/` 是否满足最低必需文件集合
