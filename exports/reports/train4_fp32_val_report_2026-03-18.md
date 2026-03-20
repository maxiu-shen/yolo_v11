# Train4 FP32 验证报告

## 基本信息
- 实验名: `train4`
- 验证日期: `2026-03-18`
- 模型: `runs/detect/train4/weights/best.pt`
- 数据集: `ultralytics/cfg/datasets/bdd100k.yaml`
- 数据集根目录: `datasets/BDD100K`
- 类别: `pedestrian, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign`
- 输入尺寸: `640`
- 设备: `CUDA:0`
- 验证输出目录: `runs/detect/train4_fp32_val`

## 验证命令
```powershell
conda run -n yolo_v11 yolo task=detect mode=val model="runs/detect/train4/weights/best.pt" data="ultralytics/cfg/datasets/bdd100k.yaml" device=0 plots=True project="runs/detect" name="train4_fp32_val"
```

## 环境信息
- Ultralytics: `8.3.185`
- Python: `3.10.19`
- PyTorch: `2.0.1+cu118`
- GPU: `NVIDIA GeForce RTX 4060 Ti (8188 MiB)`
- 模型摘要: `YOLO11n fused, 100 layers, 2,584,102 params, 6.3 GFLOPs`

## 数据集覆盖情况
- Images: `10000`
- Instances: `185526`
- Background images: `0`
- Corrupt images: `0`

## 整体 FP32 指标
- Precision: `0.662`
- Recall: `0.410`
- mAP50: `0.449`
- mAP50-95: `0.253`

## 分类别指标
| 类别 | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| pedestrian | 3220 | 13262 | 0.649 | 0.463 | 0.524 | 0.250 |
| rider | 515 | 649 | 0.566 | 0.319 | 0.351 | 0.170 |
| car | 9879 | 102506 | 0.717 | 0.699 | 0.744 | 0.462 |
| truck | 2689 | 4245 | 0.610 | 0.533 | 0.570 | 0.413 |
| bus | 1242 | 1597 | 0.631 | 0.493 | 0.552 | 0.424 |
| train | 14 | 15 | 1.000 | 0.000 | 0.000 | 0.000 |
| motorcycle | 334 | 452 | 0.602 | 0.288 | 0.333 | 0.159 |
| bicycle | 578 | 1007 | 0.490 | 0.337 | 0.342 | 0.169 |
| traffic light | 5653 | 26885 | 0.653 | 0.475 | 0.515 | 0.190 |
| traffic sign | 8221 | 34908 | 0.699 | 0.493 | 0.563 | 0.291 |

## 运行信息
- 总验证耗时: `81.674 s`
- Preprocess: `0.1 ms/image`
- Inference: `1.2 ms/image`
- Postprocess: `0.9 ms/image`

## 与训练日志对比
- 本次独立 FP32 验证结果与 `runs/detect/train4/results.csv` 中第 100 个 epoch 的内置验证结果一致。
- 训练末轮结果: `P=0.66255, R=0.40947, mAP50=0.44852, mAP50-95=0.25231`
- 独立验证结果: `P=0.662, R=0.410, mAP50=0.449, mAP50-95=0.253`
- 结论: `best.pt` 的最终 FP32 结果已确认，可作为后续 `INT8 PTQ`、`INT8 val`、`ONNX export` 与部署产物生成的基线。

## 结果观察
- 当前表现最强的类别是 `car`，其 `mAP50-95=0.462`。
- `truck` 与 `bus` 的结果相对稳定，具备一定可用性。
- `traffic light` 和 `traffic sign` 的样本量很大，但定位质量一般，这很可能是整体 `mAP50-95` 偏低的原因之一。
- `rider`、`motorcycle`、`bicycle` 是当前弱项类别，可能受到类别不均衡和小目标检测难度影响。
- `train` 在验证集中仅有 `15` 个实例，因此该类别指标统计意义较弱。

## 已保存产物
- 曲线图: `runs/detect/train4_fp32_val/BoxF1_curve.png`, `BoxPR_curve.png`, `BoxP_curve.png`, `BoxR_curve.png`
- 混淆矩阵: `runs/detect/train4_fp32_val/confusion_matrix.png`, `confusion_matrix_normalized.png`
- 预测样例: `runs/detect/train4_fp32_val/val_batch0_pred.jpg`, `val_batch1_pred.jpg`, `val_batch2_pred.jpg`

## 下一步
- 该报告可作为开始 `INT8 PTQ` 之前的正式 FP32 基线留档。
