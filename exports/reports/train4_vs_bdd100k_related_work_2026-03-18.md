# Train4 与 BDD100K 相关工作的对比报告

## 说明

本报告只收录基于 `BDD100K` 的、能够追溯到论文、官方仓库或官方文档的结果。

为保证真实性，本报告对参数量采用以下规则：

- 能从本地实测、官方文档、官方仓库直接确认的，写明确数值。
- 原论文或公开页面未披露参数量的，明确标记为 `未公开`。
- 不使用我主观估算的参数量去填表。

同时需要注意，不同工作之间的横向比较并不完全公平，主要原因包括：

- 指标定义不同：有的使用 `AP / AP50`，有的使用 `mAP50`。
- 类别设置不同：有的是完整 `BDD100K 10-class`，有的是 `7-class`、`3-class` 子集。
- 任务不同：有的工作是纯检测，有的是多任务感知模型。
- 模型规模、输入尺寸、训练策略和数据划分细节并不完全一致。

## 你的 Train4 基线

- 模型: `YOLO11n`
- 任务: `BDD100K 10-class` 检测
- 权重: `runs/detect/train4/weights/best.pt`
- FP32 独立验证结果:
  - Precision: `0.662`
  - Recall: `0.410`
  - mAP50: `0.449`
  - mAP50-95(AP): `0.253`
- 参数量:
  - 本地验证日志实测: `2,584,102`
  - Ultralytics 官方 `YOLO11n` 公布值: `2.6M`

## 表 1：完整或接近完整 BDD100K 检测结果对比

| 年份 | 工作 | 模型 | 任务口径 | 主要指标 | 参数量 | 参数说明 | 结论性对比 |
|---|---|---|---|---|---:|---|---|
| 2026 | 你的 `train4` | `YOLO11n` | `BDD100K 10-class detection` | `AP=25.3, AP50=44.9, P=66.2, R=41.0` | `2.58M` | 本地 `FP32 val` 实测 | 轻量，AP 已高于部分早期/轻量基线 |
| 2021 | YOLOX | `YOLOX (CSPDarknet53)` | `BDD100K detection` | `AP=12.6, AP50=26.2, AP75=11.1` | `63.7M` | YOLOX 官方 `YOLOX-Darknet53` Model Zoo | 指标明显低于 `train4`，参数量远大于 `train4` |
| 2024 | SES-YOLOv8n | `SES-YOLOv8n` | `BDD100K detection` | `AP=22.3, AP50=41.9` | `未公开` | 论文表格未给出；仅能确认其基线来自 `YOLOv8n` 家族 | `train4` 的 AP、AP50 均略高 |
| 2025 | DGSS-YOLOv8s | `DGSS-YOLOv8s` | `BDD100K detection` | `AP=24.0, AP50=51.3` | `未公开` | 论文表格未给出；仅能确认其基线来自 `YOLOv8s` 家族 | `train4` 的 AP 略高，但 AP50 更低 |
| 2020 | Dynamic R-CNN | `RN101` | `BDD100K detection` | `AP=24.9, AP50=42.4, AP75=22.8` | `未公开` | 引用表未给出 | `train4` 的 AP50 更高，AP 也略高 |
| 2022 | ViTDet | `ViT-B` | `BDD100K detection` | `AP=21.1, AP50=39.0, AP75=19.4` | `未公开` | 引用表未给出 | `train4` 的 AP、AP50 均更高 |
| 2022 | DINO | `RN101` | `BDD100K detection` | `AP=27.0, AP50=52.7, AP75=23.5` | `未公开` | 引用表未给出 | `train4` 低于该强基线 |
| 2023 | CO-DETR | `RN101` | `BDD100K detection` | `AP=31.5, AP50=55.1, AP75=30.1` | `未公开` | 引用表未给出 | `train4` 明显低于该强基线 |
| 2023 | DiffusionDet | `RN101` | `BDD100K detection` | `AP=27.9, AP50=53.8, AP75=24.9` | `未公开` | 引用表未给出 | `train4` 低于该基线 |
| 2023 | DDQ | `RN101` | `BDD100K detection` | `AP=26.4, AP50=50.0, AP75=24.8` | `未公开` | 引用表未给出 | `train4` 略低于该基线 |
| 2024 | LSFM | `ConvMLP-Pin` | `BDD100K detection` | `AP=28.2, AP50=55.7, AP75=24.4` | `未公开` | 引用表未给出 | `train4` 低于该基线 |
| 2025 | DDRN | `RN101` | `BDD100K detection` | `AP=33.1, AP50=57.4, AP75=31.9` | `未公开` | 引用表未给出 | `train4` 明显低于当前强结果 |

## 表 2：BDD100K 子集任务或非完全同口径工作

| 年份 | 工作 | 模型 | 任务口径 | 主要指标 | 参数量 | 参数说明 | 备注 |
|---|---|---|---|---|---:|---|---|
| 2021 | VRU 论文 | `YOLOv3-416` | `BDD100K 10-class test subset` | `mAP50=45.15%` | `未公开` | 文中未给出可直接核对的参数总量 | 和 `train4` 的 `AP50=44.9` 接近，但指标协议不同 |
| 2021 | VRU 论文 | `YOLOv4-416` | `BDD100K 7-class` | `mAP50=52.21%` | `未公开` | 文中未给出可直接核对的参数总量 | 只做 `7-class`，不能与完整 `10-class` 直接横比 |
| 2021 | YOLOP | `YOLOP` | `BDD100K multi-task driving perception` | `Recall=89.2%, mAP50=76.5%, 41 FPS` | `未公开` | 官方 README/Hub 页未给出参数总量 | 这是多任务模型，不等同于纯检测模型 |
| 2023 | 改进检测论文 | `YOLOv5` | `BDD100K 3-class(Person/Bike/Car)` | `mAP=59.0%` | `未公开` | 论文未说明具体 YOLOv5 尺寸 | 仅三类，不可与 `10-class` 直接比较 |
| 2023 | 改进检测论文 | `PP-YOLOE` | `BDD100K 3-class(Person/Bike/Car)` | `mAP=59.8%` | `未公开` | 引用表未给出 | 仅三类 |
| 2023 | 改进检测论文 | `YOLOv5(double head)` | `BDD100K 3-class(Person/Bike/Car)` | `mAP=63.1%` | `未公开` | 引用表未给出 | 为改进版结构 |
| 2023 | 改进检测论文 | `YOLOv5(double head + micro target detection layer)` | `BDD100K 3-class(Person/Bike/Car)` | `mAP=65.7%` | `未公开` | 引用表未给出 | 为改进版结构 |
| 2023 | 改进检测论文 | `YOLOv5(double head + micro target detection layer + VF loss + ML-AFP)` | `BDD100K 3-class(Person/Bike/Car)` | `mAP=66.8%` | `未公开` | 引用表未给出 | 为改进版结构 |

## 表 3：可直接确认的 YOLO 家族公开参数量参考

这个表不是 `BDD100K` 成绩表，而是为了给“参数量对比”提供可追溯参考。对于论文中未披露参数的变体，可以用这个表理解其基线级别，但不能把这里的参数直接当成论文改进版的真实参数。

| 模型 | 官方参数量 | 来源说明 |
|---|---:|---|
| `YOLO11n` | `2.6M` | Ultralytics 官方 `YOLO11` 文档 |
| `YOLOv8n` | `3.2M` | Ultralytics 官方 `YOLOv8` 文档 |
| `YOLOv8s` | `11.2M` | Ultralytics 官方 `YOLOv8` 文档 |
| `YOLOX-Darknet53` | `63.7M` | YOLOX 官方 Model Zoo |
| `YOLOv5s` | `7.2M` | YOLOv5 官方 PyTorch Hub/官方文档常见公开值 |

## 对 Train4 的位置判断

如果只看更接近完整 `BDD100K detection` 口径的工作，你的 `train4` 可以这样定位：

- 作为一个仅 `2.58M` 参数量级的轻量模型，`AP=25.3` 是合理且不差的结果。
- 相比公开的 `YOLOX` 与 `SES-YOLOv8n` 结果，`train4` 已经占优。
- 相比 `DGSS-YOLOv8s`，你的 `AP` 略高，但 `AP50` 偏低，说明高 IoU 区间未必吃亏，主要问题更像是召回与粗粒度检出能力不足。
- 与 `DINO`、`CO-DETR`、`DiffusionDet`、`DDQ`、`LSFM`、`DDRN` 这类更强检测器相比，仍有明显差距。
- 从“参数量 / 精度”角度看，`train4` 的性价比并不差，但若追求公开 SOTA 级别精度，仅靠 `YOLO11n` 很难达到。

## 对参数对比的结论

- 在当前可核对的工作里，`train4` 的参数量是最小一档，约 `2.58M`。
- `YOLOX-Darknet53` 这类较早检测器公开参数量约 `63.7M`，远高于 `train4`。
- `YOLOv8n`、`YOLO11n` 这类 nano 级模型通常在 `2.6M~3.2M`。
- 很多 BDD100K 改进论文只公布精度，不公布参数量，因此不能严谨地说它们“比你的模型更大或更小”，只能说它们在当前公开材料里“参数未公开”。

## 参考来源

1. 你的 `train4` FP32 留档: `exports/reports/train4_fp32_val_report_2026-03-18.md`
2. YOLOv3/YOLOv4 on BDD100K: [High-Profile VRU Detection on Resource-Constrained Hardware Using YOLOv3/v4 on BDD100K](https://pmc.ncbi.nlm.nih.gov/articles/PMC8321163/)
3. BDD100K 多模型比较表: [Scientific Reports 2025 Table 1](https://www.nature.com/articles/s41598-025-28305-x/tables/1)
4. BDD100K 三类改进检测比较表: [Scientific Reports 2023 Table 1](https://www.nature.com/articles/s41598-023-43458-3/tables/1)
5. YOLOP 官方说明: [PyTorch Hub - YOLOP](https://pytorch.org/hub/hustvl_yolop/)
6. YOLOP 仓库说明镜像: [Hugging Face README - YOLOP](https://huggingface.co/Riser/YOLOP/resolve/1780fc74a775c7061fbcae825ff165c5f8b48c53/README.md)
7. YOLO11 官方参数表: [Ultralytics YOLO11 Docs](https://docs.ultralytics.com/models/yolo11/)
8. YOLOv8 官方参数表: [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/models/yolov8/)
9. YOLOX 官方参数表: [YOLOX Model Zoo](https://yolox.readthedocs.io/en/latest/model_zoo.html)
10. YOLOv5 官方文档: [Ultralytics YOLOv5 Docs](https://docs.ultralytics.com/models/yolov5)
