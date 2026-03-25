# BDD100K 数据集下载与准备指南

本文档用于在新机器上从零重建 `datasets/BDD100K/` 目录，使其与本项目的训练、验证流程完全兼容。

---

## 1. 目标目录结构

最终需要得到如下结构（相对于项目根目录 `yolo_v11/`）：

```
datasets/BDD100K/
├── images/
│   ├── train/          # 70,000 张 JPG（1280×720）
│   └── val/            # 10,000 张 JPG（1280×720）
├── labels/
│   ├── train/          # 69,863 个 TXT（YOLO 格式）
│   └── val/            # 10,000 个 TXT（YOLO 格式）
└── raw/                # 下载的原始文件（可选保留，后续不参与训练）
    ├── train.zip
    ├── val.zip
    └── iic/.../bdd100k_labels_images_train.json
         .../bdd100k_labels_images_val.json
```

## 2. 下载源

BDD100K 数据集通过 **ModelScope** 下载。

- 数据集页面：<https://modelscope.cn/datasets/iic/BDD100K>
- namespace：`iic`（备选 `damo`）
- split 映射：train → `train`，val → `validation`

## 3. 环境准备

```bash
# 激活 conda 环境
conda activate yolo_v11

# 确保 modelscope 已安装
pip install modelscope
```

## 4. 步骤一：下载原始数据

项目中已提供下载脚本 `tools/download_bdd100k_modelscope.py`，在项目根目录执行：

```bash
python tools/download_bdd100k_modelscope.py
```

该脚本会：
- 将原始下载缓存存放到 `datasets/BDD100K/raw/`
- 将图片整理到 `datasets/BDD100K/images/train/` 和 `images/val/`
- 将原始标注（per-image JSON）存放到 `datasets/BDD100K/labels_raw/`

> **注意**：下载总量约 5.5 GB（train.zip ≈ 3.9 GB，val.zip ≈ 555 MB），请确保磁盘空间充足（建议预留 25 GB）。
> 下载速度取决于 ModelScope CDN，国内网络通常 10–50 MB/s。

### 如果脚本下载失败

也可以手动下载：
1. 访问 <https://modelscope.cn/datasets/iic/BDD100K/files>
2. 下载 `train.zip`、`val.zip` 和标注 JSON 文件
3. 将 zip 放入 `datasets/BDD100K/raw/`
4. 解压图片到对应目录：
   ```bash
   mkdir -p datasets/BDD100K/images/train datasets/BDD100K/images/val
   unzip datasets/BDD100K/raw/train.zip -d datasets/BDD100K/images/train/
   unzip datasets/BDD100K/raw/val.zip -d datasets/BDD100K/images/val/
   ```
   > 解压后确认图片直接位于 `train/` 和 `val/` 下（不要有多余嵌套子目录）。

## 5. 步骤二：转换标注为 YOLO 格式

项目中已提供转换脚本 `tools/bdd100k_to_yolo.py`，它读取 BDD100K 的聚合标注 JSON 文件，输出 YOLO 格式的 per-image `.txt` 标注。

```bash
python tools/bdd100k_to_yolo.py
```

默认参数：
- 输入：自动在 `datasets/BDD100K/raw/` 下递归搜索 `bdd100k_labels_images_train.json` 和 `bdd100k_labels_images_val.json`
- 输出：`datasets/BDD100K/labels/train/` 和 `datasets/BDD100K/labels/val/`

也可指定路径：
```bash
python tools/bdd100k_to_yolo.py --raw-root datasets/BDD100K/raw --output-root datasets/BDD100K/labels
```

### 标注格式说明

每个 `.txt` 文件一行一个目标，格式为：
```
<class_id> <x_center> <y_center> <width> <height>
```
所有坐标归一化到 `[0, 1]`，原始图像尺寸为 `1280×720`。

## 6. 类别定义（10 类）

| class_id | 类别名 |
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

对应配置文件：`ultralytics/cfg/datasets/bdd100k.yaml`

## 7. 步骤三：验证数据完整性

完成后执行以下检查：

```bash
# 检查图片数量
ls datasets/BDD100K/images/train/ | wc -l   # 期望：70000
ls datasets/BDD100K/images/val/ | wc -l     # 期望：10000

# 检查标注数量
ls datasets/BDD100K/labels/train/ | wc -l   # 期望：69863
ls datasets/BDD100K/labels/val/ | wc -l     # 期望：10000

# 抽查一个标注文件内容
head -3 datasets/BDD100K/labels/train/0000f77c-6257be58.txt
# 期望输出类似：
# 8 0.891750 0.238931 0.024278 0.107904
# 8 0.917378 0.241328 0.026976 0.103108
# 9 0.887704 0.308811 0.053952 0.031172
```

> **train 标注少于图片数**是正常的（69,863 vs 70,000），因为有 137 张图片没有可用的检测标注。

## 8. 步骤四：测试训练流程

```bash
conda activate yolo_v11
yolo task=detect mode=train model=weights/yolo11n.pt data=ultralytics/cfg/datasets/bdd100k.yaml epochs=1 imgsz=640 batch=16 workers=0 device=0
```

如果能正常跑 1 个 epoch，数据集准备就完成了。

## 9. 清理（可选）

`datasets/BDD100K/raw/` 目录在转换完成后不再参与训练，可按需删除以节省磁盘空间（约 15 GB）：

```bash
rm -rf datasets/BDD100K/raw/
```

---

## 快速命令总结

```bash
conda activate yolo_v11
pip install modelscope

# 下载
python tools/download_bdd100k_modelscope.py

# 转换标注
python tools/bdd100k_to_yolo.py

# 验证
ls datasets/BDD100K/images/train/ | wc -l
ls datasets/BDD100K/labels/train/ | wc -l

# 测试训练
yolo task=detect mode=train model=weights/yolo11n.pt data=ultralytics/cfg/datasets/bdd100k.yaml epochs=1 imgsz=640 batch=16 workers=0 device=0
```
