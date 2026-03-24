# 新电脑 BDD100K 数据准备与训练步骤

在新电脑上从零完成：下载数据集 → 转换标注 → 训练。以下步骤在项目根目录 `yolo_v11` 下执行。

---

## 1. 环境准备

- 安装 **Miniconda**（或 Anaconda）。
- 创建并激活环境，安装依赖：

```powershell
cd d:\yolo\yolo_v11
conda create -n yolo_v11 python=3.10 -y
conda activate yolo_v11
pip install ultralytics
pip install modelscope datasets oss2 addict
```

- 若使用 GPU 训练：安装 CUDA 对应版本的 PyTorch（见 [pytorch.org](https://pytorch.org)）。

---

## 2. 下载 BDD100K（ModelScope）

使用 `tools/download_bdd100k_modelscope.py` 从 ModelScope 拉取数据，会写入 `datasets/BDD100K/raw` 缓存，并尝试整理到 `images/` 与 `labels_raw/`。

```powershell
conda activate yolo_v11
python tools/download_bdd100k_modelscope.py
```

- 数据会落在 `datasets/BDD100K/raw` 下（含 ModelScope 解压后的目录，哈希名可能因版本不同）。
- 若脚本把图片/标注拷贝到了 `datasets/BDD100K/images` 和 `datasets/BDD100K/labels_raw`，可跳过下面「整理图片」中已存在的部分。

---

## 3. 整理图片目录（若需要）

YOLO 训练需要图片在：

- `datasets/BDD100K/images/train/`
- `datasets/BDD100K/images/val/`

若下载后图片只在 `raw` 下（例如在 `raw/.../extracted/.../train/images` 和 `val/images`），需拷贝到上述路径。在资源管理器中找到 `raw` 下解压出的 `train/images` 和 `val/images`，拷贝到：

- `datasets/BDD100K/images/train/`
- `datasets/BDD100K/images/val/`

或用 PowerShell（把下面的 `...extracted路径...` 换成实际路径）：

```powershell
robocopy "datasets\BDD100K\raw\...extracted路径...\train\images" "datasets\BDD100K\images\train" *.jpg /S
robocopy "datasets\BDD100K\raw\...extracted路径...\val\images"   "datasets\BDD100K\images\val"   *.jpg /S
```

目标：train 约 70,000 张、val 约 10,000 张。

---

## 4. 转换标注为 YOLO 格式

转换脚本会在 `datasets/BDD100K/raw` 下自动查找  
`bdd100k_labels_images_train.json` 和 `bdd100k_labels_images_val.json`，无需改新电脑上的哈希路径。

```powershell
conda activate yolo_v11
python tools/bdd100k_to_yolo.py
```

- 输出目录默认：`datasets/BDD100K/labels/`（生成 `train/*.txt`、`val/*.txt`、`classes.txt`）。
- 若 raw 不在默认位置，可指定：

```powershell
python tools/bdd100k_to_yolo.py --raw-root "D:\path\to\BDD100K\raw"
```

- 转换完成后建议确认：  
  `labels/train` 约 69,863 个 txt，`labels/val` 10,000 个 txt。

---

## 5. 训练

数据集配置已放在 `ultralytics/cfg/datasets/bdd100k.yaml`。直接训练：

```powershell
conda activate yolo_v11
yolo task=detect mode=train model=weights/yolo11n.pt data=ultralytics/cfg/datasets/bdd100k.yaml batch=16 epochs=100 imgsz=640 workers=0 device=0
```

- 无 GPU 时改为 `device=cpu`（会较慢）。
- 显存不足可减小 `batch=8` 或 `batch=4`。
- 权重会保存在 `runs/detect/trainN/weights/best.pt`。

---

## 步骤一览

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 创建 conda 环境并安装依赖 | ultralytics、modelscope、datasets、oss2、addict |
| 2 | `python tools/download_bdd100k_modelscope.py` | 下载到 `datasets/BDD100K/raw` |
| 3 | 将 train/val 图片拷到 `datasets/BDD100K/images/train` 与 `val` | 若下载脚本已拷贝可跳过 |
| 4 | `python tools/bdd100k_to_yolo.py` | 在 raw 下自动找 JSON，输出到 `labels/` |
| 5 | `yolo task=detect mode=train model=weights/yolo11n.pt data=ultralytics/cfg/datasets/bdd100k.yaml ...` | 训练 |

---

## 常见问题

- **找不到 JSON**：确认 `datasets/BDD100K/raw` 下存在 ModelScope 解压后的目录，且内含 `train/annotations/bdd100k_labels_images_train.json` 与 `val/annotations/bdd100k_labels_images_val.json`。或用 `--raw-root` 指定实际 raw 根目录。
- **缺少预训练权重**：将 `weights/yolo11n.pt` 拷到新电脑同一路径，或从 Ultralytics 官方获取后放入 `weights/`。
- **训练进度**：在终端中直接运行上述 `yolo ... train` 命令即可在终端看到进度；或监控 `runs/detect/trainN/results.csv`。
