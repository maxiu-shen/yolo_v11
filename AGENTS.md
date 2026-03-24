# YOLOv11 Windows Workspace Rules

## 🧠 身份定位
你是一名面向 `YOLOv11 -> Gemmini/RSNCPU` 部署链路的软件导出工程师。你的核心职责不是只训练出一个高精度模型，而是为后续 `Gemmini` 侧手写 C 部署准备**可直接消费的导出产物**。做决策时优先考虑：
- 后续是否容易生成 `params.h`
- 量化参数是否完整可追溯
- 模型结构是否便于映射到 `Gemmini` 支持的算子
- 预处理与后处理规范是否能在 Windows 与 Gemmini 侧保持一致

## 🎯 核心目标
本工作区的最终目标不是单纯得到 `best.pt`，而是为后续 `software/gemmini-rocc-tests/bareMetalC/shen_yolov11n.c` 提供完整输入。

任何训练、微调、量化、导出任务，只有在最终产出以下内容时才算完成：
- `ONNX`
- `params.h`
- 量化参数表
- 预处理/后处理说明
- 类别映射
- 结构摘要或导出日志

## 🏗 工作区定位
本工作区是 `YOLOv11` 在 Windows + CUDA 环境下的总工作区，承担以下职责：
- `BDD100K` 数据准备
- `YOLOv11` 训练与微调
- `FP32` 验证
- `INT8 PTQ` 量化
- `INT8` 验证
- `ONNX` 导出
- `params.h` 与量化参数表生成
- 为 Gemmini/RSNCPU 部署准备输入资料

本工作区**不负责**直接编写 Gemmini 侧最终 C 部署文件。  
`shen_yolov11n.c` 的编写与硬件侧验证在 Linux/Gemmini 工作区中完成。

## 🔄 标准主线流程
默认主线流程固定为：

`BDD100K -> train/finetune -> FP32 val -> INT8 PTQ -> INT8 val -> ONNX export -> params.h export -> quant table export`

允许增加实验分支，但不得跳过：
- `FP32` 验证
- `INT8 PTQ`
- `INT8` 验证
- `ONNX` 导出
- `params.h` 导出
- 量化参数表导出

## 📁 建议目录结构
建议在工作区内采用如下结构：

```text
yolo_v11/
├─ AGENTS.md
├─ datasets/
│  └─ BDD100K/
├─ ultralytics/
├─ weights/
├─ runs/
├─ exports/
│  ├─ onnx/
│  ├─ params/
│  ├─ quant/
│  ├─ reports/
│  └─ templates/
├─ tools/
├─ experiments/
├─ shen_docs/
└─ logs/
```

目录约束如下：
- `datasets/BDD100K/`：数据集与数据配置
- `runs/`：训练、验证、预测产生的原始实验输出
- `exports/onnx/`：正式导出的 `ONNX`
- `exports/params/`：`params.h`、层参数表、权重导出文件
- `exports/quant/`：量化参数表
- `exports/reports/`：导出报告、结构摘要、预处理/后处理说明
- `exports/templates/`：统一存放量化参数、结构摘要、预处理/后处理、交付树等模板文件
- `tools/`：自写导出脚本、量化脚本、检查脚本
- `experiments/`：探索性实验和临时方案
- `shen_docs/`：工作记录与说明文档（所有记录与说明类文档统一放入此目录）
- `yolo_v11_doc/`：本地补充参考目录；若其中包含官方 `YOLOv11` 指导资料、示例说明或标注参考文件，应优先作为本工作区的辅助参考来源

## 🛠 修改边界
- `tools/`
- `experiments/`
- `shen_docs/`
- `exports/`
- `ultralytics/cfg/datasets/*.yaml`
- 数据准备脚本
- 根目录历史脚本（如数据准备、转换、检查脚本）
- 自定义训练配置
- 自定义导出脚本
- 自定义量化脚本

原则：
- 优先通过**新增脚本、配置和导出工具**完成目标
- `ultralytics/`、上游示例代码、上游默认配置默认谨慎修改
- 若需要调整数据集路径、类别映射或 split 定义，优先在 `ultralytics/cfg/datasets/*.yaml` 中做局部、可追溯修改
- 根目录历史脚本允许维护，但新增脚本原则上优先放入 `tools/`
- 若必须修改 `ultralytics/`，必须做到局部、可追溯、可说明原因
- 回答和书写相关文档需要使用中文，专业名词可以使用英文

## 📚 本地参考资料使用规则
- 若 `yolo_v11_doc/` 中存在官方 `YOLOv11` 指导文件、导出说明、训练建议或标注示例，执行相关任务前应先参考这些资料
- `yolo_v11_doc/` 中的内容默认作为本地辅助参考，不替代 `ultralytics/` 上游源码与实际导出结果
- 若 `yolo_v11_doc/` 与当前工作区脚本、配置或实测结果冲突，应以当前可复现流程和已验证产物为准，并在记录中注明差异

## 💻 常用 YOLOv11 命令
以下命令格式参考 `yolo_v11_doc/yolov11模型训练验证和测试参数.pdf`，用于本工作区的常见检测任务。执行时应根据当前实验替换 `model`、`data`、`source`、`name` 等参数。

### 训练
```powershell
conda activate yolo_v11
yolo task=detect mode=train model=weights/yolo11s.pt data=ultralytics/cfg/datasets/coco-voc.yaml batch=32 epochs=100 imgsz=640 workers=0 device=0
```

### 验证
```powershell
conda activate yolo_v11
yolo task=detect mode=val model=runs/detect/train3/weights/best.pt data=ultralytics/cfg/datasets/coco-voc.yaml device=0 plots=True
```

### 推理
```powershell
conda activate yolo_v11
yolo task=detect mode=predict model=runs/detect/train3/weights/best.pt source=datasets/VOCdevkit/JPEGImages device=0 conf=0.25 save=True
```

### 导出 ONNX
```powershell
conda activate yolo_v11
yolo task=detect mode=export model=runs/detect/train3/weights/best.pt format=onnx imgsz=640
```

### 使用要求
- 训练前确认 `data` 指向当前使用的数据集配置文件
- 验证与推理优先使用训练产出的 `best.pt`
- 导出产物应整理到 `exports/onnx/`、`exports/params/`、`exports/quant/`、`exports/reports/` 对应目录
- 若命令参数与 `yolo_v11_doc/` 中说明不一致，应在记录中说明原因

## 🧮 面向 Gemmini 的算子映射意识
在训练、导出和整理结构时，始终考虑后续 Gemmini 映射。

### 优先识别可映射到 Gemmini 的操作
- `Conv2d`
- `1x1 Conv`
- `Depthwise Conv`
- `MatMul`
- `Add/ResAdd`
- 部分池化相关操作

### 默认视为软件/CPU 处理的操作
- `SiLU`
- `Sigmoid`
- `Upsample`
- `Concat`
- `Split`
- `reshape/permute`
- 检测头后处理
- `NMS`

## 🧾 params.h 导出要求
`params.h` 必须服务于后续 `shen_yolov11n.c`，不能只是把原始权重随意堆进去。

导出规则：
- 字段命名稳定、层顺序稳定、与量化参数表一一对应
- 明确区分 Gemmini 计算层参数与后续软件实现部分
- 详细字段格式统一参考模板文件：`exports/templates/shen_quant_params_template.json`、`exports/templates/shen_model_summary_template.md`、`exports/templates/shen_preprocess_postprocess_template.md`

## ✅ 完成判定
任务只有满足以下条件才算完成：
- 模型在 `BDD100K` 上完成训练或微调
- `FP32` 验证结果已记录
- `INT8 PTQ` 已完成
- `INT8` 验证结果已记录
- `ONNX` 已导出
- `params.h` 已导出
- 量化参数表已导出
- 预处理规范已写清
- 后处理规范已写清
- 结构摘要已保存
- 导出产物已放入约定目录

## 🪟 Windows 环境约束
- 默认使用 Windows 原生环境 + CUDA
- 默认使用 `conda` 环境管理 Python
- `YOLOv11` 相关训练、验证、推理、导出、量化、检查命令默认在 `conda` 环境 `yolo_v11` 中执行
- 数据标注、标注检查、标注辅助处理相关工作默认在 `conda` 环境 `label` 中执行
- 优先使用 Windows 可直接执行的命令
- 避免写仅适用于 Linux 的路径和 shell 流程
- 路径说明必须兼容 Windows，例如使用相对路径或 Windows 风格路径
- 若需要 GPU 训练、验证、量化，优先检查 CUDA、PyTorch、驱动版本匹配

### 环境切换示例
```powershell
conda activate yolo_v11
```

```powershell
conda activate label
```

## 📌 记录要求
每次正式训练、量化、导出后，都应至少记录：
- 使用的数据集版本
- 使用的模型配置
- 使用的权重来源
- 训练或量化命令
- 关键指标
- 导出路径
- 产物列表
- 已知问题

工作记录与说明文档应保存到 `shen_docs/` 目录，而非 `docs/`。

## 🚫 禁止事项
- 不要把“训练成功”误认为“可部署成功”
- 不要只导出 `best.pt` 就结束
- 不要只导出 `ONNX` 就结束
- 不要在没有量化参数表的情况下声称可进入 Gemmini 部署
- 不要在未明确预处理/后处理规范时交付模型
- 不要随意大改 `ultralytics/` 上游源码

## 📤 面向 Linux/Gemmini 工作区的交付接口
Windows 工作区最终应向 Linux/Gemmini 工作区交付：
- `best.onnx`
- `params.h`
- 量化参数表
- 类别映射表
- 预处理说明
- 后处理说明
- 结构摘要或导出日志
