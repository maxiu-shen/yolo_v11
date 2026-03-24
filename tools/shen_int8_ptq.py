"""
YOLOv11n INT8 Post-Training Quantization (PTQ) v2

使用 ONNX Runtime 静态量化，基于 BDD100K 训练集子集进行校准。
输出 QDQ 格式的 INT8 ONNX 模型，便于后续提取每层量化参数用于 Gemmini 部署。

v2 改动:
  - per_channel=True (按通道量化权重)
  - 排除检测头 model.23 的后处理节点 (Concat/Split/Sigmoid/DFL 等) 避免
    分类通道因 scale 被 bbox 主导而全部量化为零
  - 校准样本增加到 500 张

用法:
    conda activate yolo_v11
    python tools/shen_int8_ptq.py
"""

import os
import sys
import glob
import time
import random
import numpy as np
import cv2
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
    CalibrationMethod,
)


FP32_ONNX = "exports/onnx/train4_best_fp32.onnx"
INT8_ONNX = "exports/onnx/train4_best_int8.onnx"
CALIB_DIR = "datasets/BDD100K/images/train"
NUM_CALIB = 500
IMG_SIZE = 640
SEED = 42


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """YOLOv11 标准 letterbox：保持长宽比缩放 + 填充。"""
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img


def preprocess(img_path):
    """标准 YOLOv11 预处理：BGR -> RGB, letterbox, /255, NCHW float32。"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = letterbox(img, (IMG_SIZE, IMG_SIZE))
    img = img[:, :, ::-1]  # BGR -> RGB
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)   # -> NCHW
    return img.astype(np.float32)


class BDD100KCalibReader(CalibrationDataReader):
    """BDD100K 校准数据读取器。"""

    def __init__(self, calib_dir, num_images, input_name="images"):
        exts = ("*.jpg", "*.jpeg", "*.png")
        all_images = []
        for ext in exts:
            all_images.extend(glob.glob(os.path.join(calib_dir, ext)))
        if len(all_images) == 0:
            raise FileNotFoundError(f"No images found in {calib_dir}")

        random.seed(SEED)
        if len(all_images) > num_images:
            all_images = random.sample(all_images, num_images)

        self.image_paths = sorted(all_images)
        self.input_name = input_name
        self.idx = 0
        print(f"[校准] 共选取 {len(self.image_paths)} 张校准图片")

    def get_next(self):
        if self.idx >= len(self.image_paths):
            return None
        img = preprocess(self.image_paths[self.idx])
        self.idx += 1
        if self.idx % 50 == 0:
            print(f"[校准] 已处理 {self.idx}/{len(self.image_paths)}")
        if img is None:
            return self.get_next()
        return {self.input_name: img}

    def rewind(self):
        self.idx = 0


def get_detect_head_nodes(model_path):
    """获取检测头 (model.23) 中需要排除量化的节点名。
    
    排除所有 /model.23/ 下的节点，因为：
    - 分类头 Conv (cv3) 输出值极小，INT8 per-tensor 量化会全部归零
    - 后处理节点 (Concat/Split/DFL/Sigmoid) 混合不同 scale 的数据
    - 检测头计算量占比小，保持 FP32 对整体性能影响很小
    """
    model = onnx.load(model_path)
    exclude = []
    for node in model.graph.node:
        if "/model.23/" in node.name:
            exclude.append(node.name)
    del model
    return exclude


def main():
    if not os.path.exists(FP32_ONNX):
        print(f"[错误] 未找到 FP32 ONNX: {FP32_ONNX}")
        sys.exit(1)

    os.makedirs(os.path.dirname(INT8_ONNX), exist_ok=True)

    model = onnx.load(FP32_ONNX)
    input_name = model.graph.input[0].name
    print(f"[信息] FP32 模型输入名称: {input_name}")
    print(f"[信息] FP32 模型节点数: {len(model.graph.node)}")
    del model

    exclude_nodes = get_detect_head_nodes(FP32_ONNX)
    print(f"[信息] 排除量化的检测头节点数: {len(exclude_nodes)}")
    for n in exclude_nodes:
        print(f"  - {n}")

    calib_reader = BDD100KCalibReader(CALIB_DIR, NUM_CALIB, input_name)

    print(f"\n[量化] 开始 INT8 PTQ 量化 (v2)...")
    print(f"  - 量化格式: QDQ (QuantizeLinear/DequantizeLinear)")
    print(f"  - 权重类型: INT8 (per-channel, symmetric)")
    print(f"  - 激活类型: INT8 (symmetric)")
    print(f"  - 校准方法: MinMax")
    print(f"  - 校准图片数: {NUM_CALIB}")
    print(f"  - 检测头 (model.23): FP32 (不量化)")

    t0 = time.time()
    quantize_static(
        model_input=FP32_ONNX,
        model_output=INT8_ONNX,
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
        nodes_to_exclude=exclude_nodes,
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
        },
    )
    elapsed = time.time() - t0
    print(f"[量化] 完成，耗时 {elapsed:.1f}s")

    if os.path.exists(INT8_ONNX):
        size_fp32 = os.path.getsize(FP32_ONNX) / 1024 / 1024
        size_int8 = os.path.getsize(INT8_ONNX) / 1024 / 1024
        print(f"\n[结果]")
        print(f"  FP32 ONNX: {FP32_ONNX} ({size_fp32:.1f} MB)")
        print(f"  INT8 ONNX: {INT8_ONNX} ({size_int8:.1f} MB)")
        print(f"  压缩比: {size_fp32/size_int8:.2f}x")

        int8_model = onnx.load(INT8_ONNX)
        node_types = {}
        for node in int8_model.graph.node:
            node_types[node.op_type] = node_types.get(node.op_type, 0) + 1
        print(f"\n[INT8 模型算子统计]")
        for op, cnt in sorted(node_types.items()):
            print(f"  {op}: {cnt}")

        qdq_count = node_types.get("QuantizeLinear", 0)
        dqdq_count = node_types.get("DequantizeLinear", 0)
        print(f"\n  QuantizeLinear 节点数: {qdq_count}")
        print(f"  DequantizeLinear 节点数: {dqdq_count}")
    else:
        print(f"[错误] INT8 ONNX 未生成")
        sys.exit(1)

    print(f"\n[完成] INT8 PTQ 量化成功，输出: {INT8_ONNX}")


if __name__ == "__main__":
    main()
