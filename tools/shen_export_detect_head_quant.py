"""
为检测头 (model.23) 的 Conv 层单独计算 INT8 量化参数。

backbone/neck 的量化参数已在 INT8 QDQ ONNX 中，
检测头因 ONNX Runtime 的输出 Concat 问题被排除。
但在 Gemmini 裸机部署中每层独立计算，不存在此问题。
本脚本通过 FP32 模型推理收集检测头各 Conv 输入/输出的激活统计，
生成 per-layer 对称量化参数。

用法:
    python tools/shen_export_detect_head_quant.py
"""

import os
import json
import glob
import random
import numpy as np
import cv2
import onnx
from onnx import numpy_helper
import onnxruntime as ort

FP32_ONNX = "exports/onnx/train4_best_fp32.onnx"
CALIB_DIR = "datasets/BDD100K/images/train"
NUM_CALIB = 200
IMG_SIZE = 640
SEED = 42
OUTPUT_JSON = "exports/quant/train4_detect_head_quant.json"


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img


def preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = letterbox(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inp = img.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))[np.newaxis]
    return inp


def main():
    model = onnx.load(FP32_ONNX)
    init_map = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}

    det_convs = [n for n in model.graph.node if "model.23" in n.name and n.op_type == "Conv"]
    print(f"[信息] 检测头 Conv 层数: {len(det_convs)}")

    tensor_names = set()
    for n in det_convs:
        tensor_names.add(n.input[0])
        tensor_names.add(n.output[0])

    extra_outputs = []
    for t in tensor_names:
        if t not in init_map:
            extra_outputs.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))

    modified = onnx.ModelProto()
    modified.CopyFrom(model)
    for eo in extra_outputs:
        modified.graph.output.append(eo)

    temp_path = "exports/onnx/_temp_calib.onnx"
    onnx.save(modified, temp_path)

    sess = ort.InferenceSession(temp_path, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    input_name = sess.get_inputs()[0].name

    all_imgs = sorted(glob.glob(os.path.join(CALIB_DIR, "*.jpg")))
    random.seed(SEED)
    sample = random.sample(all_imgs, min(NUM_CALIB, len(all_imgs)))
    print(f"[校准] 使用 {len(sample)} 张校准图片")

    stats = {name: {"absmax": 0.0} for name in out_names}

    for idx, img_path in enumerate(sample):
        inp = preprocess(img_path)
        if inp is None:
            continue
        results = sess.run(out_names, {input_name: inp})
        for name, val in zip(out_names, results):
            amax = float(np.abs(val).max())
            if amax > stats[name]["absmax"]:
                stats[name]["absmax"] = amax
        if (idx + 1) % 50 == 0:
            print(f"[校准] {idx + 1}/{len(sample)}")

    print(f"[校准] 完成")

    layers = []
    for n in det_convs:
        w_name = n.input[1]
        w = init_map.get(w_name)
        b_name = n.input[2] if len(n.input) > 2 and n.input[2] else None
        b = init_map.get(b_name) if b_name else None

        ks = [0, 0]
        stride = [1, 1]
        group = 1
        pads = [0, 0, 0, 0]
        for attr in n.attribute:
            if attr.name == "kernel_shape":
                ks = list(attr.ints)
            elif attr.name == "strides":
                stride = list(attr.ints)
            elif attr.name == "group":
                group = int(attr.i)
            elif attr.name == "pads":
                pads = list(attr.ints)

        inp_tensor = n.input[0]
        out_tensor = n.output[0]
        inp_absmax = stats.get(inp_tensor, {}).get("absmax", 0.0)
        out_absmax = stats.get(out_tensor, {}).get("absmax", 0.0)

        inp_scale = inp_absmax / 127.0 if inp_absmax > 0 else 1.0
        out_scale = out_absmax / 127.0 if out_absmax > 0 else 1.0

        w_per_channel_scales = []
        w_shape = list(w.shape)
        for oc in range(w_shape[0]):
            ch_absmax = float(np.abs(w[oc]).max())
            ch_scale = ch_absmax / 127.0 if ch_absmax > 0 else 1.0
            w_per_channel_scales.append(ch_scale)

        w_int8 = np.zeros_like(w, dtype=np.int8)
        for oc in range(w_shape[0]):
            w_int8[oc] = np.clip(np.round(w[oc] / w_per_channel_scales[oc]), -128, 127).astype(np.int8)

        requant_scales = []
        for ws in w_per_channel_scales:
            rq = (inp_scale * ws) / out_scale if out_scale > 0 else 0.0
            requant_scales.append(rq)

        b_int32 = None
        b_scale = None
        if b is not None:
            b_scale_val = inp_scale * np.mean(w_per_channel_scales)
            b_int32_arr = np.clip(np.round(b / b_scale_val), -(2**31), 2**31 - 1).astype(np.int32)
            b_int32 = b_int32_arr.tolist()
            b_scale = float(b_scale_val)

        layer = {
            "node_name": n.name,
            "weight_shape": w_shape,
            "kernel_shape": ks,
            "strides": stride,
            "pads": pads,
            "group": group,
            "has_bias": b is not None,
            "input_activation": {
                "absmax": float(inp_absmax),
                "scale": float(inp_scale),
                "zero_point": 0,
            },
            "output_activation": {
                "absmax": float(out_absmax),
                "scale": float(out_scale),
                "zero_point": 0,
            },
            "weight": {
                "per_channel": True,
                "scales": w_per_channel_scales,
                "zero_points": [0] * w_shape[0],
            },
            "requant_scales": requant_scales,
        }
        if b_scale is not None:
            layer["bias"] = {
                "scale": b_scale,
                "int32_values": b_int32,
            }

        layers.append(layer)

    branch_map = {}
    for l in layers:
        nm = l["node_name"]
        if "cv2" in nm:
            branch = "cv2_bbox"
        elif "cv3" in nm:
            branch = "cv3_cls"
        elif "dfl" in nm:
            branch = "dfl"
        else:
            branch = "other"
        l["branch"] = branch
        branch_map.setdefault(branch, []).append(nm)

    output = {
        "description": "检测头 (model.23) 各 Conv 层的独立 INT8 量化参数",
        "note": "这些参数用于 Gemmini 裸机部署，每层独立量化，不存在 ONNX Runtime Concat 混合 scale 问题",
        "quantization": {
            "method": "MinMax (per-layer activation, per-channel weight)",
            "symmetric": True,
            "calibration_images": NUM_CALIB,
        },
        "branches": {
            "cv2_bbox": {
                "description": "bbox 回归分支 (3 个 scale × 3 Conv)",
                "layers": branch_map.get("cv2_bbox", []),
            },
            "cv3_cls": {
                "description": "分类分支 (3 个 scale × 5 Conv, 含 depthwise)",
                "layers": branch_map.get("cv3_cls", []),
            },
            "dfl": {
                "description": "DFL 解码 Conv (1×1, 固定权重)",
                "layers": branch_map.get("dfl", []),
            },
        },
        "num_layers": len(layers),
        "layers": layers,
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[导出] 检测头量化参数: {OUTPUT_JSON}")
    print(f"  总 Conv 层: {len(layers)}")
    print(f"  cv2 (bbox): {len(branch_map.get('cv2_bbox', []))} 层")
    print(f"  cv3 (cls):  {len(branch_map.get('cv3_cls', []))} 层")
    print(f"  DFL:        {len(branch_map.get('dfl', []))} 层")

    print(f"\n[摘要] 各层激活 scale:")
    for l in layers:
        nm = l["node_name"].replace("/model.23/", "")
        inp_s = l["input_activation"]["scale"]
        out_s = l["output_activation"]["scale"]
        ws = l["weight"]["scales"]
        print(f"  {nm:45s}  inp_scale={inp_s:.6f}  out_scale={out_s:.6f}  "
              f"w_scale=[{min(ws):.6f}, {max(ws):.6f}]")

    os.remove(temp_path)
    print(f"\n[完成]")


if __name__ == "__main__":
    main()
