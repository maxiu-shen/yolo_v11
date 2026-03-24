"""
分析检测头 (model.23) 的详细结构，用于 Gemmini 部署规划。
"""
import os
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import cv2
import glob
import random

FP32_ONNX = "exports/onnx/train4_best_fp32.onnx"
INT8_ONNX = "exports/onnx/train4_best_int8.onnx"
CALIB_DIR = "datasets/BDD100K/images/val"
IMG_SIZE = 640

model = onnx.load(FP32_ONNX)

print("=" * 70)
print("检测头 Conv 层详细结构")
print("=" * 70)

det_nodes = [n for n in model.graph.node if "model.23" in n.name and n.op_type == "Conv"]
init_map = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}

cv2_layers = []
cv3_layers = []
dfl_layers = []
other_layers = []

for n in det_nodes:
    w_name = n.input[1]
    w = init_map.get(w_name)
    w_shape = list(w.shape) if w is not None else []
    has_bias = len(n.input) > 2 and n.input[2] and n.input[2] in init_map

    ks = [0, 0]
    stride = [1, 1]
    group = 1
    for attr in n.attribute:
        if attr.name == "kernel_shape":
            ks = list(attr.ints)
        elif attr.name == "strides":
            stride = list(attr.ints)
        elif attr.name == "group":
            group = int(attr.i)

    info = {
        "name": n.name,
        "weight_shape": w_shape,
        "kernel": ks,
        "stride": stride,
        "group": group,
        "has_bias": has_bias,
        "w_absmax": float(np.abs(w).max()) if w is not None else 0,
        "w_mean_abs": float(np.abs(w).mean()) if w is not None else 0,
    }

    if "cv2" in n.name:
        cv2_layers.append(info)
    elif "cv3" in n.name:
        cv3_layers.append(info)
    elif "dfl" in n.name:
        dfl_layers.append(info)
    else:
        other_layers.append(info)

print(f"\ncv2 (bbox 回归) Conv: {len(cv2_layers)} 层")
for l in cv2_layers:
    tag = "DW" if l["group"] > 1 else f"{l['kernel'][0]}x{l['kernel'][1]}"
    print(f"  {l['name']}")
    print(f"    shape={l['weight_shape']}, {tag}, stride={l['stride']}, "
          f"bias={l['has_bias']}, w_absmax={l['w_absmax']:.4f}, w_mean={l['w_mean_abs']:.4f}")

print(f"\ncv3 (分类) Conv: {len(cv3_layers)} 层")
for l in cv3_layers:
    tag = "DW" if l["group"] > 1 else f"{l['kernel'][0]}x{l['kernel'][1]}"
    print(f"  {l['name']}")
    print(f"    shape={l['weight_shape']}, {tag}, stride={l['stride']}, "
          f"bias={l['has_bias']}, w_absmax={l['w_absmax']:.4f}, w_mean={l['w_mean_abs']:.4f}")

print(f"\nDFL Conv: {len(dfl_layers)} 层")
for l in dfl_layers:
    print(f"  {l['name']}")
    print(f"    shape={l['weight_shape']}, bias={l['has_bias']}, "
          f"w_absmax={l['w_absmax']:.4f}")

total_det_params = sum(np.prod(l["weight_shape"]) for l in cv2_layers + cv3_layers + dfl_layers)
total_model_params = sum(np.prod(list(init_map[k].shape)) for k in init_map if "weight" in k and init_map[k].ndim >= 2)
print(f"\n检测头权重参数量: {total_det_params:,} ({total_det_params/total_model_params*100:.1f}% of total)")

print("\n" + "=" * 70)
print("检测头各分支激活值范围分析 (10 张 val 图)")
print("=" * 70)

det_conv_outputs = set()
for n in det_nodes:
    det_conv_outputs.add(n.output[0])

all_intermediate_names = set()
for n in model.graph.node:
    if "model.23" in n.name:
        for out in n.output:
            all_intermediate_names.add(out)

key_tensors = []
for n in det_nodes:
    key_tensors.append(n.output[0])
split_node = [n for n in model.graph.node if n.name == "/model.23/Split"]
if split_node:
    for out in split_node[0].output:
        key_tensors.append(out)
sigmoid_node = [n for n in model.graph.node if n.name == "/model.23/Sigmoid"]
if sigmoid_node:
    key_tensors.append(sigmoid_node[0].output[0])

extra_outputs = []
for t in key_tensors[:30]:
    extra_outputs.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))

modified_model = onnx.ModelProto()
modified_model.CopyFrom(model)
for eo in extra_outputs:
    modified_model.graph.output.append(eo)

temp_path = "exports/onnx/_temp_debug.onnx"
onnx.save(modified_model, temp_path)

sess = ort.InferenceSession(temp_path, providers=["CPUExecutionProvider"])
output_names = [o.name for o in sess.get_outputs()]

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

imgs = sorted(glob.glob(os.path.join(CALIB_DIR, "*.jpg")))
random.seed(42)
sample = random.sample(imgs, min(10, len(imgs)))

stats = {name: {"min": float("inf"), "max": float("-inf"), "absmax": 0} for name in output_names}

for img_path in sample:
    img = cv2.imread(img_path)
    img = letterbox(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inp = img.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))[np.newaxis]

    results = sess.run(output_names, {sess.get_inputs()[0].name: inp})
    for name, val in zip(output_names, results):
        stats[name]["min"] = min(stats[name]["min"], float(val.min()))
        stats[name]["max"] = max(stats[name]["max"], float(val.max()))
        stats[name]["absmax"] = max(stats[name]["absmax"], float(np.abs(val).max()))

node_name_map = {}
for n in model.graph.node:
    for out in n.output:
        node_name_map[out] = n.name

print("\n各 Conv 输出激活范围:")
for name in key_tensors:
    if name in stats and name in node_name_map:
        s = stats[name]
        nname = node_name_map[name]
        scale_sym = s["absmax"] / 127.0
        print(f"  {nname}")
        print(f"    range: [{s['min']:.4f}, {s['max']:.4f}], absmax={s['absmax']:.4f}, "
              f"INT8_scale={scale_sym:.6f}")

os.remove(temp_path)

print("\n" + "=" * 70)
print("Gemmini 部署方案分析")
print("=" * 70)

print("""
检测头每个 Conv 可以独立量化:
  - cv2 (bbox): 输出值范围较大，INT8 per-tensor 可行
  - cv3 (cls):  输出值范围小但非零，INT8 per-tensor 可行（单独量化不受 bbox 干扰）
  - DFL:        1x1 Conv，值域确定

后处理 (非 Conv) 必须在 CPU 执行:
  - Concat (合并 3 个 scale 的输出)
  - Split (分离 bbox/cls)
  - Sigmoid (分类置信度)
  - Softmax + DFL Conv (box 分布解码)
  - Box 解码 (anchor-free)
  - NMS
""")

del model
