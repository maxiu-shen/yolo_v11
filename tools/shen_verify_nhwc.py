"""验证 NHWC 权重布局转换的正确性。"""
import json

with open("exports/params/train4_layer_params.json", "r", encoding="utf-8") as f:
    d = json.load(f)

print("=== 布局转换验证 ===\n")
print(f"weight_layout: {d['weight_layout']}")
print(f"总层数: {d['total_layers']}")
print()

print("前 10 层示例:")
for l in d["layers"][:10]:
    grp = l.get("group", "")
    oihw = l.get("weight_shape_oihw", "")
    nhwc = l.get("weight_shape_nhwc", "")
    layout = l.get("weight_layout", "")
    print(f"  {l['name']:10s} {l['op_type']:7s} "
          f"group={str(grp):3s} "
          f"OIHW={str(oihw):20s} -> {layout:8s} = {nhwc}")

print("\nDW Conv 层:")
for l in d["layers"]:
    if l.get("weight_layout") == "dw_chw":
        print(f"  {l['name']:10s} OIHW={l['weight_shape_oihw']} -> CHW={l['weight_shape_nhwc']}")

print("\nMatMul 层:")
for l in d["layers"]:
    if l["op_type"] == "MatMul":
        print(f"  {l['name']:10s} shape={l['weight_shape_nhwc']} layout={l['weight_layout']}")

print("\nFP32 检测头层 (前 5 个):")
fp32 = [l for l in d["layers"] if not l.get("quantized")]
for l in fp32[:5]:
    print(f"  {l['name']:10s} {l.get('weight_layout',''):8s} "
          f"OIHW={l.get('weight_shape_oihw','')} -> NHWC={l.get('weight_shape_nhwc','')}")

print("\n=== 数值验证 ===")
import numpy as np
import onnx
from onnx import numpy_helper

model = onnx.load("exports/onnx/train4_best_int8.onnx")
init_map = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
node_map = {o: n for n in model.graph.node for o in n.output}

conv0 = [n for n in model.graph.node if n.op_type == "Conv"][0]
wn = node_map.get(conv0.input[1])
raw_w = init_map[wn.input[0]]
print(f"\nconv_0 原始权重 (OIHW): shape={raw_w.shape}, dtype={raw_w.dtype}")
print(f"  raw[0,0,:,:] =\n{raw_w[0, 0, :, :]}")

converted = raw_w.transpose(2, 3, 1, 0).reshape(-1, raw_w.shape[0])
print(f"\nconv_0 转换后 (HWIO): shape={converted.shape}")
print(f"  前 9 行第 0 列 (应为 raw[0,0,:,:].flatten()):")
print(f"  {converted[:9, 0]}")
print(f"  原始 raw[0,0,:,:].flatten():")
print(f"  {raw_w[0, 0, :, :].flatten()}")
match = np.array_equal(converted[:9, 0], raw_w[0, 0, :, :].flatten())
print(f"  匹配: {match}")

conv_1x1 = None
for n in model.graph.node:
    if n.op_type != "Conv":
        continue
    for attr in n.attribute:
        if attr.name == "kernel_shape" and list(attr.ints) == [1, 1]:
            wn2 = node_map.get(n.input[1])
            if wn2 and wn2.op_type == "DequantizeLinear":
                w2 = init_map.get(wn2.input[0])
                if w2 is not None:
                    conv_1x1 = (n.name, w2)
                    break
    if conv_1x1:
        break

if conv_1x1:
    name, w = conv_1x1
    converted_1x1 = np.squeeze(w, axis=(2, 3)).T
    print(f"\n1x1 Conv ({name}):")
    print(f"  OIHW: {w.shape} -> IO: {converted_1x1.shape}")
    print(f"  w[0,:,0,0] (OC=0的所有IC权重): {w[0, :4, 0, 0]}")
    print(f"  converted_1x1[:4,0] (IC前4行,OC=0列): {converted_1x1[:4, 0]}")
    match2 = np.array_equal(w[0, :, 0, 0], converted_1x1[:, 0])
    print(f"  匹配: {match2}")
