"""验证导出产物的正确性。"""
import json

with open("exports/quant/train4_quant_params.json", "r", encoding="utf-8") as f:
    d = json.load(f)

print("=== JSON 元数据 ===")
for k, v in d.items():
    if k != "layers":
        print(f"  {k}: {v}")

layers = d["layers"]
q_layers = [l for l in layers if l.get("quantized")]
fp32_layers = [l for l in layers if not l.get("quantized")]
print(f"\n量化层: {len(q_layers)}, FP32 层: {len(fp32_layers)}")

l0 = q_layers[0]
print(f"\n量化层示例: {l0['layer_name']}")
print(f"  quant_granularity: {l0.get('quant_granularity')}")
w_scale = l0.get("weight", {}).get("scale")
if isinstance(w_scale, list):
    print(f"  weight.scale: list[{len(w_scale)}] = [{w_scale[0]:.6f}, ..., {w_scale[-1]:.6f}]")
else:
    print(f"  weight.scale: {w_scale}")
print(f"  input.scale: {l0.get('input', {}).get('scale')}")
print(f"  output.scale: {l0.get('output', {}).get('scale')}")
rq = l0.get("output", {}).get("requant_scale")
if isinstance(rq, list):
    print(f"  requant_scale: list[{len(rq)}] = [{rq[0]:.6f}, ..., {rq[-1]:.6f}]")

f0 = fp32_layers[0]
print(f"\nFP32 层示例: {f0['layer_name']}")
print(f"  quantized: {f0.get('quantized')}")
print(f"  weight: {f0.get('weight')}")
print(f"  node_name: {f0.get('node_name')}")

print("\n=== layer_params.json ===")
with open("exports/params/train4_layer_params.json", "r", encoding="utf-8") as f:
    lp = json.load(f)
print(f"  total_layers: {lp['total_layers']}")
print(f"  quantized_layers: {lp['quantized_layers']}")
print(f"  fp32_layers: {lp['fp32_layers']}")
print(f"  per_channel_weight: {lp['per_channel_weight']}")

ql = [l for l in lp["layers"] if l.get("quantized")]
fl = [l for l in lp["layers"] if not l.get("quantized")]
print(f"\n  量化层[0]: {ql[0]['name']}")
print(f"    weight_per_channel: {ql[0].get('weight_per_channel')}")
sc = ql[0].get("weight_scale")
if isinstance(sc, list):
    print(f"    weight_scale: list[{len(sc)}]")
else:
    print(f"    weight_scale: {sc}")

print(f"\n  FP32 层[0]: {fl[0]['name']}")
print(f"    quantized: {fl[0].get('quantized')}")
print(f"    note: {fl[0].get('note')}")
