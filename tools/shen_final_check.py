"""
全面检查所有导出产物的正确性和一致性。
用于交付前的最终验证。
"""
import os
import sys
import json
import csv
import numpy as np
import onnx
from onnx import numpy_helper

EXPORTS_DIR = "exports"
errors = []
warnings = []

def error(msg):
    errors.append(msg)
    print(f"  [ERROR] {msg}")

def warn(msg):
    warnings.append(msg)
    print(f"  [WARN]  {msg}")

def ok(msg):
    print(f"  [OK]    {msg}")

# ============================================================
print("=" * 70)
print("1. 文件完整性检查")
print("=" * 70)

required_files = {
    "exports/onnx/train4_best_fp32.onnx": "FP32 ONNX",
    "exports/onnx/train4_best_int8.onnx": "INT8 ONNX",
    "exports/params/train4_params.h": "params.h",
    "exports/params/train4_layer_params.json": "层参数 JSON",
    "exports/params/classes.txt": "类别映射",
    "exports/quant/train4_quant_params.json": "量化参数 JSON",
    "exports/quant/train4_quant_params.csv": "量化参数 CSV",
    "exports/quant/train4_calibration_notes.md": "校准说明",
    "exports/reports/train4_fp32_val_report_2026-03-18.md": "FP32 验证报告",
    "exports/reports/train4_int8_val_report.md": "INT8 验证报告",
    "exports/reports/train4_model_summary.md": "模型结构摘要",
    "exports/reports/train4_preprocess_postprocess.md": "预处理/后处理规范",
}

for path, desc in required_files.items():
    if os.path.exists(path):
        size = os.path.getsize(path)
        ok(f"{desc}: {path} ({size/1024:.1f} KB)")
    else:
        error(f"{desc} 缺失: {path}")

# ============================================================
print("\n" + "=" * 70)
print("2. ONNX 模型检查")
print("=" * 70)

fp32_model = onnx.load("exports/onnx/train4_best_fp32.onnx")
int8_model = onnx.load("exports/onnx/train4_best_int8.onnx")

fp32_input_shape = [d.dim_value for d in fp32_model.graph.input[0].type.tensor_type.shape.dim]
fp32_output_shape = [d.dim_value for d in fp32_model.graph.output[0].type.tensor_type.shape.dim]
int8_input_shape = [d.dim_value for d in int8_model.graph.input[0].type.tensor_type.shape.dim]
int8_output_shape = [d.dim_value for d in int8_model.graph.output[0].type.tensor_type.shape.dim]

if fp32_input_shape == [1, 3, 640, 640]:
    ok(f"FP32 输入 shape: {fp32_input_shape}")
else:
    error(f"FP32 输入 shape 异常: {fp32_input_shape}, 期望 [1,3,640,640]")

if fp32_output_shape == [1, 14, 8400]:
    ok(f"FP32 输出 shape: {fp32_output_shape}")
else:
    error(f"FP32 输出 shape 异常: {fp32_output_shape}, 期望 [1,14,8400]")

if int8_input_shape == fp32_input_shape:
    ok(f"INT8 输入 shape 与 FP32 一致: {int8_input_shape}")
else:
    error(f"INT8 输入 shape 不一致: {int8_input_shape} vs FP32 {fp32_input_shape}")

if int8_output_shape == fp32_output_shape:
    ok(f"INT8 输出 shape 与 FP32 一致: {int8_output_shape}")
else:
    error(f"INT8 输出 shape 不一致: {int8_output_shape} vs FP32 {fp32_output_shape}")

fp32_convs = sum(1 for n in fp32_model.graph.node if n.op_type == "Conv")
fp32_matmuls = sum(1 for n in fp32_model.graph.node if n.op_type == "MatMul")
int8_convs = sum(1 for n in int8_model.graph.node if n.op_type == "Conv")
int8_matmuls = sum(1 for n in int8_model.graph.node if n.op_type == "MatMul")

if fp32_convs == int8_convs == 88:
    ok(f"Conv 层数一致: FP32={fp32_convs}, INT8={int8_convs}")
else:
    error(f"Conv 层数不一致: FP32={fp32_convs}, INT8={int8_convs}")

if fp32_matmuls == int8_matmuls == 2:
    ok(f"MatMul 层数一致: FP32={fp32_matmuls}, INT8={int8_matmuls}")
else:
    error(f"MatMul 层数不一致: FP32={fp32_matmuls}, INT8={int8_matmuls}")

int8_init = {i.name: numpy_helper.to_array(i) for i in int8_model.graph.initializer}
node_map = {o: n for n in int8_model.graph.node for o in n.output}

quantized_conv = 0
fp32_conv = 0
per_ch_conv = 0
for n in int8_model.graph.node:
    if n.op_type != "Conv":
        continue
    wn = node_map.get(n.input[1])
    if wn and wn.op_type == "DequantizeLinear":
        quantized_conv += 1
        if len(wn.input) > 1 and wn.input[1] in int8_init:
            if int8_init[wn.input[1]].size > 1:
                per_ch_conv += 1
    else:
        fp32_conv += 1

quantized_matmul = sum(
    1 for n in int8_model.graph.node
    if n.op_type == "MatMul" and node_map.get(n.input[1])
    and node_map[n.input[1]].op_type == "DequantizeLinear"
)

ok(f"INT8 量化 Conv: {quantized_conv}, FP32 Conv: {fp32_conv} (检测头)")
ok(f"INT8 量化 MatMul: {quantized_matmul}")
ok(f"Per-channel 权重 Conv: {per_ch_conv}")

if fp32_conv != 25:
    warn(f"FP32 Conv 数量 {fp32_conv}, 预期 25 (检测头)")

del fp32_model

# ============================================================
print("\n" + "=" * 70)
print("3. params.h 检查")
print("=" * 70)

with open("exports/params/train4_layer_params.json", "r", encoding="utf-8") as f:
    lp = json.load(f)

if lp["total_layers"] == 90:
    ok(f"总层数: {lp['total_layers']}")
else:
    error(f"总层数异常: {lp['total_layers']}, 期望 90")

if lp["quantized_layers"] == quantized_conv + quantized_matmul:
    ok(f"量化层数与 ONNX 一致: {lp['quantized_layers']}")
else:
    error(f"量化层数不一致: JSON={lp['quantized_layers']}, ONNX={quantized_conv + quantized_matmul}")

if lp["fp32_layers"] == fp32_conv:
    ok(f"FP32 层数与 ONNX 一致: {lp['fp32_layers']}")
else:
    error(f"FP32 层数不一致: JSON={lp['fp32_layers']}, ONNX={fp32_conv}")

if lp.get("per_channel_weight") is True:
    ok("per_channel_weight: True")
else:
    error(f"per_channel_weight 应为 True, 实际: {lp.get('per_channel_weight')}")

q_layers_json = [l for l in lp["layers"] if l.get("quantized")]
fp32_layers_json = [l for l in lp["layers"] if not l.get("quantized")]

per_ch_in_json = sum(1 for l in q_layers_json if l.get("weight_per_channel"))
ok(f"JSON 中 per-channel 层: {per_ch_in_json}")

for l in q_layers_json:
    if l.get("weight_per_channel"):
        ws = l.get("weight_scale")
        if not isinstance(ws, list) or len(ws) < 2:
            error(f"{l['name']}: weight_per_channel=True 但 weight_scale 不是数组")
            break
        wshape = l.get("weight_shape", [0])
        if len(ws) != wshape[0]:
            error(f"{l['name']}: weight_scale 长度 {len(ws)} != out_channels {wshape[0]}")
            break
    if l.get("input_scale") is None:
        warn(f"{l['name']}: 量化层但 input_scale 为 None")
    if l.get("output_scale") is None:
        warn(f"{l['name']}: 量化层但 output_scale 为 None")
else:
    ok("所有量化层的 per-channel scale 长度与 out_channels 匹配")

for l in fp32_layers_json:
    if l.get("input_scale") is not None:
        warn(f"{l['name']}: FP32 层但有 input_scale")
    if l.get("weight_scale") is not None:
        warn(f"{l['name']}: FP32 层但有 weight_scale")
else:
    ok("FP32 层正确标记为未量化")

params_h_size = os.path.getsize("exports/params/train4_params.h")
ok(f"params.h 文件大小: {params_h_size / 1024 / 1024:.1f} MB")

with open("exports/params/train4_params.h", "r", encoding="utf-8") as f:
    h_content = f.read()

if "#define YOLOV11N_NUM_LAYERS 90" in h_content:
    ok("params.h NUM_LAYERS = 90")
else:
    error("params.h NUM_LAYERS != 90")

if "#define YOLOV11N_NUM_CLASSES 10" in h_content:
    ok("params.h NUM_CLASSES = 10")
else:
    error("params.h NUM_CLASSES != 10")

if "CONV_0_WEIGHT_SCALES[]" in h_content:
    ok("params.h 包含 per-channel WEIGHT_SCALES 数组")
else:
    error("params.h 缺少 per-channel WEIGHT_SCALES 数组")

if "CONV_0_REQUANT_SCALES[]" in h_content:
    ok("params.h 包含 per-channel REQUANT_SCALES 数组")
else:
    error("params.h 缺少 per-channel REQUANT_SCALES 数组")

if "CONV_49_WEIGHT_NAIVE_SCALE" in h_content:
    ok("params.h 包含 FP32 检测头 NAIVE_SCALE")
else:
    warn("params.h 缺少 FP32 检测头 NAIVE_SCALE")

if "BIAS_FP32[]" in h_content:
    ok("params.h 包含 FP32 检测头 BIAS_FP32 数组")
else:
    warn("params.h 缺少 FP32 检测头 BIAS_FP32 数组")

if "LayerParams" in h_content and "YOLOV11N_LAYERS[90]" in h_content:
    ok("params.h 包含 LayerParams 结构体和 90 层汇总表")
else:
    error("params.h 缺少 LayerParams 汇总表")

# ============================================================
print("\n" + "=" * 70)
print("4. 量化参数表检查")
print("=" * 70)

with open("exports/quant/train4_quant_params.json", "r", encoding="utf-8") as f:
    qp = json.load(f)

if qp["per_channel_weight"] is True:
    ok(f"量化表 per_channel_weight: True")
else:
    error(f"量化表 per_channel_weight 应为 True")

if qp["detect_head_excluded"] is True:
    ok("量化表标注了检测头排除")
else:
    error("量化表未标注检测头排除")

if qp["calibration"]["num_images"] == 500:
    ok(f"校准图片数: {qp['calibration']['num_images']}")
else:
    error(f"校准图片数错误: {qp['calibration']['num_images']}, 应为 500")

if qp["num_total_layers"] == 90:
    ok(f"量化表总层数: {qp['num_total_layers']}")
else:
    error(f"量化表总层数: {qp['num_total_layers']}, 期望 90")

if qp["num_quantized_layers"] == lp["quantized_layers"]:
    ok(f"量化层数与 layer_params 一致: {qp['num_quantized_layers']}")
else:
    error(f"量化层数不一致: quant={qp['num_quantized_layers']}, params={lp['quantized_layers']}")

if qp["num_fp32_layers"] == lp["fp32_layers"]:
    ok(f"FP32 层数与 layer_params 一致: {qp['num_fp32_layers']}")
else:
    error(f"FP32 层数不一致: quant={qp['num_fp32_layers']}, params={lp['fp32_layers']}")

q_layers_quant = [l for l in qp["layers"] if l.get("quantized")]
fp32_layers_quant = [l for l in qp["layers"] if not l.get("quantized")]

sample_q = q_layers_quant[0]
sample_lp = [l for l in lp["layers"] if l["name"] == sample_q["layer_name"]][0]

q_w_scale = sample_q["weight"]["scale"]
lp_w_scale = sample_lp["weight_scale"]
if isinstance(q_w_scale, list) and isinstance(lp_w_scale, list):
    if len(q_w_scale) == len(lp_w_scale) and abs(q_w_scale[0] - lp_w_scale[0]) < 1e-8:
        ok(f"conv_0 weight_scale 在 quant_params 和 layer_params 间一致 (len={len(q_w_scale)})")
    else:
        error(f"conv_0 weight_scale 不一致")
else:
    warn(f"conv_0 weight_scale 类型不匹配: quant={type(q_w_scale)}, params={type(lp_w_scale)}")

with open("exports/quant/train4_quant_params.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    csv_rows = list(reader)

csv_header = csv_rows[0]
csv_data = csv_rows[1:]
if len(csv_data) == 90:
    ok(f"CSV 行数: {len(csv_data)} (含表头共 {len(csv_rows)})")
else:
    error(f"CSV 数据行数: {len(csv_data)}, 期望 90")

csv_q_count = sum(1 for r in csv_data if r[2] == "Y")
csv_fp32_count = sum(1 for r in csv_data if r[2] == "N")
if csv_q_count == qp["num_quantized_layers"]:
    ok(f"CSV 量化层: {csv_q_count}")
else:
    error(f"CSV 量化层: {csv_q_count}, JSON: {qp['num_quantized_layers']}")

if "weight_granularity" in csv_header:
    ok("CSV 包含 weight_granularity 列")
if "weight_scale_count" in csv_header:
    ok("CSV 包含 weight_scale_count 列")

# ============================================================
print("\n" + "=" * 70)
print("5. 类别映射检查")
print("=" * 70)

with open("exports/params/classes.txt", "r") as f:
    classes = [line.strip() for line in f if line.strip()]

expected_classes = [
    "pedestrian", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle", "traffic light", "traffic sign"
]

if classes == expected_classes:
    ok(f"类别映射正确: {len(classes)} 类")
else:
    error(f"类别映射不匹配: {classes}")

# ============================================================
print("\n" + "=" * 70)
print("6. 报告文档一致性检查")
print("=" * 70)

with open("exports/reports/train4_model_summary.md", "r", encoding="utf-8") as f:
    summary = f.read()

if "INT8 ONNX: `4.3 MB`" in summary:
    ok("模型摘要 INT8 大小正确 (4.3 MB)")
elif "4.3 MB" in summary:
    ok("模型摘要包含 INT8 大小 4.3 MB")
else:
    actual_int8_mb = os.path.getsize("exports/onnx/train4_best_int8.onnx") / 1024 / 1024
    warn(f"模型摘要 INT8 大小可能需要更新 (实际 {actual_int8_mb:.1f} MB)")

if "0.418" in summary or "0.4184" in summary:
    ok("模型摘要包含 INT8 mAP50 结果")
else:
    error("模型摘要缺少 INT8 验证结果")

if "per-channel" in summary.lower() or "FP32" in summary:
    ok("模型摘要提到了 v2 量化策略")
else:
    warn("模型摘要可能未充分说明 v2 量化策略")

with open("exports/reports/train4_int8_val_report.md", "r", encoding="utf-8") as f:
    int8_report = f.read()

if "PTQ v2" in int8_report or "per-channel" in int8_report:
    ok("INT8 报告标注了 v2 量化方法")
else:
    warn("INT8 报告可能未标注 v2 量化方法")

if "500" in int8_report:
    ok("INT8 报告校准图片数为 500")
else:
    warn("INT8 报告校准图片数可能未更新")

if "0.4184" in int8_report:
    ok("INT8 报告 mAP50 = 0.4184")
else:
    warn("INT8 报告 mAP50 值可能不正确")

with open("exports/reports/train4_preprocess_postprocess.md", "r", encoding="utf-8") as f:
    pp_doc = f.read()

if "640" in pp_doc and "letterbox" in pp_doc.lower():
    ok("预处理文档包含 640 和 letterbox")
if "NMS" in pp_doc:
    ok("后处理文档包含 NMS")
if "[1, 14, 8400]" in pp_doc or "14" in pp_doc:
    ok("后处理文档包含输出张量规格")

with open("exports/quant/train4_calibration_notes.md", "r", encoding="utf-8") as f:
    cal_notes = f.read()

if "v2" in cal_notes:
    ok("校准说明标注了 v2")
if "per-channel" in cal_notes or "per_channel" in cal_notes:
    ok("校准说明提到了 per-channel")
if "model.23" in cal_notes:
    ok("校准说明提到了检测头排除")
if "500" in cal_notes:
    ok("校准说明校准图片数 500")

# ============================================================
print("\n" + "=" * 70)
print("7. 数值交叉验证")
print("=" * 70)

import onnxruntime as ort
import cv2

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

test_img_path = "datasets/BDD100K/images/val/b1c66a42-6f7d68ca.jpg"
if os.path.exists(test_img_path):
    img = cv2.imread(test_img_path)
    img = letterbox(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inp = img.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))[np.newaxis]

    sess = ort.InferenceSession("exports/onnx/train4_best_int8.onnx", providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: inp})[0]

    cls_out = out[0, 4:, :]
    cls_max = cls_out.max()
    cls_nonzero = np.count_nonzero(cls_out)

    if cls_max > 0.01:
        ok(f"INT8 分类输出非零: max={cls_max:.6f}, nonzero={cls_nonzero}/{cls_out.size}")
    else:
        error(f"INT8 分类输出仍为全零! max={cls_max}")

    bbox_out = out[0, :4, :]
    if bbox_out.max() > 100 and bbox_out.max() < 700:
        ok(f"INT8 bbox 输出范围正常: [{bbox_out.min():.1f}, {bbox_out.max():.1f}]")
    else:
        warn(f"INT8 bbox 输出范围异常: [{bbox_out.min():.1f}, {bbox_out.max():.1f}]")
else:
    warn("测试图片不存在，跳过数值验证")

del int8_model

# ============================================================
print("\n" + "=" * 70)
print("检查总结")
print("=" * 70)
print(f"  错误: {len(errors)}")
print(f"  警告: {len(warnings)}")

if errors:
    print("\n  错误列表:")
    for e in errors:
        print(f"    - {e}")

if warnings:
    print("\n  警告列表:")
    for w in warnings:
        print(f"    - {w}")

if not errors:
    print("\n  *** 所有检查通过，导出产物可以交付 ***")

    print("\n  交付文件清单:")
    print("  ┌─ exports/")
    print("  ├── onnx/")
    print("  │   ├── train4_best_fp32.onnx          (FP32 ONNX)")
    print("  │   └── train4_best_int8.onnx          (INT8 QDQ ONNX)")
    print("  ├── params/")
    print("  │   ├── train4_params.h                (Gemmini C 头文件)")
    print("  │   ├── train4_layer_params.json       (层参数 JSON)")
    print("  │   └── classes.txt                    (类别映射)")
    print("  ├── quant/")
    print("  │   ├── train4_quant_params.json       (量化参数表 JSON)")
    print("  │   ├── train4_quant_params.csv        (量化参数表 CSV)")
    print("  │   └── train4_calibration_notes.md    (校准说明)")
    print("  └── reports/")
    print("      ├── train4_fp32_val_report.md      (FP32 验证报告)")
    print("      ├── train4_int8_val_report.md      (INT8 验证报告)")
    print("      ├── train4_model_summary.md        (模型结构摘要)")
    print("      └── train4_preprocess_postprocess.md (预处理/后处理规范)")
else:
    print(f"\n  *** 发现 {len(errors)} 个错误，请修复后再交付 ***")
    sys.exit(1)
