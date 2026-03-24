"""
YOLOv11n params.h 导出脚本 (v3 - NHWC for Gemmini)

从 INT8 QDQ ONNX 模型中提取量化权重、偏置和卷积参数，
转换为 Gemmini NHWC 布局，生成 C 头文件。

v3 改动 (NHWC):
  - 标准 Conv 权重: OIHW [OC,IC,kH,kW] → HWIO [kH*kW*IC, OC]
  - 1×1 Conv 权重:  OIHW [OC,IC,1,1]   → IO   [IC, OC]
  - DW Conv 权重:   [C,1,kH,kW]        → [C, kH*kW]  (squeeze + reshape)
  - Bias:           [OC]               → [OC]  (不变)
  - MatMul 权重:    保持原始 2D layout

用法:
    conda activate yolo_v11
    python tools/shen_export_params.py
"""

import os
import sys
import json
import numpy as np
import onnx
from onnx import numpy_helper


INT8_ONNX = "exports/onnx/train4_best_int8.onnx"
FP32_ONNX = "exports/onnx/train4_best_fp32.onnx"
PARAMS_H = "exports/params/train4_params.h"
LAYER_PARAMS_JSON = "exports/params/train4_layer_params.json"

NUM_CLASSES = 10
INPUT_H = 640
INPUT_W = 640
INPUT_C = 3
CLASS_NAMES = [
    "pedestrian", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle", "traffic light", "traffic sign"
]


def convert_weight_nhwc(w, layer):
    """将 ONNX OIHW 权重转换为 Gemmini NHWC 布局。

    返回 (converted_array, layout_str, nhwc_shape)
    """
    if layer["op_type"] == "MatMul":
        return w, "matmul", list(w.shape)

    ks = layer.get("kernel_shape", [0, 0])
    group = layer.get("group", 1)
    oc = w.shape[0]

    if group > 1 and w.shape[1] == 1:
        # DW Conv: [C, 1, kH, kW] → [C, kH*kW]
        c = w.shape[0]
        out = np.squeeze(w, axis=1).reshape(c, -1)
        return out, "dw_chw", list(out.shape)

    if ks == [1, 1]:
        # 1×1 Conv: [OC, IC, 1, 1] → [IC, OC]
        out = np.squeeze(w, axis=(2, 3)).T
        return out, "1x1_io", list(out.shape)

    # Standard Conv: [OC, IC, kH, kW] → permute(2,3,1,0) → [kH,kW,IC,OC] → reshape [kH*kW*IC, OC]
    out = w.transpose(2, 3, 1, 0).reshape(-1, oc)
    return out, "hwio", list(out.shape)


def load_model_and_initializers(onnx_path):
    model = onnx.load(onnx_path)
    init_map = {}
    for init in model.graph.initializer:
        init_map[init.name] = numpy_helper.to_array(init)
    return model, init_map


def extract_conv_layers(model, init_map):
    node_output_to_node = {}
    for node in model.graph.node:
        for out in node.output:
            node_output_to_node[out] = node

    layers = []
    conv_idx = 0
    matmul_idx = 0

    for node in model.graph.node:
        if node.op_type not in ("Conv", "MatMul"):
            continue

        layer = {"op_type": node.op_type, "node_name": node.name}

        if node.op_type == "Conv":
            layer["name"] = f"conv_{conv_idx}"
            conv_idx += 1
            for attr in node.attribute:
                if attr.name == "kernel_shape":
                    layer["kernel_shape"] = list(attr.ints)
                elif attr.name == "strides":
                    layer["strides"] = list(attr.ints)
                elif attr.name == "pads":
                    layer["pads"] = list(attr.ints)
                elif attr.name == "group":
                    layer["group"] = int(attr.i)
                elif attr.name == "dilations":
                    layer["dilations"] = list(attr.ints)
            layer.setdefault("group", 1)
            layer.setdefault("strides", [1, 1])
            layer.setdefault("pads", [0, 0, 0, 0])
            layer.setdefault("dilations", [1, 1])
        else:
            layer["name"] = f"matmul_{matmul_idx}"
            matmul_idx += 1

        weight_input = node.input[1]
        weight_node = node_output_to_node.get(weight_input)
        if weight_node and weight_node.op_type == "DequantizeLinear":
            quant_data_name = weight_node.input[0]
            if quant_data_name in init_map:
                raw_w = init_map[quant_data_name]
                layer["weight_shape_oihw"] = list(raw_w.shape)
                converted, layout, nhwc_shape = convert_weight_nhwc(raw_w, layer)
                layer["weight_int8"] = converted
                layer["weight_layout"] = layout
                layer["weight_shape"] = nhwc_shape
            if len(weight_node.input) > 1 and weight_node.input[1] in init_map:
                scale_arr = init_map[weight_node.input[1]]
                if scale_arr.size == 1:
                    layer["weight_scale"] = float(scale_arr.flat[0])
                    layer["weight_per_channel"] = False
                else:
                    layer["weight_scale"] = [float(s) for s in scale_arr.flat]
                    layer["weight_per_channel"] = True
            if len(weight_node.input) > 2 and weight_node.input[2] in init_map:
                zp_arr = init_map[weight_node.input[2]]
                if zp_arr.size == 1:
                    layer["weight_zp"] = int(zp_arr.flat[0])
                else:
                    layer["weight_zp"] = [int(z) for z in zp_arr.flat]
            else:
                layer["weight_zp"] = 0
            layer["quantized"] = True
        elif weight_input in init_map:
            raw_w = init_map[weight_input]
            layer["weight_shape_oihw"] = list(raw_w.shape)
            converted, layout, nhwc_shape = convert_weight_nhwc(raw_w, layer)
            layer["weight_fp32"] = converted
            layer["weight_layout"] = layout
            layer["weight_shape"] = nhwc_shape
            layer["quantized"] = False
        else:
            layer["quantized"] = False

        if node.op_type == "Conv" and len(node.input) > 2 and node.input[2]:
            bias_input = node.input[2]
            bias_node = node_output_to_node.get(bias_input)
            if bias_node and bias_node.op_type == "DequantizeLinear":
                quant_data_name = bias_node.input[0]
                if quant_data_name in init_map:
                    layer["bias_int32"] = init_map[quant_data_name]
                if len(bias_node.input) > 1 and bias_node.input[1] in init_map:
                    layer["bias_scale"] = float(init_map[bias_node.input[1]].flat[0])
            elif bias_input in init_map:
                layer["bias_fp32"] = init_map[bias_input]

        act_input = node.input[0]
        act_node = node_output_to_node.get(act_input)
        if act_node and act_node.op_type == "DequantizeLinear":
            if len(act_node.input) > 1 and act_node.input[1] in init_map:
                layer["input_scale"] = float(init_map[act_node.input[1]].flat[0])
            if len(act_node.input) > 2 and act_node.input[2] in init_map:
                layer["input_zp"] = int(init_map[act_node.input[2]].flat[0])
            else:
                layer["input_zp"] = 0

        for n2 in model.graph.node:
            if n2.op_type == "QuantizeLinear" and node.output[0] in n2.input:
                if len(n2.input) > 1 and n2.input[1] in init_map:
                    layer["output_scale"] = float(init_map[n2.input[1]].flat[0])
                if len(n2.input) > 2 and n2.input[2] in init_map:
                    layer["output_zp"] = int(init_map[n2.input[2]].flat[0])
                else:
                    layer["output_zp"] = 0
                break

        layers.append(layer)

    return layers


def float_array_to_c(arr, name, indent="    "):
    lines = [f"static const float {name}[] = {{"]
    chunk_size = 8
    for i in range(0, len(arr), chunk_size):
        chunk = arr[i:i + chunk_size]
        vals = ", ".join(f"{v:.10e}f" for v in chunk)
        lines.append(f"{indent}{vals},")
    lines.append("};")
    return "\n".join(lines)


def array_to_c(arr, name, dtype_str, indent="    "):
    flat = arr.flatten()
    lines = [f"static const {dtype_str} {name}[] = {{"]
    chunk_size = 16
    for i in range(0, len(flat), chunk_size):
        chunk = flat[i:i + chunk_size]
        vals = ", ".join(str(int(v)) for v in chunk)
        lines.append(f"{indent}{vals},")
    lines.append("};")
    return "\n".join(lines)


LAYOUT_COMMENT = {
    "hwio": "HWIO [kH*kW*IC, OC]",
    "1x1_io": "IO [IC, OC]",
    "dw_chw": "DW [C, kH*kW]",
    "matmul": "MatMul [M, N]",
}


def generate_params_h(layers, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    quantized_count = sum(1 for l in layers if l.get("quantized"))
    fp32_count = len(layers) - quantized_count

    lines = []
    lines.append("/* Auto-generated by shen_export_params.py (v3 - NHWC) */")
    lines.append("/* YOLOv11n INT8 params for Gemmini deployment */")
    lines.append(f"/* Model: train4, Dataset: BDD100K, Classes: {NUM_CLASSES} */")
    lines.append("/* Weight layout: NHWC (Gemmini native) */")
    lines.append("/*   Standard Conv: OIHW -> HWIO [kH*kW*IC, OC] */")
    lines.append("/*   1x1 Conv:      OIHW -> IO   [IC, OC]       */")
    lines.append("/*   DW Conv:       [C,1,kH,kW] -> [C,kH*kW]   */")
    lines.append("/*   Bias:          [OC] (unchanged)             */")
    lines.append("")
    lines.append("#ifndef YOLOV11N_PARAMS_H")
    lines.append("#define YOLOV11N_PARAMS_H")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")

    lines.append(f"#define YOLOV11N_INPUT_H {INPUT_H}")
    lines.append(f"#define YOLOV11N_INPUT_W {INPUT_W}")
    lines.append(f"#define YOLOV11N_INPUT_C {INPUT_C}")
    lines.append(f"#define YOLOV11N_NUM_CLASSES {NUM_CLASSES}")
    lines.append(f"#define YOLOV11N_NUM_CONV_LAYERS {sum(1 for l in layers if l['op_type'] == 'Conv')}")
    lines.append(f"#define YOLOV11N_NUM_MATMUL_LAYERS {sum(1 for l in layers if l['op_type'] == 'MatMul')}")
    lines.append(f"#define YOLOV11N_NUM_LAYERS {len(layers)}")
    lines.append(f"#define YOLOV11N_NUM_QUANTIZED_LAYERS {quantized_count}")
    lines.append(f"#define YOLOV11N_NUM_FP32_LAYERS {fp32_count}")
    lines.append("")

    lines.append("/* Class names */")
    lines.append("static const char* YOLOV11N_CLASS_NAMES[] = {")
    for name in CLASS_NAMES:
        lines.append(f'    "{name}",')
    lines.append("};")
    lines.append("")

    lines.append("/* ===== Layer Quantization Parameters ===== */")
    lines.append("")

    for i, layer in enumerate(layers):
        lname = layer["name"].upper()
        is_q = layer.get("quantized", False)
        tag = "INT8" if is_q else "FP32"
        layout = LAYOUT_COMMENT.get(layer.get("weight_layout", ""), "")
        lines.append(f"/* Layer {i}: {layer['name']} ({layer['op_type']}) [{tag}] layout={layout} */")
        lines.append(f"#define {lname}_QUANTIZED {1 if is_q else 0}")

        if layer["op_type"] == "Conv":
            oihw = layer.get("weight_shape_oihw", [0, 0, 0, 0])
            ks = layer.get("kernel_shape", [0, 0])
            st = layer.get("strides", [1, 1])
            pd = layer.get("pads", [0, 0, 0, 0])
            gr = layer.get("group", 1)

            lines.append(f"#define {lname}_OUT_C {oihw[0]}")
            lines.append(f"#define {lname}_IN_C {oihw[1]}")
            lines.append(f"#define {lname}_KH {ks[0]}")
            lines.append(f"#define {lname}_KW {ks[1]}")
            lines.append(f"#define {lname}_STRIDE_H {st[0]}")
            lines.append(f"#define {lname}_STRIDE_W {st[1]}")
            lines.append(f"#define {lname}_PAD_H {pd[0]}")
            lines.append(f"#define {lname}_PAD_W {pd[1]}")
            lines.append(f"#define {lname}_GROUP {gr}")

            nhwc_shape = layer.get("weight_shape", [])
            lines.append(f"#define {lname}_WEIGHT_ROWS {nhwc_shape[0] if len(nhwc_shape) >= 1 else 0}")
            lines.append(f"#define {lname}_WEIGHT_COLS {nhwc_shape[1] if len(nhwc_shape) >= 2 else 0}")

        if is_q:
            if "input_scale" in layer:
                lines.append(f"#define {lname}_INPUT_SCALE {layer['input_scale']:.10e}f")
                lines.append(f"#define {lname}_INPUT_ZP {layer.get('input_zp', 0)}")

            w_scale = layer.get("weight_scale")
            if layer.get("weight_per_channel") and isinstance(w_scale, list):
                lines.append(f"#define {lname}_WEIGHT_PER_CHANNEL 1")
                lines.append(f"#define {lname}_WEIGHT_SCALE_COUNT {len(w_scale)}")
            elif w_scale is not None:
                lines.append(f"#define {lname}_WEIGHT_PER_CHANNEL 0")
                lines.append(f"#define {lname}_WEIGHT_SCALE {w_scale:.10e}f")

            w_zp = layer.get("weight_zp", 0)
            if isinstance(w_zp, list):
                lines.append(f"#define {lname}_WEIGHT_ZP 0  /* all zeros (symmetric) */")
            else:
                lines.append(f"#define {lname}_WEIGHT_ZP {w_zp}")

            if "output_scale" in layer:
                lines.append(f"#define {lname}_OUTPUT_SCALE {layer['output_scale']:.10e}f")
                lines.append(f"#define {lname}_OUTPUT_ZP {layer.get('output_zp', 0)}")

            if "bias_scale" in layer:
                lines.append(f"#define {lname}_BIAS_SCALE {layer['bias_scale']:.10e}f")

        lines.append("")

    lines.append("/* ===== Per-Channel Weight Scale Arrays ===== */")
    lines.append("/* Absorption strategy: unified = max(requant_scales)")
    lines.append("   → all ratio[oc] = rq[oc]/unified ≤ 1.0 → ZERO clip errors")
    lines.append("   → low-scale channels lose some precision (round only, no clip)")
    lines.append("   Bias is also absorbed: bias_absorbed[oc] = round(bias[oc] * ratio[oc]) */")
    lines.append("")

    for i, layer in enumerate(layers):
        if not layer.get("weight_per_channel"):
            continue
        lname = layer["name"].upper()
        w_scale = layer["weight_scale"]
        lines.append(f"/* {layer['name']} per-channel weight scales: {len(w_scale)} channels */")
        lines.append(float_array_to_c(w_scale, f"{lname}_WEIGHT_SCALES"))
        lines.append("")

        if "input_scale" in layer and "output_scale" in layer:
            i_scale = layer["input_scale"]
            o_scale = layer["output_scale"]
            rq_scales = [(i_scale * ws) / o_scale for ws in w_scale]
            max_rq = max(rq_scales)

            lines.append(f"/* {layer['name']} per-channel requant scales (original) */")
            lines.append(float_array_to_c(rq_scales, f"{lname}_REQUANT_SCALES"))
            lines.append("")
            lines.append(f"/* {layer['name']} unified requant scale (max of per-channel, zero-clip) */")
            lines.append(f"#define {lname}_REQUANT_UNIFIED {max_rq:.10e}f")
            lines.append("")

            layer["_rq_scales"] = rq_scales
            layer["_rq_unified"] = max_rq

    lines.append("/* ===== Layer Weight Arrays (INT8 quantized, NHWC layout) ===== */")
    lines.append("/* _WEIGHT = original per-channel quantized weights */")
    lines.append("/* _WEIGHT_ABSORBED = weights with per-channel requant absorbed (for Gemmini scalar output_scale) */")
    lines.append("")

    for i, layer in enumerate(layers):
        lname = layer["name"].upper()

        if "weight_int8" in layer:
            w = layer["weight_int8"]
            layout = LAYOUT_COMMENT.get(layer.get("weight_layout", ""), "")
            oihw = layer.get("weight_shape_oihw", [])
            lines.append(f"/* {layer['name']} weight [INT8 {layout}]: "
                         f"OIHW {oihw} -> NHWC {list(w.shape)}, total {w.size} */")
            lines.append(array_to_c(w, f"{lname}_WEIGHT", "int8_t"))
            lines.append("")

            rq_scales = layer.get("_rq_scales")
            rq_unified = layer.get("_rq_unified")
            if rq_scales and rq_unified and rq_unified > 0:
                w_float = w.astype(np.float64)
                ratios = np.array([rs / rq_unified for rs in rq_scales])
                layer["_ratios"] = ratios
                wl = layer.get("weight_layout", "")
                if wl in ("hwio", "1x1_io"):
                    absorbed = np.clip(np.round(w_float * ratios[np.newaxis, :]), -128, 127).astype(np.int8)
                elif wl == "dw_chw":
                    absorbed = np.clip(np.round(w_float * ratios[:, np.newaxis]), -128, 127).astype(np.int8)
                else:
                    absorbed = w

                n_clip = int(np.sum(np.abs(np.round(w_float * (ratios[np.newaxis, :] if wl in ("hwio", "1x1_io") else ratios[:, np.newaxis]))) > 127)) if wl in ("hwio", "1x1_io", "dw_chw") else 0
                lines.append(f"/* {layer['name']} weight ABSORBED (unified=max, zero-clip): "
                             f"requant = {rq_unified:.6e}, "
                             f"ratio range [{ratios.min():.3f}, {ratios.max():.3f}], "
                             f"clip count = {n_clip} */")
                lines.append(array_to_c(absorbed, f"{lname}_WEIGHT_ABSORBED", "int8_t"))
                lines.append("")

        if "bias_int32" in layer:
            b = layer["bias_int32"]
            lines.append(f"/* {layer['name']} bias [INT32]: {list(b.shape)}, total {b.size} */")
            lines.append(array_to_c(b, f"{lname}_BIAS", "int32_t"))
            lines.append("")

            ratios = layer.get("_ratios")
            if ratios is not None:
                b_float = b.astype(np.float64)
                bias_absorbed = np.round(b_float * ratios).astype(np.int32)
                lines.append(f"/* {layer['name']} bias ABSORBED: bias_absorbed[oc] = round(bias[oc] * ratio[oc]) */")
                lines.append(array_to_c(bias_absorbed, f"{lname}_BIAS_ABSORBED", "int32_t"))
                lines.append("")

    lines.append("/* ===== Layer Weight Arrays (FP32 detect head, NHWC layout) ===== */")
    lines.append("")

    for i, layer in enumerate(layers):
        lname = layer["name"].upper()

        if "weight_fp32" in layer:
            w_nhwc = layer["weight_fp32"]
            oihw = layer.get("weight_shape_oihw", [])
            layout = LAYOUT_COMMENT.get(layer.get("weight_layout", ""), "")

            w_int8 = np.clip(np.round(w_nhwc * 127.0 / (np.abs(w_nhwc).max() + 1e-10)), -128, 127).astype(np.int8)
            fp32_scale = float(np.abs(w_nhwc).max() / 127.0)
            lines.append(f"/* {layer['name']} weight [FP32->INT8 naive {layout}]: "
                         f"OIHW {oihw} -> NHWC {list(w_nhwc.shape)}, total {w_nhwc.size} */")
            lines.append(f"#define {lname}_WEIGHT_NAIVE_SCALE {fp32_scale:.10e}f")
            lines.append(array_to_c(w_int8, f"{lname}_WEIGHT", "int8_t"))
            lines.append("")

        if "bias_fp32" in layer:
            b = layer["bias_fp32"]
            lines.append(f"/* {layer['name']} bias [FP32]: {list(b.shape)}, total {b.size} */")
            flat = b.flatten()
            blines = [f"static const float {lname}_BIAS_FP32[] = {{"]
            chunk_size = 8
            for ci in range(0, len(flat), chunk_size):
                chunk = flat[ci:ci + chunk_size]
                vals = ", ".join(f"{float(v):.8e}f" for v in chunk)
                blines.append(f"    {vals},")
            blines.append("};")
            lines.append("\n".join(blines))
            lines.append("")

    lines.append("/* ===== Layer Parameter Summary Table ===== */")
    lines.append("")
    lines.append("typedef struct {")
    lines.append("    const char* name;")
    lines.append("    const char* op_type;")
    lines.append("    int quantized;")
    lines.append("    int out_c, in_c, kh, kw;")
    lines.append("    int stride_h, stride_w;")
    lines.append("    int pad_h, pad_w;")
    lines.append("    int group;")
    lines.append("    int weight_rows, weight_cols;")
    lines.append("    float input_scale, output_scale;")
    lines.append("    int input_zp, output_zp;")
    lines.append("    int weight_per_channel;")
    lines.append("    int weight_scale_count;")
    lines.append("    const float* weight_scales;")
    lines.append("    const float* requant_scales;")
    lines.append("    float requant_unified;          /* max(requant_scales), scalar for Gemmini */")
    lines.append("    const int8_t* weight;")
    lines.append("    const int8_t* weight_absorbed;  /* per-ch requant absorbed into weight */")
    lines.append("    const int32_t* bias;")
    lines.append("    const int32_t* bias_absorbed;   /* bias with ratio absorbed */")
    lines.append("    const float* bias_fp32;")
    lines.append("    int weight_size;")
    lines.append("    int bias_size;")
    lines.append("} LayerParams;")
    lines.append("")

    lines.append(f"static const LayerParams YOLOV11N_LAYERS[{len(layers)}] = {{")
    for i, layer in enumerate(layers):
        lname = layer["name"].upper()
        oihw = layer.get("weight_shape_oihw", [0, 0, 0, 0])
        nhwc = layer.get("weight_shape", [0, 0])
        ks = layer.get("kernel_shape", [0, 0])
        st = layer.get("strides", [1, 1])
        pd = layer.get("pads", [0, 0, 0, 0])
        gr = layer.get("group", 1)
        is_q = layer.get("quantized", False)

        out_c = oihw[0] if len(oihw) >= 1 else 0
        in_c = oihw[1] if len(oihw) >= 2 else 0
        kh = ks[0] if len(ks) >= 1 else 0
        kw = ks[1] if len(ks) >= 2 else 0
        w_rows = nhwc[0] if len(nhwc) >= 1 else 0
        w_cols = nhwc[1] if len(nhwc) >= 2 else 0

        i_scale = layer.get("input_scale", 0.0)
        o_scale = layer.get("output_scale", 0.0)
        i_zp = layer.get("input_zp", 0)
        o_zp = layer.get("output_zp", 0)

        per_ch = 1 if layer.get("weight_per_channel") else 0
        w_scale_arr = layer.get("weight_scale")
        if isinstance(w_scale_arr, list):
            scale_count = len(w_scale_arr)
            scales_ptr = f"{lname}_WEIGHT_SCALES"
            rq_ptr = f"{lname}_REQUANT_SCALES" if (i_scale and o_scale) else "NULL"
        else:
            scale_count = 1 if w_scale_arr else 0
            scales_ptr = "NULL"
            rq_ptr = "NULL"

        has_int8_w = "weight_int8" in layer
        has_fp32_w = "weight_fp32" in layer
        w_ptr = f"{lname}_WEIGHT" if (has_int8_w or has_fp32_w) else "NULL"
        w_size = 0
        if has_int8_w:
            w_size = layer["weight_int8"].size
        elif has_fp32_w:
            w_size = layer["weight_fp32"].size

        rq_unified_val = layer.get("_rq_unified", 0.0)
        has_absorbed = layer.get("_rq_scales") is not None and has_int8_w
        w_abs_ptr = f"{lname}_WEIGHT_ABSORBED" if has_absorbed else "NULL"

        has_int32_b = "bias_int32" in layer
        has_fp32_b = "bias_fp32" in layer
        b_ptr = f"{lname}_BIAS" if has_int32_b else "NULL"
        b_abs_ptr = f"{lname}_BIAS_ABSORBED" if (has_int32_b and has_absorbed) else "NULL"
        b_fp32_ptr = f"{lname}_BIAS_FP32" if has_fp32_b else "NULL"
        b_size = 0
        if has_int32_b:
            b_size = layer["bias_int32"].size
        elif has_fp32_b:
            b_size = layer["bias_fp32"].size

        lines.append(f"    /* [{i}] {layer['name']} {'[INT8]' if is_q else '[FP32]'} */")
        lines.append(f'    {{"{layer["name"]}", "{layer["op_type"]}", '
                     f"{1 if is_q else 0}, "
                     f"{out_c}, {in_c}, {kh}, {kw}, "
                     f"{st[0]}, {st[1]}, {pd[0]}, {pd[1]}, {gr}, "
                     f"{w_rows}, {w_cols}, "
                     f"{i_scale:.10e}f, {o_scale:.10e}f, "
                     f"{i_zp}, {o_zp}, "
                     f"{per_ch}, {scale_count}, "
                     f"{scales_ptr}, {rq_ptr}, "
                     f"{rq_unified_val:.10e}f, "
                     f"{w_ptr}, {w_abs_ptr}, {b_ptr}, {b_abs_ptr}, {b_fp32_ptr}, "
                     f"{w_size}, {b_size}}},")

    lines.append("};")
    lines.append("")
    lines.append("#endif /* YOLOV11N_PARAMS_H */")
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return len(layers)


def export_layer_params_json(layers, output_path):
    json_layers = []
    for layer in layers:
        info = {
            "name": layer["name"],
            "op_type": layer["op_type"],
            "node_name": layer.get("node_name"),
            "quantized": layer.get("quantized", False),
            "weight_shape_oihw": layer.get("weight_shape_oihw"),
            "weight_shape_nhwc": layer.get("weight_shape"),
            "weight_layout": layer.get("weight_layout"),
        }
        if layer["op_type"] == "Conv":
            info["kernel_shape"] = layer.get("kernel_shape")
            info["strides"] = layer.get("strides")
            info["pads"] = layer.get("pads")
            info["group"] = layer.get("group")

        info["weight_per_channel"] = layer.get("weight_per_channel", False)

        if layer.get("quantized"):
            info["input_scale"] = layer.get("input_scale")
            info["input_zp"] = layer.get("input_zp")
            w_scale = layer.get("weight_scale")
            if isinstance(w_scale, list):
                info["weight_scale_count"] = len(w_scale)
                info["weight_scale_min"] = min(w_scale)
                info["weight_scale_max"] = max(w_scale)
                info["weight_scale"] = w_scale
            else:
                info["weight_scale"] = w_scale
            info["weight_zp"] = layer.get("weight_zp")
            info["output_scale"] = layer.get("output_scale")
            info["output_zp"] = layer.get("output_zp")
            rq_unified = layer.get("_rq_unified")
            if rq_unified:
                info["requant_unified"] = rq_unified
                info["has_absorbed_weight"] = True
            if "bias_int32" in layer:
                info["bias_size"] = int(layer["bias_int32"].size)
                info["bias_scale"] = layer.get("bias_scale")
        else:
            info["note"] = "FP32 (not quantized, detect head)"
            if "bias_fp32" in layer:
                info["bias_size"] = int(layer["bias_fp32"].size)

        json_layers.append(info)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "yolov11n",
            "dataset": "BDD100K",
            "num_classes": NUM_CLASSES,
            "input_shape_nchw": [1, INPUT_C, INPUT_H, INPUT_W],
            "input_shape_nhwc": [1, INPUT_H, INPUT_W, INPUT_C],
            "weight_layout": "NHWC (Gemmini native)",
            "layout_conversion": {
                "standard_conv": "OIHW [OC,IC,kH,kW] -> HWIO [kH*kW*IC, OC]",
                "1x1_conv": "OIHW [OC,IC,1,1] -> IO [IC, OC]",
                "dw_conv": "[C,1,kH,kW] -> [C,kH*kW]",
                "matmul": "unchanged",
                "bias": "[OC] unchanged",
            },
            "total_layers": len(json_layers),
            "quantized_layers": sum(1 for l in json_layers if l.get("quantized")),
            "fp32_layers": sum(1 for l in json_layers if not l.get("quantized")),
            "per_channel_weight": True,
            "layers": json_layers,
        }, f, indent=2, ensure_ascii=False)
    print(f"[导出] 层参数 JSON: {output_path}")


def main():
    if not os.path.exists(INT8_ONNX):
        print(f"[错误] 未找到 INT8 ONNX: {INT8_ONNX}")
        sys.exit(1)

    print(f"[加载] INT8 ONNX: {INT8_ONNX}")
    model, init_map = load_model_and_initializers(INT8_ONNX)

    print("[提取] 层参数与权重...")
    layers = extract_conv_layers(model, init_map)

    conv_count = sum(1 for l in layers if l["op_type"] == "Conv")
    matmul_count = sum(1 for l in layers if l["op_type"] == "MatMul")
    quantized_count = sum(1 for l in layers if l.get("quantized"))
    fp32_count = len(layers) - quantized_count
    print(f"[信息] 共提取 {len(layers)} 层: Conv={conv_count}, MatMul={matmul_count}")
    print(f"[信息] 已量化: {quantized_count}, FP32: {fp32_count}")

    layout_stats = {}
    for l in layers:
        lt = l.get("weight_layout", "unknown")
        layout_stats[lt] = layout_stats.get(lt, 0) + 1
    print(f"[布局] 权重转换统计:")
    for lt, cnt in sorted(layout_stats.items()):
        desc = LAYOUT_COMMENT.get(lt, lt)
        print(f"  {lt}: {cnt} 层 → {desc}")

    total_weight_params = sum(l.get("weight_int8", l.get("weight_fp32", np.array([]))).size
                              for l in layers if "weight_int8" in l or "weight_fp32" in l)
    total_bias_params = sum(
        l.get("bias_int32", l.get("bias_fp32", np.array([]))).size
        for l in layers if "bias_int32" in l or "bias_fp32" in l
    )
    print(f"[信息] 总权重参数: {total_weight_params}, 总偏置参数: {total_bias_params}")

    print(f"[生成] params.h (NHWC): {PARAMS_H}")
    num_layers = generate_params_h(layers, PARAMS_H)

    file_size = os.path.getsize(PARAMS_H) / 1024 / 1024
    print(f"[信息] params.h 大小: {file_size:.1f} MB")

    export_layer_params_json(layers, LAYER_PARAMS_JSON)

    print(f"\n[完成] params.h 导出成功 (NHWC layout)")
    print(f"  params.h:        {PARAMS_H}")
    print(f"  layer_params.json: {LAYER_PARAMS_JSON}")
    print(f"  总层数:          {num_layers}")
    print(f"  已量化层:        {quantized_count}")
    print(f"  FP32 层:         {fp32_count}")
    print(f"  总权重参数:      {total_weight_params}")
    print(f"  总偏置参数:      {total_bias_params}")


if __name__ == "__main__":
    main()
