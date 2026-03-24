"""
YOLOv11n 量化参数表导出脚本 (v2)

从 INT8 QDQ ONNX 模型中提取每层的量化参数（scale, zero_point, dtype），
按模板格式输出 JSON 和 CSV。

v2 改动:
  - 正确处理 per-channel 权重 scale（数组而非标量）
  - 区分已量化层 vs FP32 层（检测头 model.23）
  - 修正 JSON 元数据
  - Per-channel requant_scale 支持

用法:
    conda activate yolo_v11
    python tools/shen_export_quant.py
"""

import os
import sys
import json
import csv
import numpy as np
import onnx
from onnx import numpy_helper


INT8_ONNX = "exports/onnx/train4_best_int8.onnx"
FP32_ONNX = "exports/onnx/train4_best_fp32.onnx"
QUANT_JSON = "exports/quant/train4_quant_params.json"
QUANT_CSV = "exports/quant/train4_quant_params.csv"

GEMMINI_OPS = {"Conv", "MatMul", "Add"}
CPU_OPS = {"Sigmoid", "Mul", "Resize", "Concat", "Split", "Reshape",
           "Transpose", "Softmax", "Slice", "Div", "Sub", "MaxPool"}


def get_initializer_map(model):
    """构建 initializer name -> numpy array 映射。"""
    init_map = {}
    for init in model.graph.initializer:
        init_map[init.name] = numpy_helper.to_array(init)
    return init_map


def extract_quant_params(model, init_map):
    """从 QDQ 节点中提取量化参数。"""
    layers = []
    node_output_to_node = {}
    for node in model.graph.node:
        for out in node.output:
            node_output_to_node[out] = node

    conv_idx = 0
    matmul_idx = 0

    for node in model.graph.node:
        if node.op_type not in ("Conv", "MatMul"):
            continue

        layer_info = {
            "layer_type": node.op_type,
            "node_name": node.name,
            "activation_layout": "NHWC",
            "weight_layout": "NHWC (see train4_layer_params.json for per-layer details)",
        }

        if node.op_type == "Conv":
            layer_info["layer_name"] = f"conv_{conv_idx}"
            conv_idx += 1

            for attr in node.attribute:
                if attr.name == "kernel_shape":
                    layer_info["kernel_shape"] = list(attr.ints)
                elif attr.name == "strides":
                    layer_info["strides"] = list(attr.ints)
                elif attr.name == "pads":
                    layer_info["pads"] = list(attr.ints)
                elif attr.name == "group":
                    layer_info["group"] = attr.i
        else:
            layer_info["layer_name"] = f"matmul_{matmul_idx}"
            matmul_idx += 1

        input_quant = _trace_quant_params(node.input[0], node_output_to_node, init_map, "input")
        weight_quant = _trace_quant_params(node.input[1], node_output_to_node, init_map, "weight")

        bias_quant = None
        if node.op_type == "Conv" and len(node.input) > 2 and node.input[2]:
            bias_quant = _trace_quant_params(node.input[2], node_output_to_node, init_map, "bias")

        output_quant = _trace_output_quant(node.output[0], model.graph.node, init_map)

        is_quantized = (weight_quant is not None and weight_quant.get("scale") is not None)
        layer_info["quantized"] = is_quantized

        if is_quantized:
            w_scale = weight_quant.get("scale")
            is_per_channel = isinstance(w_scale, list) and len(w_scale) > 1
            layer_info["quant_granularity"] = {
                "weight": "per_channel" if is_per_channel else "per_tensor",
                "activation": "per_tensor",
            }
        else:
            layer_info["quant_granularity"] = None

        layer_info["input"] = input_quant if input_quant else {"dtype": "float32", "scale": None, "zero_point": None}
        layer_info["weight"] = weight_quant if weight_quant else {"dtype": "float32", "scale": None, "zero_point": None}
        if bias_quant:
            layer_info["bias"] = bias_quant
        elif node.op_type == "Conv" and len(node.input) > 2:
            layer_info["bias"] = {"dtype": "float32" if not is_quantized else "int32", "scale": None}
        layer_info["output"] = output_quant if output_quant else {"dtype": "float32", "scale": None, "zero_point": None}

        if is_quantized and output_quant and output_quant.get("scale") and input_quant and input_quant.get("scale"):
            w_scale = weight_quant["scale"]
            i_scale = input_quant["scale"]
            o_scale = output_quant["scale"]
            if isinstance(w_scale, list):
                layer_info["output"]["requant_scale"] = [
                    float((i_scale * ws) / o_scale) for ws in w_scale
                ]
            else:
                layer_info["output"]["requant_scale"] = float((i_scale * w_scale) / o_scale)

        backend = "gemmini" if node.op_type in GEMMINI_OPS else "cpu"
        op_class = "standard_conv"
        if node.op_type == "Conv":
            group = layer_info.get("group", 1)
            ks = layer_info.get("kernel_shape", [3, 3])
            if group > 1 and ks == [1, 1]:
                op_class = "pointwise_conv"
            elif group > 1:
                op_class = "depthwise_conv"
            elif ks == [1, 1]:
                op_class = "pointwise_conv"
        elif node.op_type == "MatMul":
            op_class = "matmul"

        layer_info["gemmini_mapping"] = {
            "expected_backend": backend,
            "operator_class": op_class,
        }

        layers.append(layer_info)

    return layers


def _trace_quant_params(tensor_name, node_map, init_map, role):
    """追溯 DequantizeLinear 节点获取量化参数。"""
    if tensor_name not in node_map:
        return None

    node = node_map[tensor_name]
    if node.op_type != "DequantizeLinear":
        return None

    result = {"dtype": "int8"}

    if len(node.input) > 1 and node.input[1] in init_map:
        scale_arr = init_map[node.input[1]]
        if scale_arr.size == 1:
            result["scale"] = float(scale_arr.flat[0])
        else:
            result["scale"] = [float(s) for s in scale_arr.flat]
    else:
        result["scale"] = None

    if len(node.input) > 2 and node.input[2] in init_map:
        zp_arr = init_map[node.input[2]]
        if zp_arr.size == 1:
            result["zero_point"] = int(zp_arr.flat[0])
        else:
            result["zero_point"] = [int(z) for z in zp_arr.flat]
    else:
        result["zero_point"] = 0

    if role == "bias":
        result["dtype"] = "int32"

    return result


def _trace_output_quant(tensor_name, all_nodes, init_map):
    """追溯 Conv/MatMul 输出端的 QuantizeLinear 获取输出量化参数。"""
    for node in all_nodes:
        if node.op_type == "QuantizeLinear" and tensor_name in node.input:
            result = {"dtype": "int8"}
            if len(node.input) > 1 and node.input[1] in init_map:
                scale_arr = init_map[node.input[1]]
                result["scale"] = float(scale_arr.flat[0])
            else:
                result["scale"] = None
            if len(node.input) > 2 and node.input[2] in init_map:
                zp_arr = init_map[node.input[2]]
                result["zero_point"] = int(zp_arr.flat[0])
            else:
                result["zero_point"] = 0
            return result
    return None


def export_json(layers, output_path):
    """导出量化参数 JSON。"""
    quantized_count = sum(1 for l in layers if l.get("quantized"))
    fp32_count = len(layers) - quantized_count

    data = {
        "model_name": "yolov11n",
        "dataset": "BDD100K",
        "quant_method": "PTQ",
        "quant_tool": "ONNX Runtime 1.15.1",
        "quant_format": "QDQ (QuantizeLinear/DequantizeLinear)",
        "calibration": {
            "method": "MinMax",
            "num_images": 500,
            "source": "BDD100K train subset",
            "symmetric_weight": True,
            "symmetric_activation": True,
        },
        "input_size": [640, 640],
        "weight_type": "int8",
        "activation_type": "int8",
        "per_channel_weight": True,
        "detect_head_excluded": True,
        "detect_head_note": "model.23 检测头保持 FP32，避免分类通道量化归零",
        "num_total_layers": len(layers),
        "num_quantized_layers": quantized_count,
        "num_fp32_layers": fp32_count,
        "layers": layers,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[导出] 量化参数 JSON: {output_path}")


def export_csv(layers, output_path):
    """导出量化参数 CSV（便于快速查看）。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "layer_name", "layer_type", "quantized", "kernel_shape",
            "weight_granularity",
            "input_scale", "input_zp",
            "weight_scale_count", "weight_scale_min", "weight_scale_max",
            "weight_zp",
            "output_scale", "output_zp",
            "backend", "op_class",
        ])
        for layer in layers:
            inp = layer.get("input", {})
            w = layer.get("weight", {})
            out = layer.get("output", {})
            gm = layer.get("gemmini_mapping", {})
            qg = layer.get("quant_granularity")

            inp_scale = inp.get("scale", "")
            inp_zp = inp.get("zero_point", "")

            w_scale = w.get("scale")
            if isinstance(w_scale, list):
                w_count = len(w_scale)
                w_min = f"{min(w_scale):.8f}"
                w_max = f"{max(w_scale):.8f}"
            elif w_scale is not None:
                w_count = 1
                w_min = f"{w_scale:.8f}"
                w_max = w_min
            else:
                w_count = ""
                w_min = ""
                w_max = ""

            w_zp = w.get("zero_point", "")
            if isinstance(w_zp, list):
                w_zp = w_zp[0] if all(z == w_zp[0] for z in w_zp) else "mixed"

            out_scale = out.get("scale", "")
            out_zp = out.get("zero_point", "")

            w_gran = ""
            if qg:
                w_gran = qg.get("weight", "")

            writer.writerow([
                layer["layer_name"],
                layer["layer_type"],
                "Y" if layer.get("quantized") else "N",
                str(layer.get("kernel_shape", "")),
                w_gran,
                inp_scale, inp_zp,
                w_count, w_min, w_max,
                w_zp,
                out_scale, out_zp,
                gm.get("expected_backend", ""),
                gm.get("operator_class", ""),
            ])
    print(f"[导出] 量化参数 CSV: {output_path}")


def main():
    if not os.path.exists(INT8_ONNX):
        print(f"[错误] 未找到 INT8 ONNX: {INT8_ONNX}")
        sys.exit(1)

    print(f"[加载] INT8 ONNX: {INT8_ONNX}")
    model = onnx.load(INT8_ONNX)
    init_map = get_initializer_map(model)

    print("[提取] 量化参数...")
    layers = extract_quant_params(model, init_map)

    print(f"[信息] 共提取 {len(layers)} 个计算层")
    conv_count = sum(1 for l in layers if l["layer_type"] == "Conv")
    matmul_count = sum(1 for l in layers if l["layer_type"] == "MatMul")
    quantized_count = sum(1 for l in layers if l.get("quantized"))
    fp32_count = len(layers) - quantized_count
    print(f"  Conv: {conv_count}, MatMul: {matmul_count}")
    print(f"  已量化: {quantized_count}, FP32: {fp32_count}")

    per_ch_count = sum(1 for l in layers if l.get("quant_granularity")
                       and l["quant_granularity"].get("weight") == "per_channel")
    print(f"  Per-channel 权重: {per_ch_count}")

    gemmini_count = sum(1 for l in layers if l["gemmini_mapping"]["expected_backend"] == "gemmini")
    cpu_count = len(layers) - gemmini_count
    print(f"  Gemmini: {gemmini_count}, CPU: {cpu_count}")

    export_json(layers, QUANT_JSON)
    export_csv(layers, QUANT_CSV)

    print(f"\n[完成] 量化参数表导出成功")
    print(f"  JSON: {QUANT_JSON}")
    print(f"  CSV:  {QUANT_CSV}")


if __name__ == "__main__":
    main()
