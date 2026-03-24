"""
分析 FP32 ONNX 模型的输出结构，找到 bbox/cls 合并的 Concat 节点
以及分类分支的关键节点，用于确定量化排除列表。
"""
import onnx

FP32_ONNX = "exports/onnx/train4_best_fp32.onnx"

model = onnx.load(FP32_ONNX)
graph = model.graph

output_name = graph.output[0].name
print(f"模型输出名: {output_name}")

name_to_node = {}
for node in graph.node:
    for out in node.output:
        name_to_node[out] = node

def trace_back(tensor_name, depth=0, max_depth=8):
    if depth > max_depth or tensor_name not in name_to_node:
        return
    node = name_to_node[tensor_name]
    prefix = "  " * depth
    print(f"{prefix}← [{node.op_type}] name={node.name}")
    print(f"{prefix}   inputs: {list(node.input)}")
    print(f"{prefix}   outputs: {list(node.output)}")
    for inp in node.input:
        if inp in name_to_node:
            trace_back(inp, depth + 1, max_depth)

print(f"\n=== 从输出 '{output_name}' 反向追踪 ===")
trace_back(output_name, max_depth=6)

print(f"\n\n=== 所有 Concat 节点 ===")
for node in graph.node:
    if node.op_type == "Concat":
        print(f"  {node.name}: inputs={list(node.input)}, outputs={list(node.output)}")

print(f"\n\n=== 最后 20 个节点 ===")
for node in graph.node[-20:]:
    print(f"  [{node.op_type}] {node.name}")
    print(f"    in: {list(node.input)}")
    print(f"    out: {list(node.output)}")
