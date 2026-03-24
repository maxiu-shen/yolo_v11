"""
YOLOv11n INT8 验证脚本

通过 Ultralytics YOLO 框架验证 INT8 ONNX 模型在 BDD100K val 上的精度，
并与 FP32 基线进行对比。

用法:
    conda activate yolo_v11
    python tools/shen_int8_val.py
"""

import os
import sys
import time

from ultralytics import YOLO


INT8_ONNX = "exports/onnx/train4_best_int8.onnx"
FP32_ONNX = "exports/onnx/train4_best_fp32.onnx"
DATA_YAML = "ultralytics/cfg/datasets/bdd100k.yaml"
PROJECT = "runs/detect"
IMG_SIZE = 640

FP32_BASELINE = {
    "Precision": 0.662,
    "Recall": 0.410,
    "mAP50": 0.449,
    "mAP50-95": 0.253,
}


def validate_model(model_path, name, device="cpu"):
    """运行验证并返回指标。"""
    print(f"\n{'='*60}")
    print(f"开始验证: {model_path}")
    print(f"设备: {device}")
    print(f"{'='*60}")

    model = YOLO(model_path, task="detect")

    t0 = time.time()
    results = model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        device=device,
        plots=True,
        project=PROJECT,
        name=name,
    )
    elapsed = time.time() - t0

    metrics = {
        "Precision": float(results.box.mp),
        "Recall": float(results.box.mr),
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
    }

    per_class = []
    names = results.names
    for i in range(len(names)):
        per_class.append({
            "class": names[i],
            "precision": float(results.box.p[i]),
            "recall": float(results.box.r[i]),
            "mAP50": float(results.box.ap50[i]),
            "mAP50-95": float(results.box.ap[i]),
        })

    return metrics, per_class, elapsed


def print_comparison(fp32, int8):
    """打印 FP32 vs INT8 对比。"""
    print(f"\n{'='*60}")
    print("FP32 vs INT8 精度对比")
    print(f"{'='*60}")
    print(f"{'指标':<15} {'FP32':>10} {'INT8':>10} {'差值':>10} {'下降%':>10}")
    print("-" * 55)
    for k in fp32:
        diff = int8[k] - fp32[k]
        pct = (diff / fp32[k] * 100) if fp32[k] != 0 else 0
        print(f"{k:<15} {fp32[k]:>10.4f} {int8[k]:>10.4f} {diff:>+10.4f} {pct:>+10.2f}%")


def main():
    if not os.path.exists(INT8_ONNX):
        print(f"[错误] 未找到 INT8 ONNX: {INT8_ONNX}")
        sys.exit(1)

    int8_metrics, int8_per_class, int8_elapsed = validate_model(
        INT8_ONNX, "train4_int8_val", device=0
    )

    print(f"\n[INT8 验证耗时] {int8_elapsed:.1f}s")
    print(f"\n[INT8 整体指标]")
    for k, v in int8_metrics.items():
        print(f"  {k}: {v:.4f}")

    print_comparison(FP32_BASELINE, int8_metrics)

    print(f"\n[INT8 分类别指标]")
    print(f"{'类别':<15} {'P':>8} {'R':>8} {'mAP50':>8} {'mAP50-95':>10}")
    print("-" * 55)
    for c in int8_per_class:
        print(f"{c['class']:<15} {c['precision']:>8.3f} {c['recall']:>8.3f} "
              f"{c['mAP50']:>8.3f} {c['mAP50-95']:>10.3f}")

    report_path = "exports/reports/train4_int8_val_report.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Train4 INT8 验证报告\n\n")
        f.write("## 基本信息\n")
        f.write(f"- 模型: `{INT8_ONNX}`\n")
        f.write(f"- 数据集: `{DATA_YAML}`\n")
        f.write(f"- 输入尺寸: `{IMG_SIZE}`\n")
        f.write(f"- 量化方法: `PTQ (ONNX Runtime, QDQ, MinMax, symmetric)`\n")
        f.write(f"- 校准图片: `200 张 BDD100K train`\n")
        f.write(f"- 验证设备: `GPU CUDA (ONNX Runtime CUDAExecutionProvider)`\n")
        f.write(f"- 验证耗时: `{int8_elapsed:.1f}s`\n\n")

        f.write("## INT8 整体指标\n")
        for k, v in int8_metrics.items():
            f.write(f"- {k}: `{v:.4f}`\n")
        f.write("\n")

        f.write("## FP32 vs INT8 对比\n")
        f.write("| 指标 | FP32 | INT8 | 差值 | 下降% |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for k in FP32_BASELINE:
            diff = int8_metrics[k] - FP32_BASELINE[k]
            pct = (diff / FP32_BASELINE[k] * 100) if FP32_BASELINE[k] != 0 else 0
            f.write(f"| {k} | {FP32_BASELINE[k]:.4f} | {int8_metrics[k]:.4f} | {diff:+.4f} | {pct:+.2f}% |\n")
        f.write("\n")

        f.write("## INT8 分类别指标\n")
        f.write("| 类别 | Precision | Recall | mAP50 | mAP50-95 |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for c in int8_per_class:
            f.write(f"| {c['class']} | {c['precision']:.3f} | {c['recall']:.3f} | "
                    f"{c['mAP50']:.3f} | {c['mAP50-95']:.3f} |\n")
        f.write("\n")

        f.write("## 结论\n")
        map_drop = int8_metrics["mAP50-95"] - FP32_BASELINE["mAP50-95"]
        map50_drop = int8_metrics["mAP50"] - FP32_BASELINE["mAP50"]
        f.write(f"- mAP50 变化: `{map50_drop:+.4f}` ({map50_drop/FP32_BASELINE['mAP50']*100:+.2f}%)\n")
        f.write(f"- mAP50-95 变化: `{map_drop:+.4f}` ({map_drop/FP32_BASELINE['mAP50-95']*100:+.2f}%)\n")
        if abs(map_drop) < 0.02:
            f.write("- 评估: INT8 量化精度损失极小，可接受\n")
        elif abs(map_drop) < 0.05:
            f.write("- 评估: INT8 量化有一定精度损失，在可接受范围内\n")
        else:
            f.write("- 评估: INT8 量化精度损失较大，可能需要调整量化策略\n")

    print(f"\n[报告] 已保存到 {report_path}")
    print("[完成] INT8 验证结束")


if __name__ == "__main__":
    main()
