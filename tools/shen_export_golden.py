"""
为 Gemmini 侧逐层验证生成 NHWC INT8 / FP32 golden reference (v2)

覆盖范围：
  Stage 0 前处理：LetterBox + BGR→RGB + INT8 量化
  Stage 1-2 Backbone：conv_0 ~ conv_39 PRE_ACT / POST_ACT (INT8)
  Stage 3 Neck：conv_43 / conv_47 / conv_60 / conv_78 + 模块输出别名
  Stage 4 检测头：P3/P4/P5 cv2 (bbox) / cv3 (cls) 输出 (FP32)
  Stage 5-6 后处理：DFL ltrb / Sigmoid cls / 最终检测结果 (FP32)

用法:
    conda activate yolo_v11
    python tools/shen_export_golden.py                          # 导出全部
    python tools/shen_export_golden.py --stage backbone         # 仅 backbone
    python tools/shen_export_golden.py --stage neck             # 仅 neck
    python tools/shen_export_golden.py --stage head             # 仅检测头
    python tools/shen_export_golden.py --stage postprocess      # 仅后处理
    python tools/shen_export_golden.py --convs 20,30,32,39      # 指定 conv 层
    python tools/shen_export_golden.py --priority p0            # 仅 P0 项

量化规则：
    PRE_ACT  = clip(round(fp32_bn_output / output_scale), -128, 127)
    POST_ACT = clip(round(fp32_silu_output / next_input_scale), -128, 127)
    检测头与后处理直接导出 FP32

对齐要求:
    - 前处理使用标准 Ultralytics LetterBox 默认参数
    - 与 Gemmini C 侧 shen_preprocess_to_int8_nhwc / shen_letterbox_bilinear 对齐
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ==================== 路径配置 ====================

TEST_IMAGE = "test_picture/b1c9c847-3bda4659.jpg"
TEST_IMAGE_FALLBACK = "datasets/BDD100K/images/val/b1c9c847-3bda4659.jpg"
BEST_PT = "runs/detect/train4/weights/best.pt"
BEST_PT_FALLBACK = "runs/detect/train2/weights/best.pt"
QUANT_JSON = "exports/quant/train4_quant_params.json"
OUTPUT_DIR = "exports/golden"

INPUT_SCALE = 7.8740157187e-03  # ≈ 1/127, conv_0 input_scale
EXPECTED_IMG_MD5 = "8424d960b86559e1d1e872ed2e1fe314"


# ==================== 层配置 ====================
# (conv_id, module_getter, description, category, priority)
# module_getter 接收 DetectionModel (含 .model 属性的 nn.Sequential)
CONV_GOLDEN_CONFIG = [
    # --- Backbone Stage 0-1 (已有) ---
    (0,  lambda m: m.model[0],         "stem conv_0",                  "backbone", "p0"),
    (1,  lambda m: m.model[1],         "stem conv_1",                  "backbone", "p0"),
    (2,  lambda m: m.model[2].cv1,     "C3k2 model.2 cv1",            "backbone", "p0"),
    (5,  lambda m: m.model[2].cv2,     "C3k2 model.2 cv2",            "backbone", "p0"),
    (10, lambda m: m.model[4].cv2,     "C3k2 model.4 cv2 = P3",       "backbone", "p0"),
    # --- Backbone Stage 2 (P0 新增) ---
    (20, lambda m: m.model[6].cv2,     "C3k2 model.6 cv2 = P4",       "backbone", "p0"),
    (30, lambda m: m.model[8].cv2,     "C3k2 model.8 cv2",            "backbone", "p0"),
    (32, lambda m: m.model[9].cv2,     "SPPF model.9 cv2",            "backbone", "p0"),
    (39, lambda m: m.model[10].cv2,    "C2PSA model.10 cv2 = P5",     "backbone", "p0"),
    # --- Backbone Stage 2 (P1 中间检查点) ---
    (21, lambda m: m.model[7],         "downsample → model.8",         "backbone", "p1"),
    (31, lambda m: m.model[9].cv1,     "SPPF model.9 cv1",            "backbone", "p1"),
    # --- Neck Stage 3 (P0) ---
    (43, lambda m: m.model[13].cv2,    "Neck model.13 cv2",            "neck",     "p0"),
    (47, lambda m: m.model[16].cv2,    "Neck model.16 cv2 = Neck P3",  "neck",     "p0"),
    (60, lambda m: m.model[19].cv2,    "Neck model.19 cv2 = Neck P4",  "neck",     "p0"),
    (78, lambda m: m.model[22].cv2,    "Neck model.22 cv2 = Neck P5",  "neck",     "p0"),
]

# POST_ACT 量化使用的 next_conv input_scale 映射
# conv_id → 下一层 conv_id（用其 input_scale 量化 SiLU 输出）
# None 表示无直接后续或多路径，POST_ACT 使用 output_scale 近似
POST_ACT_NEXT_CONV: dict[int, int | None] = {
    0: 1, 1: 2, 2: 3, 5: 6, 10: 11,
    20: 21, 21: 22, 30: 31, 32: 33, 39: 40,
    43: 44, 47: 48,
    31: None,  # → MaxPool 链, scale 不变
    60: None,  # → model.20 downsample + detect P4, 多路径
    78: None,  # → detect P5 (FP32), 无 INT8 后续
}

# Neck 模块输出别名：(别名, 对应 conv_id, 模块描述)
# POST_ACT of cv2 = module output
NECK_ALIASES = [
    ("neck_model13", 43, "Neck model.13 C3k2 output = conv_43 POST_ACT"),
    ("neck_p3",      47, "Neck model.16 C3k2 output = conv_47 POST_ACT = Neck P3"),
    ("neck_p4",      60, "Neck model.19 C3k2 output = conv_60 POST_ACT = Neck P4"),
    ("neck_p5",      78, "Neck model.22 C3k2 output = conv_78 POST_ACT = Neck P5"),
]

# 检测头 FP32 输出：(名称, module_getter, 描述, priority)
DETECT_GOLDEN_CONFIG = [
    ("det_p3_cv2", lambda m: m.model[23].cv2[0], "P3 bbox branch",  "p0"),
    ("det_p3_cv3", lambda m: m.model[23].cv3[0], "P3 cls branch",   "p0"),
    ("det_p4_cv2", lambda m: m.model[23].cv2[1], "P4 bbox branch",  "p1"),
    ("det_p4_cv3", lambda m: m.model[23].cv3[1], "P4 cls branch",   "p1"),
    ("det_p5_cv2", lambda m: m.model[23].cv2[2], "P5 bbox branch",  "p1"),
    ("det_p5_cv3", lambda m: m.model[23].cv3[2], "P5 cls branch",   "p1"),
]


# ==================== 工具函数 ====================

def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def find_test_image() -> tuple[str, str]:
    for p in (TEST_IMAGE, TEST_IMAGE_FALLBACK):
        full = REPO_ROOT / p
        if full.exists():
            return str(full), file_md5(str(full))
    raise FileNotFoundError(
        "未找到测试图片，请将 b1c9c847-3bda4659.jpg 放入 test_picture/ 或 datasets/BDD100K/images/val/"
    )


def find_best_pt(override: str | None = None) -> str:
    if override:
        p = Path(override) if Path(override).is_absolute() else REPO_ROOT / override
        if p.exists():
            return str(p)
        raise FileNotFoundError(f"指定的模型路径不存在: {override}")
    for p in (BEST_PT, BEST_PT_FALLBACK):
        full = REPO_ROOT / p
        if full.exists():
            return str(full)
    raise FileNotFoundError("未找到 best.pt，请先完成 train4 或 train2 训练")


def load_quant_params(json_path: str | None = None) -> tuple[dict[str, float], dict[str, float]]:
    """加载量化参数，返回 (output_scales, input_scales) 两个字典"""
    p = REPO_ROOT / (json_path or QUANT_JSON)
    if not p.exists():
        raise FileNotFoundError(f"未找到量化参数: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_scales: dict[str, float] = {}
    input_scales: dict[str, float] = {}
    for layer in data.get("layers", []):
        name = layer.get("layer_name")
        if not name or not name.startswith("conv_"):
            continue
        o = layer.get("output", {})
        if isinstance(o.get("scale"), (int, float)):
            output_scales[name] = float(o["scale"])
        i = layer.get("input", {})
        if isinstance(i.get("scale"), (int, float)):
            input_scales[name] = float(i["scale"])
    return output_scales, input_scales


def int8_array_to_c(arr: np.ndarray) -> str:
    """将 int8 数组转为 C 数组字符串，每行 16 个"""
    flat = arr.flatten()
    lines = []
    for i in range(0, len(flat), 16):
        chunk = flat[i : i + 16]
        lines.append("    " + ", ".join(str(int(x)) for x in chunk) + ",")
    return "\n".join(lines)


def fp32_array_to_c(arr: np.ndarray) -> str:
    """将 float32 数组转为 C 数组字符串，每行 8 个"""
    flat = arr.flatten()
    lines = []
    for i in range(0, len(flat), 8):
        chunk = flat[i : i + 8]
        lines.append("    " + ", ".join(f"{float(x):.8e}f" for x in chunk) + ",")
    return "\n".join(lines)


def write_header(path: Path, guard: str, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"// Auto-generated by shen_export_golden.py v2\n")
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(content)
        f.write(f"\n\n#endif /* {guard} */\n")


def write_bin(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.flatten().tofile(str(path))


def quantize_int8(fp32: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(
        np.round(fp32.astype(np.float64) / scale),
        -128, 127,
    ).astype(np.int8)


# ==================== 前处理 golden ====================

def export_preprocess_golden(img_path: str, out_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    from ultralytics.data.augment import LetterBox

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise RuntimeError(f"无法读取图片: {img_path}")

    letterbox = LetterBox(
        new_shape=(640, 640), auto=False, scaleup=True, center=True, stride=32,
    )
    img_letterboxed = letterbox(image=img_bgr.copy())
    assert img_letterboxed.shape == (640, 640, 3), f"letterbox shape {img_letterboxed.shape}"

    img_rgb = img_letterboxed[:, :, ::-1].copy()
    img_fp32 = img_rgb.astype(np.float32) / 255.0
    img_int8 = quantize_int8(img_fp32, INPUT_SCALE)

    # letterbox BGR .h
    content = "// letterbox BGR uint8, NHWC [640][640][3]\n"
    content += f"static const uint8_t SHEN_GOLDEN_LETTERBOX_BGR[{640*640*3}] = {{\n"
    content += int8_array_to_c(img_letterboxed.astype(np.int8))
    content += "\n};"
    write_header(out_dir / "shen_golden_letterbox_bgr.h", "SHEN_GOLDEN_LETTERBOX_BGR_H", content)

    # preprocess INT8 RGB .h
    content = f"// INT8 NHWC RGB [640][640][3], scale={INPUT_SCALE}\n"
    content += f"static const int8_t SHEN_GOLDEN_INPUT[{640*640*3}] = {{\n"
    content += int8_array_to_c(img_int8)
    content += "\n};"
    write_header(out_dir / "shen_golden_preprocess_input.h", "SHEN_GOLDEN_PREPROCESS_INPUT_H", content)
    write_bin(out_dir / "shen_golden_preprocess_input.bin", img_int8)

    stats = {
        "letterbox_bgr_shape": list(img_letterboxed.shape),
        "letterbox_bgr_min": int(img_letterboxed.min()),
        "letterbox_bgr_max": int(img_letterboxed.max()),
        "letterbox_bgr_mean": float(img_letterboxed.mean()),
        "preprocess_int8_min": int(img_int8.min()),
        "preprocess_int8_max": int(img_int8.max()),
        "preprocess_int8_mean": float(img_int8.mean()),
    }
    print(f"  [preprocess] letterbox_bgr.h, preprocess_input.h/.bin")
    return img_letterboxed, img_int8, stats


# ==================== Conv 层 golden（单次前向） ====================

class ConvHookCapture:
    """在单次前向传播中捕获多个 Conv 层的 BN 输出 (PRE_ACT) 和模块输出 (POST_ACT)"""

    def __init__(self):
        self.pre_act: dict[int, torch.Tensor] = {}
        self.post_act: dict[int, torch.Tensor] = {}
        self._hooks: list = []

    def register(self, conv_id: int, target_module):
        if not hasattr(target_module, "bn"):
            raise ValueError(f"conv_{conv_id}: 模块无 bn 属性，不是标准 Conv 模块")

        def _bn_hook(_mod, _inp, out, cid=conv_id):
            self.pre_act[cid] = out.detach()

        def _conv_hook(_mod, _inp, out, cid=conv_id):
            self.post_act[cid] = out.detach()

        self._hooks.append(target_module.bn.register_forward_hook(_bn_hook))
        self._hooks.append(target_module.register_forward_hook(_conv_hook))

    def cleanup(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class DetectHookCapture:
    """捕获检测头 Sequential 模块的 FP32 输出"""

    def __init__(self):
        self.outputs: dict[str, torch.Tensor] = {}
        self._hooks: list = []

    def register(self, name: str, target_module):
        def _hook(_mod, _inp, out, n=name):
            self.outputs[n] = out.detach()
        self._hooks.append(target_module.register_forward_hook(_hook))

    def cleanup(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def nchw_to_nhwc_np(tensor: torch.Tensor) -> np.ndarray:
    """NCHW tensor → NHWC numpy, squeeze batch dim"""
    return tensor.permute(0, 2, 3, 1).contiguous().squeeze(0).cpu().numpy()


def export_conv_golden(
    conv_id: int,
    pre_act_fp32: np.ndarray,
    post_act_fp32: np.ndarray,
    output_scale: float,
    post_act_scale: float | None,
    desc: str,
    out_dir: Path,
) -> dict:
    """导出单个 Conv 层的 PRE_ACT / POST_ACT golden"""
    h, w, c = pre_act_fp32.shape

    pre_int8 = quantize_int8(pre_act_fp32, output_scale)

    pa_scale = post_act_scale if post_act_scale is not None else output_scale
    pa_scale_note = f"next_input_scale={pa_scale}" if post_act_scale else f"output_scale={pa_scale} (approx)"
    post_int8 = quantize_int8(post_act_fp32, pa_scale)

    # .h 文件
    guard = f"SHEN_GOLDEN_CONV{conv_id}_H"
    body = f"// {desc}\n"
    body += f"// conv_{conv_id} PRE_ACT (BN output), NHWC [{h}][{w}][{c}], scale={output_scale}\n"
    body += f"#define SHEN_GOLDEN_CONV{conv_id}_H_DIM {h}\n"
    body += f"#define SHEN_GOLDEN_CONV{conv_id}_W_DIM {w}\n"
    body += f"#define SHEN_GOLDEN_CONV{conv_id}_C_DIM {c}\n\n"
    body += f"static const int8_t SHEN_GOLDEN_CONV{conv_id}_PRE_ACT[{h*w*c}] = {{\n"
    body += int8_array_to_c(pre_int8)
    body += "\n};\n\n"
    body += f"// conv_{conv_id} POST_ACT (SiLU output), NHWC [{h}][{w}][{c}], {pa_scale_note}\n"
    body += f"static const int8_t SHEN_GOLDEN_CONV{conv_id}_POST_ACT[{h*w*c}] = {{\n"
    body += int8_array_to_c(post_int8)
    body += "\n};"

    write_header(out_dir / f"shen_golden_conv{conv_id}.h", guard, body)
    write_bin(out_dir / f"shen_golden_conv{conv_id}_pre_act.bin", pre_int8)
    write_bin(out_dir / f"shen_golden_conv{conv_id}_post_act.bin", post_int8)

    return {
        "description": desc,
        "shape": [h, w, c],
        "output_scale": output_scale,
        "post_act_scale": pa_scale,
        "post_act_scale_exact": post_act_scale is not None,
        "pre_act_fp32_min": float(pre_act_fp32.min()),
        "pre_act_fp32_max": float(pre_act_fp32.max()),
        "pre_act_fp32_mean": float(pre_act_fp32.mean()),
        "pre_act_int8_min": int(pre_int8.min()),
        "pre_act_int8_max": int(pre_int8.max()),
        "pre_act_int8_mean": float(pre_int8.mean()),
        "post_act_fp32_min": float(post_act_fp32.min()),
        "post_act_fp32_max": float(post_act_fp32.max()),
        "post_act_fp32_mean": float(post_act_fp32.mean()),
        "post_act_int8_min": int(post_int8.min()),
        "post_act_int8_max": int(post_int8.max()),
        "post_act_int8_mean": float(post_int8.mean()),
    }


def export_neck_alias(alias: str, conv_id: int, desc: str, out_dir: Path):
    """生成 Neck 模块输出的别名头文件，引用 conv POST_ACT"""
    guard = f"SHEN_GOLDEN_{alias.upper()}_H"
    alias_upper = alias.upper()
    body = f"// {desc}\n"
    body += f'#include "shen_golden_conv{conv_id}.h"\n\n'
    body += f"#define SHEN_GOLDEN_{alias_upper}          SHEN_GOLDEN_CONV{conv_id}_POST_ACT\n"
    body += f"#define SHEN_GOLDEN_{alias_upper}_H_DIM    SHEN_GOLDEN_CONV{conv_id}_H_DIM\n"
    body += f"#define SHEN_GOLDEN_{alias_upper}_W_DIM    SHEN_GOLDEN_CONV{conv_id}_W_DIM\n"
    body += f"#define SHEN_GOLDEN_{alias_upper}_C_DIM    SHEN_GOLDEN_CONV{conv_id}_C_DIM\n"
    write_header(out_dir / f"shen_golden_{alias}.h", guard, body)


# ==================== 检测头 FP32 golden ====================

def export_detect_golden(
    name: str,
    tensor_fp32: np.ndarray,
    desc: str,
    out_dir: Path,
) -> dict:
    """导出检测头 FP32 输出（bbox / cls 分支）"""
    h, w, c = tensor_fp32.shape
    name_upper = name.upper()

    guard = f"SHEN_GOLDEN_{name_upper}_H"
    body = f"// {desc}, FP32 NHWC [{h}][{w}][{c}]\n"
    body += f"#define SHEN_GOLDEN_{name_upper}_H_DIM {h}\n"
    body += f"#define SHEN_GOLDEN_{name_upper}_W_DIM {w}\n"
    body += f"#define SHEN_GOLDEN_{name_upper}_C_DIM {c}\n\n"
    body += f"static const float SHEN_GOLDEN_{name_upper}[{h*w*c}] = {{\n"
    body += fp32_array_to_c(tensor_fp32.astype(np.float32))
    body += "\n};"

    write_header(out_dir / f"shen_golden_{name}.h", guard, body)
    write_bin(out_dir / f"shen_golden_{name}.bin", tensor_fp32.astype(np.float32))

    return {
        "description": desc,
        "shape": [h, w, c],
        "dtype": "float32",
        "fp32_min": float(tensor_fp32.min()),
        "fp32_max": float(tensor_fp32.max()),
        "fp32_mean": float(tensor_fp32.mean()),
    }


# ==================== 后处理 golden ====================

def export_postprocess_golden(
    model_output: torch.Tensor,
    det_capture: DetectHookCapture,
    detect_head,
    out_dir: Path,
) -> dict:
    """
    导出后处理 golden：
      - DFL 解码后 ltrb [8400][4] FP32
      - Sigmoid(cls) [8400][10] FP32
      - 最终检测结果 [N_det][6] FP32
    """
    stats = {}

    # model_output 在 eval 模式下返回 (y, x)
    # y: [1, 14, 8400] = [bbox(4) + cls(10)]
    # 或者直接就是 y tensor
    if isinstance(model_output, (tuple, list)):
        y = model_output[0]
    else:
        y = model_output

    if y.dim() == 3 and y.shape[1] == 14:
        y_np = y.squeeze(0).cpu().numpy()  # [14, 8400]

        # DFL 解码后的 bbox: 前 4 通道 = decoded xyxy
        dfl_ltrb = y_np[:4, :].T.astype(np.float32)  # [8400, 4]
        name = "dfl_ltrb"
        guard = "SHEN_GOLDEN_DFL_LTRB_H"
        body = f"// DFL decoded bbox, FP32 [{dfl_ltrb.shape[0]}][{dfl_ltrb.shape[1]}]\n"
        body += f"#define SHEN_GOLDEN_DFL_LTRB_COUNT {dfl_ltrb.shape[0]}\n"
        body += f"#define SHEN_GOLDEN_DFL_LTRB_DIM 4\n\n"
        body += f"static const float SHEN_GOLDEN_DFL_LTRB[{dfl_ltrb.size}] = {{\n"
        body += fp32_array_to_c(dfl_ltrb)
        body += "\n};"
        write_header(out_dir / f"shen_golden_{name}.h", guard, body)
        write_bin(out_dir / f"shen_golden_{name}.bin", dfl_ltrb)
        stats["dfl_ltrb"] = {
            "shape": list(dfl_ltrb.shape), "dtype": "float32",
            "min": float(dfl_ltrb.min()), "max": float(dfl_ltrb.max()),
            "mean": float(dfl_ltrb.mean()),
        }
        print(f"  [postprocess] dfl_ltrb: shape={dfl_ltrb.shape}")

        # Sigmoid(cls): 后 10 通道（model output 已经 sigmoid 过了）
        sigmoid_cls = y_np[4:, :].T.astype(np.float32)  # [8400, 10]
        name = "sigmoid_cls"
        guard = "SHEN_GOLDEN_SIGMOID_CLS_H"
        body = f"// Sigmoid cls scores, FP32 [{sigmoid_cls.shape[0]}][{sigmoid_cls.shape[1]}]\n"
        body += f"#define SHEN_GOLDEN_SIGMOID_CLS_COUNT {sigmoid_cls.shape[0]}\n"
        body += f"#define SHEN_GOLDEN_SIGMOID_CLS_DIM {sigmoid_cls.shape[1]}\n\n"
        body += f"static const float SHEN_GOLDEN_SIGMOID_CLS[{sigmoid_cls.size}] = {{\n"
        body += fp32_array_to_c(sigmoid_cls)
        body += "\n};"
        write_header(out_dir / f"shen_golden_{name}.h", guard, body)
        write_bin(out_dir / f"shen_golden_{name}.bin", sigmoid_cls)
        stats["sigmoid_cls"] = {
            "shape": list(sigmoid_cls.shape), "dtype": "float32",
            "min": float(sigmoid_cls.min()), "max": float(sigmoid_cls.max()),
            "mean": float(sigmoid_cls.mean()),
        }
        print(f"  [postprocess] sigmoid_cls: shape={sigmoid_cls.shape}")

        # NMS 最终检测结果
        _export_nms_results(dfl_ltrb, sigmoid_cls, out_dir, stats)

    return stats


def _export_nms_results(
    boxes_xyxy: np.ndarray,  # [8400, 4]
    cls_scores: np.ndarray,  # [8400, 10]
    out_dir: Path,
    stats: dict,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.7,
):
    """简易 NMS 导出最终检测结果"""
    max_scores = cls_scores.max(axis=1)  # [8400]
    max_classes = cls_scores.argmax(axis=1)  # [8400]

    mask = max_scores >= conf_thresh
    if mask.sum() == 0:
        print(f"  [postprocess] detection_results: 无检测结果 (conf>{conf_thresh})")
        stats["detection_results"] = {"count": 0}
        return

    sel_boxes = boxes_xyxy[mask]
    sel_scores = max_scores[mask]
    sel_classes = max_classes[mask]

    # 按类别做 NMS
    keep = []
    for cls_id in np.unique(sel_classes):
        cls_mask = sel_classes == cls_id
        cls_boxes = sel_boxes[cls_mask]
        cls_scores_c = sel_scores[cls_mask]
        cls_indices = np.where(cls_mask)[0]

        order = cls_scores_c.argsort()[::-1]
        suppressed = set()
        for i in range(len(order)):
            if order[i] in suppressed:
                continue
            keep.append(cls_indices[order[i]])
            ix = cls_boxes[order[i]]
            for j in range(i + 1, len(order)):
                if order[j] in suppressed:
                    continue
                jx = cls_boxes[order[j]]
                inter_x1 = max(ix[0], jx[0])
                inter_y1 = max(ix[1], jx[1])
                inter_x2 = min(ix[2], jx[2])
                inter_y2 = min(ix[3], jx[3])
                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                area_i = (ix[2] - ix[0]) * (ix[3] - ix[1])
                area_j = (jx[2] - jx[0]) * (jx[3] - jx[1])
                union = area_i + area_j - inter_area
                if union > 0 and inter_area / union > iou_thresh:
                    suppressed.add(order[j])

    keep = sorted(keep)
    results = np.zeros((len(keep), 6), dtype=np.float32)
    for idx, k in enumerate(keep):
        results[idx, :4] = sel_boxes[k]
        results[idx, 4] = sel_scores[k]
        results[idx, 5] = float(sel_classes[k])

    n_det = results.shape[0]
    name = "detection_results"
    guard = "SHEN_GOLDEN_DETECTION_RESULTS_H"
    body = f"// Final detection results after NMS, FP32 [{n_det}][6]\n"
    body += f"// Columns: x1, y1, x2, y2, confidence, class_id\n"
    body += f"// conf_thresh={conf_thresh}, iou_thresh={iou_thresh}\n"
    body += f"#define SHEN_GOLDEN_DET_COUNT {n_det}\n\n"
    body += f"static const float SHEN_GOLDEN_DETECTION_RESULTS[{n_det * 6}] = {{\n"
    body += fp32_array_to_c(results)
    body += "\n};"
    write_header(out_dir / f"shen_golden_{name}.h", guard, body)
    write_bin(out_dir / f"shen_golden_{name}.bin", results)

    stats["detection_results"] = {
        "count": n_det, "conf_thresh": conf_thresh, "iou_thresh": iou_thresh,
        "shape": list(results.shape), "dtype": "float32",
    }
    print(f"  [postprocess] detection_results: {n_det} detections")


# ==================== PSA 注意力 golden (P1) ====================

def export_psa_attention(model, input_tensor: torch.Tensor, out_dir: Path) -> dict | None:
    """导出 C2PSA model.10 中 PSABlock 的 Softmax 后注意力权重"""
    try:
        c2psa = model.model[10]
        psa_block = c2psa.m[0]  # 第一个 PSABlock
        attn_module = psa_block.attn  # Attention 子模块
    except (AttributeError, IndexError):
        print("  [psa] 未找到 C2PSA/PSABlock/Attention 模块，跳过")
        return None

    attn_output = []

    def _attn_hook(_mod, _inp, out):
        attn_output.append(out.detach())

    h = attn_module.register_forward_hook(_attn_hook)
    try:
        with torch.no_grad():
            _ = model(input_tensor)
    finally:
        h.remove()

    if not attn_output:
        print("  [psa] Attention hook 未捕获输出，跳过")
        return None

    attn_np = attn_output[0].squeeze(0).cpu().numpy().astype(np.float32)
    print(f"  [psa] attention output shape: {attn_np.shape}")

    name = "psa_attn"
    guard = "SHEN_GOLDEN_PSA_ATTN_H"
    shape_str = "][".join(str(s) for s in attn_np.shape)
    body = f"// C2PSA model.10 PSABlock[0] attention output, FP32 [{shape_str}]\n"
    for i, s in enumerate(attn_np.shape):
        body += f"#define SHEN_GOLDEN_PSA_ATTN_DIM{i} {s}\n"
    body += f"\nstatic const float SHEN_GOLDEN_PSA_ATTN[{attn_np.size}] = {{\n"
    body += fp32_array_to_c(attn_np)
    body += "\n};"

    write_header(out_dir / f"shen_golden_{name}.h", guard, body)
    write_bin(out_dir / f"shen_golden_{name}.bin", attn_np)

    return {
        "shape": list(attn_np.shape), "dtype": "float32",
        "min": float(attn_np.min()), "max": float(attn_np.max()),
        "mean": float(attn_np.mean()),
    }


# ==================== Summary 生成 ====================

def generate_summary(summary_data: dict, conv_configs: list, out_dir: Path):
    md = [
        "# Golden Reference 导出摘要 (v2)",
        "",
        f"导出脚本版本: v2",
        "",
        "## 元信息",
        f"- **模型权重**: `{summary_data['model_weights']}`",
        f"- **模型 MD5**: `{summary_data['model_md5']}`",
        f"- **测试图片**: `{summary_data['test_image']}`",
        f"- **图片 MD5**: `{summary_data['test_image_md5']}` (预期: `{EXPECTED_IMG_MD5}`)",
        "",
        "## LetterBox 参数",
        "```json",
        json.dumps(summary_data["letterbox_params"], indent=2, ensure_ascii=False),
        "```",
        "",
    ]

    if "preprocess" in summary_data:
        p = summary_data["preprocess"]
        md += [
            "## 前处理 golden",
            f"- input_scale: `{INPUT_SCALE}`",
            f"- letterbox_bgr: min={p['letterbox_bgr_min']}, max={p['letterbox_bgr_max']}",
            f"- preprocess_int8: min={p['preprocess_int8_min']}, max={p['preprocess_int8_max']}",
            "",
        ]

    if summary_data.get("conv_layers"):
        md += ["## Conv 层 golden (INT8)", "",
               "| conv | shape | output_scale | post_act_scale | pre_act range | post_act range | category |",
               "|------|-------|-------------|---------------|---------------|----------------|----------|"]
        for conv_id, stats in sorted(summary_data["conv_layers"].items(), key=lambda x: x[0]):
            s = stats
            pa_exact = "✓" if s.get("post_act_scale_exact") else "≈"
            md.append(
                f"| conv_{conv_id} | {s['shape']} | {s['output_scale']:.6e} "
                f"| {s['post_act_scale']:.6e} {pa_exact} "
                f"| [{s['pre_act_int8_min']}, {s['pre_act_int8_max']}] "
                f"| [{s['post_act_int8_min']}, {s['post_act_int8_max']}] "
                f"| {s.get('category', '')} |"
            )
        md.append("")

    if summary_data.get("detect_layers"):
        md += ["## 检测头 golden (FP32)", "",
               "| 名称 | shape | min | max | mean |",
               "|------|-------|-----|-----|------|"]
        for name, stats in summary_data["detect_layers"].items():
            md.append(
                f"| {name} | {stats['shape']} | {stats['fp32_min']:.4f} "
                f"| {stats['fp32_max']:.4f} | {stats['fp32_mean']:.4f} |"
            )
        md.append("")

    if summary_data.get("postprocess"):
        md += ["## 后处理 golden (FP32)", ""]
        for name, stats in summary_data["postprocess"].items():
            md.append(f"### {name}")
            for k, v in stats.items():
                md.append(f"- {k}: `{v}`")
            md.append("")

    if summary_data.get("psa_attn"):
        md += ["## PSA 注意力 golden (FP32, P1)", ""]
        for k, v in summary_data["psa_attn"].items():
            md.append(f"- {k}: `{v}`")
        md.append("")

    md += [
        "## Neck 模块输出别名",
        "",
        "| 别名头文件 | 等价数据 | 说明 |",
        "|-----------|---------|------|",
    ]
    for alias, conv_id, desc in NECK_ALIASES:
        md.append(f"| shen_golden_{alias}.h | conv_{conv_id} POST_ACT | {desc} |")
    md.append("")

    md += [
        "## 导出文件清单",
        "",
        "```",
        "exports/golden/",
    ]
    for f in sorted(out_dir.glob("shen_golden_*")):
        md.append(f"  {f.name}")
    md += ["```", ""]

    with open(out_dir / "shen_golden_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))


# ==================== 参数解析 ====================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Gemmini golden reference 导出 (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python tools/shen_export_golden.py                          # 全部导出
  python tools/shen_export_golden.py --stage backbone         # 仅 backbone
  python tools/shen_export_golden.py --stage backbone,neck    # backbone + neck
  python tools/shen_export_golden.py --convs 20,30,32,39      # 指定 conv 层
  python tools/shen_export_golden.py --priority p0            # 仅 P0 项
        """,
    )
    parser.add_argument("--stage", type=str, default="all",
                        help="导出阶段: all, preprocess, backbone, neck, head, postprocess (可逗号分隔)")
    parser.add_argument("--convs", type=str, default=None,
                        help="指定 conv 层 ID (逗号分隔), 如 --convs 20,30,32,39")
    parser.add_argument("--priority", type=str, default="all", choices=["p0", "p1", "all"],
                        help="优先级过滤: p0=仅必须, p1=含建议, all=全部")
    parser.add_argument("--model", type=str, default=None,
                        help="模型权重路径 (覆盖默认)")
    parser.add_argument("--quant", type=str, default=None,
                        help="量化参数 JSON 路径 (覆盖默认)")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"输出目录 (默认: {OUTPUT_DIR})")
    return parser.parse_args()


# ==================== 主流程 ====================

def main():
    args = parse_args()
    os.chdir(REPO_ROOT)

    stages = set(args.stage.split(","))
    do_all = "all" in stages
    do_preprocess = do_all or "preprocess" in stages
    do_backbone = do_all or "backbone" in stages
    do_neck = do_all or "neck" in stages
    do_head = do_all or "head" in stages
    do_postprocess = do_all or "postprocess" in stages

    explicit_convs: set[int] | None = None
    if args.convs:
        explicit_convs = {int(x.strip()) for x in args.convs.split(",")}
        do_backbone = True
        do_neck = True

    priority_filter = {"p0"} if args.priority == "p0" else {"p0", "p1"}

    print("=" * 60)
    print(" Gemmini Golden Reference 导出 (v2)")
    print("=" * 60)
    print()

    # 路径解析
    img_path, img_md5 = find_test_image()
    pt_path = find_best_pt(args.model)
    output_scales, input_scales = load_quant_params(args.quant)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        pt_rel = str(Path(pt_path).relative_to(REPO_ROOT))
    except ValueError:
        pt_rel = pt_path
    try:
        img_rel = str(Path(img_path).relative_to(REPO_ROOT))
    except ValueError:
        img_rel = img_path

    print(f"  测试图片 : {img_rel}")
    print(f"  图片 MD5 : {img_md5}")
    print(f"  模型权重 : {pt_rel}")
    print(f"  量化参数 : {args.quant or QUANT_JSON}")
    print(f"  输出目录 : {out_dir}")
    print(f"  导出阶段 : {args.stage}")
    print(f"  优先级   : {args.priority}")
    if explicit_convs:
        print(f"  指定 conv : {sorted(explicit_convs)}")
    print()

    if img_md5.lower() != EXPECTED_IMG_MD5.lower():
        print(f"  [警告] 图片 MD5 不一致: 预期 {EXPECTED_IMG_MD5}, 实际 {img_md5}")
        print()

    summary_data = {
        "model_weights": pt_rel,
        "model_md5": file_md5(pt_path),
        "test_image": img_rel,
        "test_image_md5": img_md5,
        "letterbox_params": {
            "new_shape": [640, 640], "auto": False, "scaleup": True,
            "center": True, "stride": 32,
            "interpolation": "cv2.INTER_LINEAR", "padding_value": 114,
        },
        "input_scale": INPUT_SCALE,
        "conv_layers": {},
        "detect_layers": {},
        "postprocess": {},
    }

    # ========== 1. 前处理 ==========
    letterbox_bgr = None
    if do_preprocess:
        print("[1] 前处理 golden...")
        letterbox_bgr, _, preprocess_stats = export_preprocess_golden(img_path, out_dir)
        summary_data["preprocess"] = preprocess_stats
        print()

    # ========== 2. 加载模型 ==========
    need_model = do_backbone or do_neck or do_head or do_postprocess
    model = None
    input_nchw = None
    model_output = None

    if need_model:
        print("[2] 加载模型...")
        from ultralytics import YOLO
        yolo = YOLO(pt_path)
        model = yolo.model
        model.eval()
        if not hasattr(model, "model"):
            model = getattr(model, "model", model)

        if letterbox_bgr is None:
            from ultralytics.data.augment import LetterBox
            img_bgr = cv2.imread(img_path)
            lb = LetterBox(new_shape=(640, 640), auto=False, scaleup=True, center=True, stride=32)
            letterbox_bgr = lb(image=img_bgr.copy())

        img_rgb = letterbox_bgr[:, :, ::-1].copy()
        img_fp32 = img_rgb.astype(np.float32) / 255.0
        input_nchw = torch.from_numpy(img_fp32).permute(2, 0, 1).unsqueeze(0).float()
        input_nchw = input_nchw.to(next(model.parameters()).device)
        print(f"  模型加载完成, device={input_nchw.device}")
        print()

    # ========== 3. 确定需要导出的 conv 层 ==========
    conv_to_export = []
    for conv_id, module_getter, desc, category, priority in CONV_GOLDEN_CONFIG:
        if explicit_convs is not None:
            if conv_id not in explicit_convs:
                continue
        else:
            if category == "backbone" and not do_backbone:
                continue
            if category == "neck" and not do_neck:
                continue
            if priority not in priority_filter:
                continue
        conv_to_export.append((conv_id, module_getter, desc, category, priority))

    # ========== 4. 注册所有 hooks + 单次前向传播 ==========
    conv_capture = ConvHookCapture()
    detect_capture = DetectHookCapture()
    detect_to_export = []
    need_forward = False

    if model is not None:
        # 注册 conv hooks
        if conv_to_export:
            for conv_id, module_getter, desc, _cat, _pri in conv_to_export:
                scale_key = f"conv_{conv_id}"
                if scale_key not in output_scales:
                    print(f"  [跳过] conv_{conv_id}: 未找到 output_scale ({desc})")
                    continue
                try:
                    conv_capture.register(conv_id, module_getter(model))
                    need_forward = True
                except Exception as e:
                    print(f"  [错误] conv_{conv_id} hook 注册失败: {e}")

        # 注册 detect head hooks
        if do_head:
            for name, module_getter, desc, priority in DETECT_GOLDEN_CONFIG:
                if priority not in priority_filter:
                    continue
                try:
                    detect_capture.register(name, module_getter(model))
                    detect_to_export.append((name, desc))
                    need_forward = True
                except Exception as e:
                    print(f"  [错误] {name} hook 注册失败: {e}")

        # 后处理也需要前向传播结果
        if do_postprocess:
            need_forward = True

        # 单次前向传播
        if need_forward:
            n_conv_hooks = len(conv_capture._hooks) // 2
            print(f"[3] 执行前向传播 (conv={n_conv_hooks} hooks, "
                  f"detect={len(detect_to_export)} hooks)...")
            with torch.no_grad():
                model_output = model(input_nchw)
            print()

        conv_capture.cleanup()
        detect_capture.cleanup()

    # ========== 5. 导出 conv golden ==========
    if conv_to_export and model is not None:
        print(f"[4] Conv 层 golden ({len(conv_to_export)} 层)...")
        for conv_id, _getter, desc, category, _pri in conv_to_export:
            scale_key = f"conv_{conv_id}"
            out_scale = output_scales.get(scale_key)
            if out_scale is None or conv_id not in conv_capture.pre_act:
                continue

            pre_nhwc = nchw_to_nhwc_np(conv_capture.pre_act[conv_id])
            post_nhwc = nchw_to_nhwc_np(conv_capture.post_act[conv_id])

            next_cid = POST_ACT_NEXT_CONV.get(conv_id)
            post_scale = None
            if next_cid is not None:
                next_key = f"conv_{next_cid}"
                post_scale = input_scales.get(next_key)

            stats = export_conv_golden(
                conv_id, pre_nhwc, post_nhwc, out_scale, post_scale, desc, out_dir,
            )
            stats["category"] = category
            summary_data["conv_layers"][conv_id] = stats

            h, w, c = pre_nhwc.shape
            pa_mark = "✓" if post_scale else "≈"
            print(f"  conv_{conv_id:>2d}: [{h}][{w}][{c}]  scale={out_scale:.4e}  "
                  f"PRE[{stats['pre_act_int8_min']:>4d},{stats['pre_act_int8_max']:>4d}]  "
                  f"POST[{stats['post_act_int8_min']:>4d},{stats['post_act_int8_max']:>4d}] {pa_mark}  "
                  f"({desc})")
        print()

    # ========== 6. 导出 detect head golden ==========
    if detect_to_export:
        print(f"[5] 检测头 golden ({len(detect_to_export)} 个, FP32)...")
        for name, desc in detect_to_export:
            if name not in detect_capture.outputs:
                print(f"  [跳过] {name}: hook 未捕获输出")
                continue
            tensor_nhwc = nchw_to_nhwc_np(detect_capture.outputs[name])
            stats = export_detect_golden(name, tensor_nhwc, desc, out_dir)
            summary_data["detect_layers"][name] = stats
            h, w, c = tensor_nhwc.shape
            print(f"  {name}: [{h}][{w}][{c}]  range=[{stats['fp32_min']:.4f}, {stats['fp32_max']:.4f}]")
        print()

    # ========== 7. Neck 模块别名 ==========
    if (do_neck or do_all) and model is not None:
        print("[6] Neck 模块输出别名...")
        exported_convs = set(summary_data["conv_layers"].keys())
        for alias, conv_id, desc in NECK_ALIASES:
            if conv_id in exported_convs:
                export_neck_alias(alias, conv_id, desc, out_dir)
                print(f"  {alias}.h → conv_{conv_id} POST_ACT")
            else:
                print(f"  [跳过] {alias}: conv_{conv_id} 未导出")
        print()

    # ========== 8. 后处理 golden ==========
    if do_postprocess and model is not None:
        print("[7] 后处理 golden (FP32)...")

        if model_output is None:
            with torch.no_grad():
                model_output = model(input_nchw)

        try:
            detect_head = model.model[23]
        except (AttributeError, IndexError):
            detect_head = None

        pp_stats = export_postprocess_golden(model_output, DetectHookCapture(), detect_head, out_dir)
        summary_data["postprocess"] = pp_stats
        print()

    # ========== 9. PSA 注意力 (P1) ==========
    if (do_backbone or do_all) and "p1" in priority_filter and model is not None:
        print("[8] PSA 注意力 golden (P1)...")
        psa_stats = export_psa_attention(model, input_nchw, out_dir)
        if psa_stats:
            summary_data["psa_attn"] = psa_stats
        print()

    # ========== 10. 生成摘要 ==========
    print("[9] 生成 shen_golden_summary.md...")
    generate_summary(summary_data, conv_to_export if conv_to_export else [], out_dir)
    print(f"  完成: {out_dir / 'shen_golden_summary.md'}")
    print()

    # 统计
    h_files = list(out_dir.glob("shen_golden_*.h"))
    bin_files = list(out_dir.glob("shen_golden_*.bin"))
    total_h = sum(f.stat().st_size for f in h_files)
    total_bin = sum(f.stat().st_size for f in bin_files)
    print("=" * 60)
    print(f" 导出完成: {out_dir.absolute()}")
    print(f"   .h  文件: {len(h_files)} 个, 共 {total_h / 1024 / 1024:.1f} MB")
    print(f"   .bin 文件: {len(bin_files)} 个, 共 {total_bin / 1024 / 1024:.1f} MB")
    print(f"   可将 exports/golden/ 同步到 Linux 工作区用于逐层验证")
    print("=" * 60)


if __name__ == "__main__":
    main()
