"""
Convert BDD100K detection annotations (aggregate JSON) to YOLO label format.

Input:  BDD100K aggregate annotation JSON files
        (bdd100k_labels_images_train.json / bdd100k_labels_images_val.json)
Output: Per-image .txt files in YOLO format:
        <class_id> <x_center> <y_center> <width> <height>
        All coordinates normalized to [0, 1].

BDD100K images are 1280x720. Categories like 'lane' and 'drivable area'
have no box2d and are skipped automatically.

Usage:
    python tools/bdd100k_to_yolo.py
    python tools/bdd100k_to_yolo.py --output-root datasets/BDD100K/labels
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

IMG_W, IMG_H = 1280, 720

YOLO_CLASSES = [
    "pedestrian",     # 0
    "rider",          # 1
    "car",            # 2
    "truck",          # 3
    "bus",            # 4
    "train",          # 5
    "motorcycle",     # 6
    "bicycle",        # 7
    "traffic light",  # 8
    "traffic sign",   # 9
]

CATEGORY_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(YOLO_CLASSES)}

CATEGORY_ALIASES: dict[str, str] = {
    "person": "pedestrian",
    "bike": "bicycle",
    "motor": "motorcycle",
}

JSON_FILES: dict[str, Path] = {
    "train": Path(
        r"datasets/BDD100K/raw/iic/BDD100K/master/data_files/extracted"
        r"/0901178fbfdcdf40ad6840a530ab611e633b3edc3b29a2bc7f16479be17a3c49"
        r"/train/annotations/bdd100k_labels_images_train.json"
    ),
    "val": Path(
        r"datasets/BDD100K/raw/iic/BDD100K/master/data_files/extracted"
        r"/58d3c2e131bde12560fe8ca8a895abaf13b8f1bb7ebcff9e796360e29b66b1b9"
        r"/val/annotations/bdd100k_labels_images_val.json"
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BDD100K JSON -> YOLO txt converter")
    p.add_argument("--output-root", type=Path, default=Path("datasets/BDD100K/labels"))
    p.add_argument("--splits", nargs="+", default=["train", "val"],
                   choices=["train", "val"])
    return p.parse_args()


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def convert_box(box2d: dict) -> tuple[float, float, float, float] | None:
    """Convert BDD100K xyxy box to YOLO normalized xywh. Returns None if invalid."""
    x1 = clamp(float(box2d["x1"]), 0.0, IMG_W)
    y1 = clamp(float(box2d["y1"]), 0.0, IMG_H)
    x2 = clamp(float(box2d["x2"]), 0.0, IMG_W)
    y2 = clamp(float(box2d["y2"]), 0.0, IMG_H)
    if x2 <= x1 or y2 <= y1:
        return None
    xc = (x1 + x2) / 2.0 / IMG_W
    yc = (y1 + y2) / 2.0 / IMG_H
    w = (x2 - x1) / IMG_W
    h = (y2 - y1) / IMG_H
    return xc, yc, w, h


def convert_split(json_path: Path, out_dir: Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Loading {json_path.name} ...", flush=True)
    t0 = time.time()
    with json_path.open("r", encoding="utf-8") as f:
        records: list[dict] = json.load(f)
    print(f"  Loaded {len(records)} records in {time.time() - t0:.1f}s", flush=True)

    stats = {
        "records": len(records),
        "files_written": 0,
        "boxes_written": 0,
        "boxes_skipped_no_box2d": 0,
        "boxes_skipped_bad_geom": 0,
        "boxes_skipped_unknown_cat": 0,
        "empty_labels": 0,
    }
    unknown_cats: dict[str, int] = {}

    t0 = time.time()
    for i, rec in enumerate(records):
        name: str | None = rec.get("name")
        if not name:
            continue

        lines: list[str] = []
        labels = rec.get("labels")
        if isinstance(labels, list):
            for lbl in labels:
                box2d = lbl.get("box2d")
                if not isinstance(box2d, dict):
                    stats["boxes_skipped_no_box2d"] += 1
                    continue

                raw_cat = str(lbl.get("category", "")).strip()
                cat = CATEGORY_ALIASES.get(raw_cat, raw_cat)
                cid = CATEGORY_TO_ID.get(cat)
                if cid is None:
                    stats["boxes_skipped_unknown_cat"] += 1
                    unknown_cats[raw_cat] = unknown_cats.get(raw_cat, 0) + 1
                    continue

                converted = convert_box(box2d)
                if converted is None:
                    stats["boxes_skipped_bad_geom"] += 1
                    continue

                xc, yc, w, h = converted
                lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                stats["boxes_written"] += 1

        stem = Path(name).stem
        (out_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        stats["files_written"] += 1
        if not lines:
            stats["empty_labels"] += 1

        if (i + 1) % 10000 == 0 or (i + 1) == len(records):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(records)}]  {rate:.0f} rec/s  "
                  f"boxes={stats['boxes_written']}", flush=True)

    if unknown_cats:
        print(f"  Unknown categories (skipped): {unknown_cats}")

    return stats


def write_classes_file(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "classes.txt").write_text(
        "\n".join(YOLO_CLASSES) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output_root: Path = args.output_root

    print(f"Output root: {output_root.resolve()}")
    print(f"Splits: {args.splits}")
    print(f"Image size: {IMG_W}x{IMG_H}")
    print(f"Classes ({len(YOLO_CLASSES)}): {YOLO_CLASSES}")
    print(f"Aliases: {CATEGORY_ALIASES}")
    print()

    write_classes_file(output_root)

    for split in args.splits:
        jp = JSON_FILES[split]
        if not jp.exists():
            print(f"ERROR: {jp} not found, skipping {split}", file=sys.stderr)
            continue

        print(f"Converting {split}:")
        out_dir = output_root / split
        stats = convert_split(jp, out_dir)

        print(f"  --- {split} summary ---")
        for k, v in stats.items():
            print(f"    {k}: {v}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
