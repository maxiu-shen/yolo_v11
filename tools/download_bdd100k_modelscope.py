"""
Download BDD100K via ModelScope and organize into datasets/BDD100K structure.
Run: conda activate yolo_v11 && python tools/download_bdd100k_modelscope.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode

CACHE_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "BDD100K" / "raw"
IMAGES_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "BDD100K" / "images"
LABELS_RAW_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "BDD100K" / "labels_raw"

# Try both namespaces; ModelScope hub may use damo or iic
NAMESPACES = ["damo", "iic"]


def load_and_organize(split: str, ms_split: str, namespace: str, cache_dir: str) -> bool:
    try:
        ds = MsDataset.load(
            "BDD100K",
            split=ms_split,
            namespace=namespace,
            download_mode=DownloadMode.FORCE_REDOWNLOAD,
            cache_dir=cache_dir,
        )
    except Exception as e:
        print(f"  load failed ({namespace}): {e}")
        return False

    split_images = IMAGES_ROOT / split
    split_labels = LABELS_RAW_ROOT / split
    split_images.mkdir(parents=True, exist_ok=True)
    split_labels.mkdir(parents=True, exist_ok=True)

    count = 0
    for item in ds:
        if not isinstance(item, dict):
            continue
        # Resolve image path (ModelScope may use different keys)
        img_path = None
        for key in ("image:FILE", "image", "Image", "image_path"):
            v = item.get(key)
            if v is not None:
                if isinstance(v, list) and v:
                    v = v[0]
                if isinstance(v, str) and Path(v).exists():
                    img_path = Path(v)
                    break
                if isinstance(v, dict) and v.get("path"):
                    p = Path(v["path"])
                    if p.exists():
                        img_path = p
                        break
        if not img_path or not img_path.exists():
            continue

        stem = img_path.stem
        dst_img = split_images / f"{stem}.jpg"
        if not dst_img.exists() or dst_img.stat().st_size == 0:
            try:
                shutil.copy2(img_path, dst_img)
            except OSError:
                continue

        # Save labels (detection JSON per image)
        label = item.get("labels:FIELD") or item.get("labels") or item.get("annotation") or item.get("Labels")
        if label is not None:
            out_path = split_labels / f"{stem}.json"
            if isinstance(label, (dict, list)):
                out_path.write_text(json.dumps(label, ensure_ascii=False), encoding="utf-8")
            elif hasattr(label, "path") and Path(label.path).exists():
                shutil.copy2(label.path, out_path)

        count += 1
        if count % 1000 == 0:
            print(f"  {split}: {count} items")
    print(f"  {split}: done, {count} items")
    return count > 0


def main() -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    IMAGES_ROOT.mkdir(parents=True, exist_ok=True)
    LABELS_RAW_ROOT.mkdir(parents=True, exist_ok=True)
    cache_dir = str(CACHE_ROOT)
    print(f"Cache: {cache_dir}")

    for split, ms_split in [("train", "train"), ("val", "validation")]:
        print(f"\n--- BDD100K {split} ---")
        for ns in NAMESPACES:
            if load_and_organize(split, ms_split, ns, cache_dir):
                break
        else:
            print(f"  Could not load {split} from any namespace")

    print("\n--- Done ---")


if __name__ == "__main__":
    main()
