# YOLOv11 Preprocess and Postprocess Spec

## Preprocess
- Input image size:
- Keep aspect ratio:
- Resize method:
- Letterbox padding value:
- Color order:
- Input dtype:
- Normalization formula:
- Input tensor layout:
- Batch size assumption:

## Class Mapping
- Class source file:
- Number of classes:
- Class order:

## Raw Output Tensors
| Tensor Name | Shape | Dtype | Meaning |
|-------------|-------|-------|---------|
| output0 |  |  |  |

## Decode Rule
- Bounding box parameterization:
- Anchor or anchor-free:
- Stride list:
- Sigmoid usage:
- Score computation:

## Thresholds
- Confidence threshold:
- IoU threshold:
- Max detections:

## NMS
- NMS type:
- Class-agnostic or per-class:
- Output format:

## Gemmini Side Notes
- Which parts can remain identical on Linux/Gemmini:
- Which parts must be reimplemented in C:
- Any assumptions that must match `shen_yolov11n.c`:
