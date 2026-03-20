# YOLOv11 Model Summary

## Basic Info
- Model name:
- Model source:
- Dataset:
- Input size:
- Training run:
- Export date:

## Accuracy Summary
- FP32 validation result:
- INT8 validation result:
- Main metrics:
- Known accuracy drops:

## Preprocess Summary
- Resize or letterbox:
- Color order:
- Normalization:
- Input tensor layout:

## Postprocess Summary
- Output tensor meaning:
- Decode rule:
- Confidence threshold:
- NMS threshold:
- Class order source:

## Layer Mapping Summary
| Order | Layer Name | Module Type | Input Shape | Output Shape | Gemmini or CPU | Notes |
|------|------------|-------------|-------------|--------------|----------------|------|
| 1 | backbone.conv0 | Conv2d | 1x3x640x640 | 1x16x320x320 | Gemmini | standard conv |

## Gemmini Mapping Notes
- Which layers are expected to map to `tiled_conv_auto`:
- Which layers are expected to map to `tiled_matmul_auto`:
- Which layers stay on CPU/software:
- Any uncertain operators:

## Export Outputs
- ONNX path:
- params.h path:
- Quant table path:
- Class map path:
- Related scripts:

## Known Risks
- 
