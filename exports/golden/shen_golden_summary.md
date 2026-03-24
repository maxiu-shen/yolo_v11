# Golden Reference 导出摘要 (v2)

导出脚本版本: v2

## 元信息
- **模型权重**: `runs\detect\train4\weights\best.pt`
- **模型 MD5**: `fc49c76342e025ae795ccaf4cd27be93`
- **测试图片**: `test_picture\b1c9c847-3bda4659.jpg`
- **图片 MD5**: `8424d960b86559e1d1e872ed2e1fe314` (预期: `8424d960b86559e1d1e872ed2e1fe314`)

## LetterBox 参数
```json
{
  "new_shape": [
    640,
    640
  ],
  "auto": false,
  "scaleup": true,
  "center": true,
  "stride": 32,
  "interpolation": "cv2.INTER_LINEAR",
  "padding_value": 114
}
```

## 前处理 golden
- input_scale: `0.0078740157187`
- letterbox_bgr: min=0, max=255
- preprocess_int8: min=0, max=127

## Conv 层 golden (INT8)

| conv | shape | output_scale | post_act_scale | pre_act range | post_act range | category |
|------|-------|-------------|---------------|---------------|----------------|----------|
| conv_20 | [40, 40, 128] | 9.431566e-02 | 8.460272e-02 ✓ | [-3, 68] | [-3, 76] | backbone |
| conv_30 | [20, 20, 256] | 1.192125e-01 | 8.989369e-02 ✓ | [-2, 60] | [-3, 80] | backbone |
| conv_32 | [20, 20, 256] | 7.998952e-02 | 6.163139e-02 ✓ | [-3, 75] | [-5, 97] | backbone |
| conv_39 | [20, 20, 256] | 1.712230e-01 | 1.712230e-01 ✓ | [-2, 112] | [-2, 112] | backbone |

## 检测头 golden (FP32)

| 名称 | shape | min | max | mean |
|------|-------|-----|-----|------|
| det_p3_cv2 | [80, 80, 64] | -65.3813 | 116.6147 | 0.9966 |
| det_p3_cv3 | [80, 80, 10] | -35.6993 | 1.4678 | -16.8926 |
| det_p4_cv2 | [40, 40, 64] | -44.3965 | 43.4004 | 0.9945 |
| det_p4_cv3 | [40, 40, 10] | -34.9933 | 2.1042 | -15.3907 |
| det_p5_cv2 | [20, 20, 64] | -13.6669 | 20.9325 | 0.9963 |
| det_p5_cv3 | [20, 20, 10] | -29.5357 | 2.5632 | -13.3467 |

## 后处理 golden (FP32)

### dfl_ltrb
- shape: `[8400, 4]`
- dtype: `float32`
- min: `2.178436279296875`
- max: `637.3733520507812`
- mean: `184.31060791015625`

### sigmoid_cls
- shape: `[8400, 10]`
- dtype: `float32`
- min: `3.133236546480129e-16`
- max: `0.9284547567367554`
- mean: `0.000994594767689705`

### detection_results
- count: `113`
- conf_thresh: `0.25`
- iou_thresh: `0.7`
- shape: `[113, 6]`
- dtype: `float32`

## PSA 注意力 golden (FP32, P1)

- shape: `[128, 20, 20]`
- dtype: `float32`
- min: `-5.122780799865723`
- max: `5.110400676727295`
- mean: `0.016768069937825203`

## Neck 模块输出别名

| 别名头文件 | 等价数据 | 说明 |
|-----------|---------|------|
| shen_golden_neck_model13.h | conv_43 POST_ACT | Neck model.13 C3k2 output = conv_43 POST_ACT |
| shen_golden_neck_p3.h | conv_47 POST_ACT | Neck model.16 C3k2 output = conv_47 POST_ACT = Neck P3 |
| shen_golden_neck_p4.h | conv_60 POST_ACT | Neck model.19 C3k2 output = conv_60 POST_ACT = Neck P4 |
| shen_golden_neck_p5.h | conv_78 POST_ACT | Neck model.22 C3k2 output = conv_78 POST_ACT = Neck P5 |

## 导出文件清单

```
exports/golden/
  shen_golden_conv0.h
  shen_golden_conv0_post_act.bin
  shen_golden_conv0_pre_act.bin
  shen_golden_conv1.h
  shen_golden_conv10.h
  shen_golden_conv10_post_act.bin
  shen_golden_conv10_pre_act.bin
  shen_golden_conv1_post_act.bin
  shen_golden_conv1_pre_act.bin
  shen_golden_conv2.h
  shen_golden_conv20.h
  shen_golden_conv20_post_act.bin
  shen_golden_conv20_pre_act.bin
  shen_golden_conv2_post_act.bin
  shen_golden_conv2_pre_act.bin
  shen_golden_conv30.h
  shen_golden_conv30_post_act.bin
  shen_golden_conv30_pre_act.bin
  shen_golden_conv32.h
  shen_golden_conv32_post_act.bin
  shen_golden_conv32_pre_act.bin
  shen_golden_conv39.h
  shen_golden_conv39_post_act.bin
  shen_golden_conv39_pre_act.bin
  shen_golden_conv5.h
  shen_golden_conv5_post_act.bin
  shen_golden_conv5_pre_act.bin
  shen_golden_det_p3_cv2.bin
  shen_golden_det_p3_cv2.h
  shen_golden_det_p3_cv3.bin
  shen_golden_det_p3_cv3.h
  shen_golden_det_p4_cv2.bin
  shen_golden_det_p4_cv2.h
  shen_golden_det_p4_cv3.bin
  shen_golden_det_p4_cv3.h
  shen_golden_det_p5_cv2.bin
  shen_golden_det_p5_cv2.h
  shen_golden_det_p5_cv3.bin
  shen_golden_det_p5_cv3.h
  shen_golden_detection_results.bin
  shen_golden_detection_results.h
  shen_golden_dfl_ltrb.bin
  shen_golden_dfl_ltrb.h
  shen_golden_letterbox_bgr.h
  shen_golden_preprocess_input.bin
  shen_golden_preprocess_input.h
  shen_golden_psa_attn.bin
  shen_golden_psa_attn.h
  shen_golden_sigmoid_cls.bin
  shen_golden_sigmoid_cls.h
  shen_golden_summary.md
```
