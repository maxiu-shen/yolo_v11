"""
INT8 模型输出诊断：检查 INT8 QDQ ONNX 模型的原始输出值
"""
import os, sys, numpy as np, cv2
import onnxruntime as ort

INT8_ONNX = "exports/onnx/train4_best_int8.onnx"
FP32_ONNX = "exports/onnx/train4_best_fp32.onnx"
TEST_IMG = "datasets/BDD100K/images/val/b1c66a42-6f7d68ca.jpg"
IMG_SIZE = 640

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img

def preprocess(img_path):
    img = cv2.imread(img_path)
    img = letterbox(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

def analyze_output(name, output):
    print(f"\n{'='*60}")
    print(f"模型: {name}")
    print(f"  输出 shape: {output.shape}")
    print(f"  dtype: {output.dtype}")
    print(f"  min: {output.min():.6f}, max: {output.max():.6f}")
    print(f"  mean: {output.mean():.6f}, std: {output.std():.6f}")
    print(f"  nan count: {np.isnan(output).sum()}")
    print(f"  inf count: {np.isinf(output).sum()}")
    print(f"  zero count: {(output == 0).sum()} / {output.size}")

    if output.ndim == 3 and output.shape[1] == 14:
        bbox_raw = output[0, :4, :]
        cls_raw = output[0, 4:, :]
        print(f"\n  bbox 通道 (前4行):")
        print(f"    min: {bbox_raw.min():.6f}, max: {bbox_raw.max():.6f}, mean: {bbox_raw.mean():.6f}")
        print(f"  cls 通道 (后10行):")
        print(f"    min: {cls_raw.min():.6f}, max: {cls_raw.max():.6f}, mean: {cls_raw.mean():.6f}")

        from scipy.special import expit
        try:
            cls_sigmoid = expit(cls_raw)
            max_conf = cls_sigmoid.max(axis=0)
            print(f"\n  sigmoid 后最大置信度:")
            print(f"    min: {max_conf.min():.6f}, max: {max_conf.max():.6f}")
            print(f"    mean: {max_conf.mean():.6f}")
            print(f"    >0.25 count: {(max_conf > 0.25).sum()}")
            print(f"    >0.1 count: {(max_conf > 0.1).sum()}")
            print(f"    >0.01 count: {(max_conf > 0.01).sum()}")
        except ImportError:
            cls_sigmoid = 1.0 / (1.0 + np.exp(-np.clip(cls_raw, -50, 50)))
            max_conf = cls_sigmoid.max(axis=0)
            print(f"\n  sigmoid 后最大置信度:")
            print(f"    min: {max_conf.min():.6f}, max: {max_conf.max():.6f}")
            print(f"    >0.25 count: {(max_conf > 0.25).sum()}")
            print(f"    >0.1 count: {(max_conf > 0.1).sum()}")
            print(f"    >0.01 count: {(max_conf > 0.01).sum()}")

        print(f"\n  前5个检测框原始输出 (bbox + cls):")
        for i in range(min(5, output.shape[2])):
            vals = output[0, :, i]
            print(f"    [{i}] bbox: {vals[:4]}, cls_max: {vals[4:].max():.6f}")

def main():
    if not os.path.exists(TEST_IMG):
        imgs = os.listdir("datasets/BDD100K/images/val")[:1]
        test_img = os.path.join("datasets/BDD100K/images/val", imgs[0])
    else:
        test_img = TEST_IMG

    print(f"测试图片: {test_img}")
    inp = preprocess(test_img)
    print(f"输入 shape: {inp.shape}, dtype: {inp.dtype}")
    print(f"输入 min: {inp.min():.4f}, max: {inp.max():.4f}")

    print(f"\nONNX Runtime version: {ort.__version__}")
    providers = ort.get_available_providers()
    print(f"可用 providers: {providers}")

    for model_path, name in [(FP32_ONNX, "FP32"), (INT8_ONNX, "INT8")]:
        if not os.path.exists(model_path):
            print(f"\n[跳过] {name} 模型不存在: {model_path}")
            continue
        sess = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        active_provider = sess.get_providers()
        print(f"\n{name} 模型实际 providers: {active_provider}")
        input_name = sess.get_inputs()[0].name
        out = sess.run(None, {input_name: inp})
        analyze_output(name, out[0])

if __name__ == "__main__":
    main()
