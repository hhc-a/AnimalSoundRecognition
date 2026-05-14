"""
model.py
CNN 模型推理 - 使用 ONNX Runtime（輕量，無需 TensorFlow）
"""

import os
import numpy as np

# ─── 動物類別 ──────────────────────────────────────────────────────────────
CLASSES = [
    "dog",
    "cat",
    "bird",
    "sheep",
]

CLASS_ZH = {
    "dog":   "狗 🐕",
    "cat":   "貓 🐈",
    "bird":  "鳥 🐦",
    "sheep": "羊 🐑",
}

MODEL_PATH = "animal_cnn_model.onnx"
IMG_SIZE   = 96

# ─── 全域 session（載入一次）────────────────────────────────────────────────
_ort_session = None

def load_model():
    """載入 ONNX 模型，回傳 ort session"""
    global _ort_session
    if _ort_session is not None:
        return _ort_session

    if not os.path.exists(MODEL_PATH):
        print(f"[model] 找不到 {MODEL_PATH}")
        return None

    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        _ort_session = ort.InferenceSession(
            MODEL_PATH,
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        print(f"[model] ONNX 模型載入完成：{MODEL_PATH}")
        return _ort_session
    except Exception as e:
        print(f"[model] 載入失敗：{e}")
        return None


def predict_from_array(session, img_array: np.ndarray) -> dict:
    """
    img_array: (96, 96, 3) uint8 numpy array
    回傳各類別機率
    """
    x = img_array.astype(np.float32) / 255.0
    x = np.expand_dims(x, 0)   # (1, 96, 96, 3)

    input_name = session.get_inputs()[0].name
    probs = session.run(None, {input_name: x})[0][0]

    results = {
        cls: float(round(prob * 100, 2))
        for cls, prob in zip(CLASSES, probs)
    }
    top_class = CLASSES[int(np.argmax(probs))]
    top_prob  = float(round(np.max(probs) * 100, 2))

    return {
        "top_class":    top_class,
        "top_class_zh": CLASS_ZH[top_class],
        "top_prob":     top_prob,
        "all_probs":    results,
    }
