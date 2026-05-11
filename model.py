"""
model.py
CNN 模型定義、載入與推理
結合 CNN（視覺特徵萃取）+ 傳統音訊特徵 的混合架構
"""

import os
import numpy as np
import json

# ─── 動物類別 ──────────────────────────────────────────────────────────────
CLASSES = [
    "dog",    # 狗
    "cat",    # 貓
    "bird",   # 鳥
    "sheep",  # 羊
]

CLASS_ZH = {
    "dog":   "狗 🐕",
    "cat":   "貓 🐈",
    "bird":  "鳥 🐦",
    "sheep": "羊 🐑",
}

MODEL_PATH = "animal_cnn_model.h5"
IMG_SIZE   = 224


# ─── CNN 模型建立 ─────────────────────────────────────────────────────────

def build_cnn_model(num_classes: int = len(CLASSES)):
    """
    建立 CNN 模型：
    - 使用預訓練的 MobileNetV2 作為 backbone（Transfer Learning）
    - 加入自訂分類頭，適應動物聲音頻譜圖分類任務
    架構說明：
        Input (224×224×3)
        → MobileNetV2 (pretrained ImageNet, frozen)
        → GlobalAveragePooling2D
        → Dense(256) + BatchNorm + Dropout(0.5)
        → Dense(128) + BatchNorm + Dropout(0.3)
        → Dense(num_classes, softmax)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model._name = 'mobilenetv2_backbone'
    # 凍結底層，只訓練分類頭
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # 頻譜圖已是 0~1，不套 preprocess_input（會把值轉成 -1~1，對頻譜圖有害）
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def load_model():
    """載入已訓練的模型，若不存在則建立新模型"""
    import tensorflow as tf
    if os.path.exists(MODEL_PATH):
        print(f"[model] 載入已訓練模型：{MODEL_PATH}")
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        print("[model] 找不到訓練好的模型，建立未訓練模型（請先執行 train_model.py）")
        return build_cnn_model()


# ─── 推理 ────────────────────────────────────────────────────────────────

def predict_from_array(model, img_array: np.ndarray) -> dict:
    """
    img_array: (224, 224, 3) uint8 numpy array
    回傳各類別機率
    """
    x = img_array.astype(np.float32) / 255.0
    x = np.expand_dims(x, 0)        # (1, 224, 224, 3)
    probs = model.predict(x, verbose=0)[0]

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


# ─── 特徵描述（供 LLM 使用）─────────────────────────────────────────────

def build_feature_description(features: dict, cnn_result: dict) -> str:
    """
    將音訊特徵數值轉為自然語言描述，傳給 LLM 做二次分析
    """
    f0   = features.get("fundamental_freq_hz", 0)
    rms  = features.get("rms_energy", 0)
    zcr  = features.get("zero_crossing_rate", 0)
    sc   = features.get("spectral_centroid", 0)
    sb   = features.get("spectral_bandwidth", 0)
    dur  = features.get("duration_sec", 0)

    # CNN 前三名
    sorted_probs = sorted(
        cnn_result["all_probs"].items(),
        key=lambda x: x[1], reverse=True
    )[:3]
    top3_str = ", ".join(
        f"{CLASS_ZH.get(cls, cls)}: {prob:.1f}%"
        for cls, prob in sorted_probs
    )

    desc = f"""
音訊特徵摘要：
- 基頻 (F0)：{f0:.1f} Hz（{'高音' if f0 > 2000 else '中音' if f0 > 500 else '低音'}）
- 音量強度 (RMS)：{rms:.4f}（{'強' if rms > 0.05 else '中' if rms > 0.01 else '弱'}）
- 過零率 (ZCR)：{zcr:.4f}（{'高雜訊/尖銳' if zcr > 0.1 else '平滑'}）
- 頻譜重心：{sc:.1f} Hz
- 頻譜帶寬：{sb:.1f} Hz
- 音訊長度：{dur:.2f} 秒

CNN 模型辨識結果（前三名）：
{top3_str}

最高信心度預測：{cnn_result['top_class_zh']}（{cnn_result['top_prob']:.1f}%）
"""
    return desc.strip()
