"""
train_model.py
CNN 模型訓練腳本 - 含音訊資料增強

資料增強策略（50 個音檔 → ~300 個）：
  1. 原始音訊
  2. 加入背景白噪音
  3. 時間位移（左右平移）
  4. 音調偏移（Pitch Shift）
  5. 時間拉伸（Time Stretch）
  6. 音量縮放

使用方法：
  1. 在 data/ 放音檔（每類 30 個以上即可）
  2. python train_model.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import librosa

from audio_processor import (
    extract_mel_spectrogram, spectrogram_to_image_array,
    reduce_noise, trim_silence,
    SAMPLE_RATE, DURATION
)
from model import build_cnn_model, CLASSES, MODEL_PATH

DATA_DIR         = "data"
BATCH_SIZE       = 16     # 資料少時用小 batch
EPOCHS           = 60
FINE_TUNE_EPOCHS = 20
AUGMENT_TIMES    = 5      # 每個音檔增強幾次（原始 + 5 種 = 6 倍資料）


# ─── 音訊增強函式 ────────────────────────────────────────────────────────

def augment_audio(y: np.ndarray, sr: int, method: str) -> np.ndarray:
    """
    對原始音訊波形做增強，回傳增強後的波形
    method: 'noise' | 'shift' | 'pitch' | 'stretch' | 'volume'
    """
    target_len = int(sr * DURATION)

    if method == 'noise':
        noise_amp = 0.008 * np.random.uniform(0.5, 1.5)
        y = y + noise_amp * np.random.randn(len(y))

    elif method == 'shift':
        shift = int(sr * np.random.uniform(0.2, 1.5))
        direction = np.random.choice([-1, 1])
        y = np.roll(y, shift * direction)
        if direction > 0:
            y[:shift] = 0
        else:
            y[shift:] = 0

    elif method == 'pitch':
        n_steps = np.random.uniform(-4, 4)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

    elif method == 'stretch':
        rate = np.random.uniform(0.75, 1.25)
        y = librosa.effects.time_stretch(y, rate=rate)

    elif method == 'volume':
        factor = np.random.uniform(0.4, 1.6)
        y = y * factor

    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    return y.astype(np.float32)


def audio_to_cnn_input(y: np.ndarray, sr: int) -> np.ndarray:
    """波形 → 頻譜圖 → CNN 輸入圖片"""
    mel_db = extract_mel_spectrogram(y, sr)
    return spectrogram_to_image_array(mel_db)


# ─── 資料載入（含增強）─────────────────────────────────────────────────────

def load_dataset_with_augmentation(data_dir: str = DATA_DIR):
    """
    載入音檔並套用音訊增強，讓每類資料量擴充約 6 倍
    """
    images, labels = [], []
    valid_ext = ('.wav', '.mp3', '.ogg', '.flac', '.m4a')
    aug_methods = ['noise', 'shift', 'pitch', 'stretch', 'volume']

    for cls_idx, cls_name in enumerate(CLASSES):
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"[warn] 找不到資料夾：{cls_dir}，跳過")
            continue

        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(valid_ext)]
        print(f"\n[data] {cls_name}: {len(files)} 個原始音檔 → 增強後約 {len(files) * (1 + AUGMENT_TIMES)} 張")

        for fname in tqdm(files, desc=f"  處理+增強 {cls_name}"):
            fpath = os.path.join(cls_dir, fname)
            try:
                y, sr = librosa.load(fpath, sr=SAMPLE_RATE, duration=DURATION)
                y = reduce_noise(y, sr)
                y = trim_silence(y)

                target_len = int(sr * DURATION)
                if len(y) < target_len:
                    y = np.pad(y, (0, target_len - len(y)))
                else:
                    y = y[:target_len]

                # 原始音訊
                images.append(audio_to_cnn_input(y, sr))
                labels.append(cls_idx)

                # 增強版本
                for method in aug_methods[:AUGMENT_TIMES]:
                    try:
                        y_aug = augment_audio(y.copy(), sr, method)
                        images.append(audio_to_cnn_input(y_aug, sr))
                        labels.append(cls_idx)
                    except Exception:
                        pass

            except Exception as e:
                print(f"  [skip] {fname}: {e}")

    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int32)


# ─── 訓練 ─────────────────────────────────────────────────────────────────

def train():
    print("=" * 50)
    print("動物聲音辨識 CNN 訓練（含音訊增強）")
    print("=" * 50)

    X, y = load_dataset_with_augmentation()

    if len(X) == 0:
        print("\n[error] 沒有找到任何訓練資料！")
        return

    print("\n各類別資料量（增強後）：")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls}: {np.sum(y == i)} 張")

    num_classes = len(CLASSES)
    y_onehot = tf.keras.utils.to_categorical(y, num_classes)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n訓練集：{len(X_train)} 張，驗證集：{len(X_val)} 張")

    X_train = X_train.astype(np.float32) / 255.0
    X_val   = X_val.astype(np.float32)   / 255.0

    # 頻譜圖層級的輕微增強
    spec_augment = tf.keras.Sequential([
        tf.keras.layers.RandomBrightness(0.15),
        tf.keras.layers.RandomContrast(0.15),
    ], name="spec_augment")

    def make_dataset(X, y, augment_flag=False):
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if augment_flag:
            ds = ds.map(
                lambda x, lbl: (spec_augment(x, training=True), lbl),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        return ds.shuffle(500).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    train_ds = make_dataset(X_train, y_train, augment_flag=True)
    val_ds   = make_dataset(X_val,   y_val,   augment_flag=False)

    model = build_cnn_model(num_classes)
    model.summary()

    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]

    # ── Phase 1：訓練分類頭 ──────────────────────────────────────────────
    print("\n[Phase 1] 訓練分類頭（Backbone 凍結）...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=1)

    # ── Phase 2：微調 ────────────────────────────────────────────────────
    print("\n[Phase 2] 微調（解凍 MobileNetV2 後 50 層）...")
    # 找 MobileNetV2 backbone（用 type 找，不依賴名稱）
    base_model = next(l for l in model.layers if 'mobilenetv2' in l.name.lower())
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS, callbacks=callbacks, verbose=1)

    print("\n[評估] 最終驗證集結果：")
    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"  Loss: {loss:.4f}  Accuracy: {acc*100:.2f}%")

    with open("classes.json", "w", encoding="utf-8") as f:
        json.dump(CLASSES, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 模型已儲存至：{MODEL_PATH}")


if __name__ == "__main__":
    train()
