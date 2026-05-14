"""
audio_processor.py
音訊前處理模組 - 負責去雜訊、裁切、轉換為頻譜圖
"""

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from scipy.signal import butter, filtfilt


# ─── 設定 ──────────────────────────────────────────
SAMPLE_RATE = 22050       # 統一取樣率
DURATION    = 4.0         # 每段音訊長度（秒）
N_MELS      = 128         # Mel 頻帶數量
HOP_LENGTH  = 512
N_FFT       = 2048
IMG_SIZE    = 96          # CNN 輸入尺寸（縮小以節省 Render 記憶體）


# ─── 工具函式 ─────────────────────────────────────

def load_audio(file_path: str, sr: int = SAMPLE_RATE, duration: float = DURATION):
    """讀取音訊並統一取樣率與長度"""
    y, sr = librosa.load(file_path, sr=sr, duration=duration)
    # 若音訊不足 duration，補零到指定長度
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y, sr


def load_audio_bytes(audio_bytes: bytes, sr: int = SAMPLE_RATE, duration: float = DURATION):
    """
    從 bytes 物件讀取音訊。
    策略（依序嘗試）：
      1. librosa.load（支援 WAV/MP3/OGG/FLAC，內部呼叫 soundfile / audioread）
      2. pydub → 轉 WAV → librosa（處理 WebM/MP4/AAC 等瀏覽器錄音格式）
      3. 都失敗則拋出清楚的錯誤
    需要系統安裝 ffmpeg（pydub 依賴）：
      Windows: https://ffmpeg.org/download.html 或 `choco install ffmpeg`
      Mac:     `brew install ffmpeg`
      Linux:   `sudo apt install ffmpeg`
    """
    import tempfile, os

    target_len = int(sr * duration)

    # ── 策略 1：直接用 librosa 讀取 ──────────────────────────────────────
    try:
        buf = io.BytesIO(audio_bytes)
        y, file_sr = librosa.load(buf, sr=sr, duration=duration, mono=True)
        if len(y) > 0:
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)))
            else:
                y = y[:target_len]
            return y, sr
    except Exception:
        pass

    # ── 策略 2：用 pydub 先轉 WAV（處理 WebM/MP4 等瀏覽器格式）─────────
    try:
        from pydub import AudioSegment

        # 寫入暫存檔（pydub 需要副檔名推斷格式）
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in_path = tmp_in.name

        tmp_out_path = tmp_in_path.replace('.webm', '.wav')
        try:
            seg = AudioSegment.from_file(tmp_in_path)          # 自動偵測格式
            seg = seg.set_channels(1).set_frame_rate(sr)       # 單聲道、統一取樣率
            seg.export(tmp_out_path, format='wav')

            y, file_sr = librosa.load(tmp_out_path, sr=sr, duration=duration, mono=True)
        finally:
            for p in [tmp_in_path, tmp_out_path]:
                if os.path.exists(p):
                    os.remove(p)

        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]
        return y, sr

    except ImportError:
        raise RuntimeError(
            "無法解析此音訊格式。\n"
            "請安裝 pydub 與 ffmpeg：\n"
            "  pip install pydub\n"
            "  # 並安裝 ffmpeg（見 README）"
        )
    except Exception as e:
        raise RuntimeError(
            f"音訊解析失敗：{e}\n"
            "請確認已安裝 ffmpeg，並且音檔格式正確。\n"
            "可用格式：WAV、MP3、OGG、FLAC、WebM、MP4"
        )


def reduce_noise(y: np.ndarray, sr: int) -> np.ndarray:
    """
    簡易去雜訊：
    1. 高通濾波器濾掉低頻背景雜訊
    2. 取前 0.5 秒估計雜訊均值並減去
    """
    # 高通濾波（截止頻率 100 Hz）
    b, a = butter(5, 100 / (sr / 2), btype='high')
    y = filtfilt(b, a, y)

    # 靜音段估計雜訊均值（spectral subtraction 簡化版）
    noise_sample = y[:int(sr * 0.5)]
    noise_power  = np.mean(noise_sample ** 2)
    y = y - np.sign(y) * np.sqrt(np.maximum(y**2 - noise_power, 0))
    return y


def trim_silence(y: np.ndarray, top_db: int = 20):
    """裁切頭尾靜音"""
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed


def extract_mel_spectrogram(y: np.ndarray, sr: int) -> np.ndarray:
    """轉換為 Mel 頻譜圖（dB 刻度），回傳 2D numpy array"""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmax=8000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def extract_features(y: np.ndarray, sr: int) -> dict:
    """
    萃取多種音訊特徵，作為 LLM 分析的補充資訊
    """
    mel_db    = extract_mel_spectrogram(y, sr)
    mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma    = librosa.feature.chroma_stft(y=y, sr=sr)
    zcr       = librosa.feature.zero_crossing_rate(y)
    rms       = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    # 計算基頻（F0）— 動物叫聲的重要特徵
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7')
    )
    f0_mean = float(np.nanmean(f0)) if np.any(voiced_flag) else 0.0

    return {
        "mel_spectrogram": mel_db,
        "mfcc_mean": mfcc.mean(axis=1).tolist(),
        "mfcc_std": mfcc.std(axis=1).tolist(),
        "chroma_mean": chroma.mean(axis=1).tolist(),
        "zero_crossing_rate": float(zcr.mean()),
        "rms_energy": float(rms.mean()),
        "spectral_centroid": float(spectral_centroid.mean()),
        "spectral_bandwidth": float(spectral_bandwidth.mean()),
        "fundamental_freq_hz": f0_mean,
        "duration_sec": len(y) / sr,
    }


def spectrogram_to_image_array(mel_db: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """
    將 Mel 頻譜圖轉為 CNN 可用的 RGB numpy array (size, size, 3)
    使用 'magma' 色彩映射，讓模型對顏色分佈更敏感
    """
    from PIL import Image

    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
    librosa.display.specshow(mel_db, ax=ax, cmap='magma')
    ax.set_aspect('auto')
    ax.axis('off')
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert('RGB').resize((size, size))
    return np.array(img)


def spectrogram_to_base64(mel_db: np.ndarray) -> str:
    """將 Mel 頻譜圖輸出為 base64 PNG，供前端顯示"""
    fig, ax = plt.subplots(figsize=(8, 4))
    img = librosa.display.specshow(
        mel_db, sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis='time', y_axis='mel',
        ax=ax, cmap='magma', fmax=8000
    )
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Mel Spectrogram', fontsize=14, pad=10)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def waveform_to_base64(y: np.ndarray, sr: int) -> str:
    """輸出波形圖 base64，供前端顯示"""
    times = np.linspace(0, len(y) / sr, len(y))
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.plot(times, y, color='#6ee7f7', linewidth=0.5, alpha=0.9)
    ax.fill_between(times, y, alpha=0.3, color='#6ee7f7')
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')
    ax.tick_params(colors='#94a3b8')
    ax.set_xlabel('Time (s)', color='#94a3b8')
    ax.set_ylabel('Amplitude', color='#94a3b8')
    ax.spines[:].set_color('#334155')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor='#0f172a')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def full_pipeline(source, from_bytes: bool = False) -> dict:
    """
    完整前處理流程
    source: 檔案路徑 (str) 或 bytes
    回傳: dict，包含所有特徵與視覺化圖片
    """
    if from_bytes:
        y, sr = load_audio_bytes(source)
    else:
        y, sr = load_audio(source)

    y = reduce_noise(y, sr)
    y = trim_silence(y)

    # 確保長度足夠
    target_len = int(sr * DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    features = extract_features(y, sr)
    mel_db   = features["mel_spectrogram"]

    return {
        "features":       features,
        "cnn_input":      spectrogram_to_image_array(mel_db),
        "spectrogram_b64": spectrogram_to_base64(mel_db),
        "waveform_b64":   waveform_to_base64(y, sr),
    }
