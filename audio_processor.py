"""
audio_processor.py
音訊前處理模組 - 相容 librosa 0.9.2（無 numba 依賴）
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
SAMPLE_RATE = 22050
DURATION    = 4.0
N_MELS      = 128
HOP_LENGTH  = 512
N_FFT       = 2048
IMG_SIZE    = 96


def load_audio(file_path: str, sr: int = SAMPLE_RATE, duration: float = DURATION):
    y, sr = librosa.load(file_path, sr=sr, duration=duration, mono=True)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y, sr


def load_audio_bytes(audio_bytes: bytes, sr: int = SAMPLE_RATE, duration: float = DURATION):
    import tempfile, os
    target_len = int(sr * duration)

    # 策略 1：直接用 librosa 讀取
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

    # 策略 2：pydub 轉 WAV
    try:
        from pydub import AudioSegment
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in_path = tmp_in.name
        tmp_out_path = tmp_in_path.replace('.webm', '.wav')
        try:
            seg = AudioSegment.from_file(tmp_in_path)
            seg = seg.set_channels(1).set_frame_rate(sr)
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
    except Exception as e:
        raise RuntimeError(f"音訊解析失敗：{e}")


def reduce_noise(y: np.ndarray, sr: int) -> np.ndarray:
    try:
        b, a = butter(5, 100 / (sr / 2), btype='high')
        y = filtfilt(b, a, y)
    except Exception:
        pass
    return y


def trim_silence(y: np.ndarray, top_db: int = 20):
    try:
        y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        return y_trimmed
    except Exception:
        return y


def extract_mel_spectrogram(y: np.ndarray, sr: int) -> np.ndarray:
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
    mel_db = extract_mel_spectrogram(y, sr)
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr    = librosa.feature.zero_crossing_rate(y)
    rms    = librosa.feature.rms(y=y)
    sc     = librosa.feature.spectral_centroid(y=y, sr=sr)
    sb     = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    # F0：用 yin（比 pyin 快，不需要 numba）
    try:
        f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_mean = float(np.mean(f0[f0 > 0])) if np.any(f0 > 0) else 0.0
    except Exception:
        f0_mean = 0.0

    return {
        "mel_spectrogram":    mel_db,
        "mfcc_mean":          mfcc.mean(axis=1).tolist(),
        "zero_crossing_rate": float(zcr.mean()),
        "rms_energy":         float(rms.mean()),
        "spectral_centroid":  float(sc.mean()),
        "spectral_bandwidth": float(sb.mean()),
        "fundamental_freq_hz": f0_mean,
        "duration_sec":       len(y) / sr,
    }


def spectrogram_to_image_array(mel_db: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    from PIL import Image
    fig, ax = plt.subplots(figsize=(0.96, 0.96), dpi=100)
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
    fig, ax = plt.subplots(figsize=(8, 4))
    img = librosa.display.specshow(
        mel_db, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
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
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#0f172a')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def full_pipeline(source, from_bytes: bool = False) -> dict:
    if from_bytes:
        y, sr = load_audio_bytes(source)
    else:
        y, sr = load_audio(source)

    y = reduce_noise(y, sr)
    y = trim_silence(y)

    target_len = int(sr * DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    features = extract_features(y, sr)
    mel_db   = features["mel_spectrogram"]

    return {
        "features":        features,
        "cnn_input":       spectrogram_to_image_array(mel_db),
        "spectrogram_b64": spectrogram_to_base64(mel_db),
        "waveform_b64":    waveform_to_base64(y, sr),
    }
