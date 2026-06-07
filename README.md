# 動物聲音辨識系統 🐾
### Deep Learning × Bioacoustics — CNN 深度學習辨識

---

## 專案架構

```
animal_sound_classifier/
├── app.py               # Flask 後端主程式
├── audio_processor.py   # 音訊前處理、頻譜圖、特徵萃取
├── model.py             # CNN 模型定義與推理
├── llm_analyzer.py      # 規則式音訊特徵分析
├── train_model.py       # 模型訓練腳本（含音訊增強）
├── requirements.txt     # 套件清單
├── templates/
│   └── index.html       # 前端介面
└── data/                # 放訓練資料
    ├── dog/
    ├── cat/
    ├── bird/
    └── sheep/
```

---

## 深度學習架構說明

### CNN（MobileNetV2 Transfer Learning）
- **Backbone**：MobileNetV2 預訓練 ImageNet 權重
- **輸入**：224×224 Mel 頻譜圖（magma 色彩映射）
- **分類頭**：GlobalAvgPool → Dense(256) + BN + Dropout(0.5) → Dense(128) + BN + Dropout(0.3) → Softmax
- **訓練策略**：兩階段（先凍結 backbone 訓練分類頭 → 再解凍後 50 層微調）

### 音訊資料增強（每類音檔擴充 6 倍）
| 方法 | 模擬情境 |
|------|---------|
| 加白噪音 | 有背景雜音的環境 |
| 時間位移 | 叫聲出現在不同時間點 |
| 音調偏移 | 同動物個體差異（±4 半音） |
| 時間拉伸 | 叫聲速度快慢不同 |
| 音量縮放 | 遠近不同的錄音距離 |

### 音訊特徵
| 特徵 | 說明 |
|------|------|
| Mel Spectrogram | 模擬人耳頻率感知的頻譜圖 |
| MFCC (13維) | 語音/聲音最常用特徵 |
| F0 基頻 | 動物叫聲的基本音高 |
| ZCR 過零率 | 聲音的尖銳程度 |
| RMS 能量 | 音量大小 |
| 頻譜重心 | 頻率分布重心 |

---

## 安裝步驟

### 1. 建立虛擬環境
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 2. 安裝套件
```bash
pip install -r requirements.txt
```

### 3. 安裝 ffmpeg（處理瀏覽器錄音格式）
```bash
# Windows（用 Chocolatey）
choco install ffmpeg
# 或直接下載：https://ffmpeg.org/download.html → 解壓後加入 PATH

# Mac
brew install ffmpeg

# Linux
sudo apt install ffmpeg
```

> **Mac M1/M2/M3 用戶**：需將 requirements.txt 中的 `tensorflow` 改為 `tensorflow-metal`

---

## 準備訓練資料

推薦資料集（免費）：
- **Kaggle Animal Sounds**：https://www.kaggle.com/datasets/caoofficial/animal-sounds
- **Freesound**：https://freesound.org

將音檔放入對應資料夾，每類至少 **50 個音檔**（訓練腳本會自動增強到約 180 張）。

---

## 訓練模型

```bash
python train_model.py
```

訓練完成後會產生：
- `animal_cnn_model.keras` — 訓練好的模型
- `classes.json` — 類別清單

> 若已有舊模型需重新訓練，請先刪除 `animal_cnn_model.keras` 再執行

---

## 啟動系統

```bash
python app.py
```

開啟瀏覽器：**http://localhost:5000**

---

## 使用方法

1. **上傳音檔**：點擊上傳區或拖放 WAV / MP3 / OGG / FLAC / M4A
2. **即時錄音**：點擊「開始錄音」，對麥克風發出動物聲音，再點「停止錄音」
3. **開始分析**：點擊「開始分析」
4. 系統顯示：
   - CNN 辨識動物結果
   - Mel 頻譜圖與波形圖
   - F0 基頻、RMS 能量、過零率、頻譜重心等音訊特徵數值

---

## 支援動物類別

| 類別 | 中文 | 叫聲特徵 |
|------|------|---------|
| dog  | 狗 🐕 | 低頻、週期性、400–1000 Hz |
| cat  | 貓 🐈 | 中頻、滑音、700–1500 Hz |
| bird | 鳥 🐦 | 高頻、快速變化、2000–8000 Hz |
| sheep| 羊 🐑 | 中低頻、顫音、200–600 Hz |

---

1. 刪除舊的 `animal_cnn_model.keras`
2. 重新執行 `python train_model.py`
