"""
app.py
Flask 後端主程式
提供：
  POST /api/predict  - 上傳音檔或錄音 bytes，回傳辨識結果
  GET  /             - 前端介面
"""

import os
import json
import tempfile
import traceback
from flask import Flask, request, jsonify, render_template, send_from_directory

# ─── 記憶體優化（Render 免費方案）────────────────────────────────────────
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'          # 關閉 TF 詳細 log
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from audio_processor import full_pipeline
from model import load_model, predict_from_array, CLASSES, CLASS_ZH

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB 上限

# ─── 全域載入模型（啟動時一次）─────────────────────────────────────────
print("[app] 載入 CNN 模型中...")
MODEL_FILE = "animal_cnn_model.h5"
if not os.path.exists(MODEL_FILE):
    print(f"[警告] 找不到 {MODEL_FILE}，請將訓練好的模型上傳至 GitHub")
    cnn_model = None
else:
    cnn_model = load_model()
    print("[app] 模型載入完成")


# ─── 路由 ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    接受：
      - 表單上傳音檔：request.files['audio']
      - raw bytes（錄音）：request.data + header Content-Type: audio/wav
    回傳 JSON：
    {
      "success": true,
      "cnn": { top_class, top_class_zh, top_prob, all_probs },
      "llm": { summary, confidence, reasoning },
      "spectrogram": "<base64 PNG>",
      "waveform":    "<base64 PNG>",
    }
    """
    try:
        audio_bytes = None

        # 判斷來源
        if 'audio' in request.files:
            f = request.files['audio']
            audio_bytes = f.read()
            from_bytes = True
        elif request.data:
            audio_bytes = request.data
            from_bytes = True
        else:
            return jsonify({"success": False, "error": "未收到音訊資料"}), 400

        # 音訊處理
        pipeline_result = full_pipeline(audio_bytes, from_bytes=True)
        features  = pipeline_result["features"]
        cnn_input = pipeline_result["cnn_input"]
        spec_b64  = pipeline_result["spectrogram_b64"]
        wave_b64  = pipeline_result["waveform_b64"]

        # CNN 推理
        if cnn_model is None:
            return jsonify({"success": False, "error": "模型尚未載入，請確認 animal_cnn_model.h5 已上傳至 GitHub"}), 500
        cnn_result = predict_from_array(cnn_model, cnn_input)

        return jsonify({
            "success":      True,
            "cnn":          cnn_result,
            "spectrogram":  spec_b64,
            "waveform":     wave_b64,
            "features": {
                "f0_hz":      features["fundamental_freq_hz"],
                "rms":        features["rms_energy"],
                "zcr":        features["zero_crossing_rate"],
                "sc_hz":      features["spectral_centroid"],
                "duration_s": features["duration_sec"],
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/classes')
def get_classes():
    """回傳支援的動物類別列表"""
    return jsonify({
        "classes": [
            {"id": cls, "label": CLASS_ZH[cls]}
            for cls in CLASSES
        ]
    })


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
