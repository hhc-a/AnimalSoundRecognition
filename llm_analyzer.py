"""
llm_analyzer.py
規則式音訊特徵分析（不需要 API Key）
根據 F0、ZCR、RMS、頻譜重心等物理特徵輔助判斷
"""

# 各動物叫聲的典型特徵範圍
ANIMAL_PROFILES = {
    "dog": {
        "zh": "狗 🐕",
        "f0_range": (200, 1200),
        "zcr_range": (0.02, 0.15),
        "sc_range":  (500, 3000),
        "desc": "狗叫聲通常為週期性低中頻，基頻約 200–1200 Hz，過零率中等。",
        "tips": "狗的叫聲頻率因品種差異很大，小型犬音調較高，大型犬較低沉。"
    },
    "cat": {
        "zh": "貓 🐈",
        "f0_range": (500, 1500),
        "zcr_range": (0.05, 0.20),
        "sc_range":  (1000, 4000),
        "desc": "貓叫聲為中高頻，常帶有滑音特性，基頻約 500–1500 Hz。",
        "tips": "貓的喉嚨構造特殊，能同時發出吸氣與呼氣的呼嚕聲，頻率約 25–150 Hz。"
    },
    "bird": {
        "zh": "鳥 🐦",
        "f0_range": (1000, 8000),
        "zcr_range": (0.08, 0.40),
        "sc_range":  (2000, 8000),
        "desc": "鳥鳴聲為高頻，變化快速，過零率高，頻譜重心通常超過 2000 Hz。",
        "tips": "鳥類擁有鳴管（syrinx），能同時產生兩個獨立音源，創造複雜的鳴唱。"
    },
    "sheep": {
        "zh": "羊 🐑",
        "f0_range": (100, 600),
        "zcr_range": (0.01, 0.10),
        "sc_range":  (300, 2000),
        "desc": "羊叫聲為中低頻顫音，基頻約 100–600 Hz，過零率較低。",
        "tips": "羊能透過叫聲辨認彼此，母羊與小羊之間有獨特的叫聲溝通方式。"
    },
}

CONFIDENCE_LABEL = {
    "高": "高",
    "中": "中",
    "低": "低",
}


def analyze_with_llm(feature_description: str) -> dict:
    """
    根據音訊特徵字串解析數值，套用規則式分析輔助 CNN 結果。
    介面與原 LLM 版本相同，app.py 不需修改。
    """
    # 從特徵描述字串中擷取數值
    f0  = _parse_float(feature_description, "基頻 (F0)：", " Hz")
    zcr = _parse_float(feature_description, "過零率 (ZCR)：")
    sc  = _parse_float(feature_description, "頻譜重心：", " Hz")

    # 擷取 CNN 最高預測
    cnn_top = _parse_cnn_top(feature_description)

    # 對每個動物計算特徵符合分數
    scores = {}
    for animal, profile in ANIMAL_PROFILES.items():
        score = 0
        if f0 > 0:
            f0_lo, f0_hi = profile["f0_range"]
            if f0_lo <= f0 <= f0_hi:
                score += 2
            elif f0 < f0_lo * 0.7 or f0 > f0_hi * 1.3:
                score -= 1
        if zcr > 0:
            z_lo, z_hi = profile["zcr_range"]
            if z_lo <= zcr <= z_hi:
                score += 1
        if sc > 0:
            s_lo, s_hi = profile["sc_range"]
            if s_lo <= sc <= s_hi:
                score += 1
        scores[animal] = score

    # 找出規則分析最高分
    rule_top = max(scores, key=lambda k: scores[k])
    rule_score = scores[rule_top]

    # 綜合判斷：CNN + 規則
    if cnn_top and cnn_top == rule_top:
        final = cnn_top
        confidence = "高"
    elif cnn_top and rule_score <= 0:
        final = cnn_top   # 規則無法判斷，採用 CNN
        confidence = "中"
    elif rule_score >= 3:
        final = rule_top  # 規則信心強
        confidence = "中"
    else:
        final = cnn_top or rule_top
        confidence = "低"

    profile = ANIMAL_PROFILES.get(final, ANIMAL_PROFILES["dog"])

    # 產生特徵摘要說明
    f0_desc  = f"基頻 {f0:.0f} Hz" if f0 > 0 else "基頻無法偵測"
    zcr_desc = f"過零率 {zcr:.3f}（{'高' if zcr > 0.15 else '中' if zcr > 0.05 else '低'}）"
    sc_desc  = f"頻譜重心 {sc:.0f} Hz" if sc > 0 else ""

    reasoning = (
        f"音訊特徵：{f0_desc}、{zcr_desc}、{sc_desc}。"
        f"{profile['desc']}"
        f"規則分析得分：{', '.join(f'{a}={s}' for a,s in sorted(scores.items(), key=lambda x:-x[1]))}。"
    )

    return {
        "final_class":    final,
        "final_class_zh": profile["zh"],
        "confidence":     confidence,
        "summary":        f"規則分析判斷為「{profile['zh']}」，信心度：{confidence}",
        "reasoning":      reasoning,
        "tips":           profile["tips"],
        "llm_available":  True,
    }


# ─── 內部工具 ──────────────────────────────────────────────────────────────

def _parse_float(text: str, key: str, end: str = None) -> float:
    try:
        idx = text.index(key) + len(key)
        sub = text[idx:idx+20].strip()
        if end:
            sub = sub[:sub.index(end)] if end in sub else sub
        return float(sub.split()[0])
    except Exception:
        return 0.0


def _parse_cnn_top(text: str) -> str:
    """從特徵描述中找出 CNN 最高預測類別"""
    try:
        # 格式：最高信心度預測：狗 🐕（27.5%）
        marker = "最高信心度預測："
        idx = text.index(marker) + len(marker)
        segment = text[idx:idx+30]
        for animal in ["dog", "cat", "bird", "sheep"]:
            # 也從中文對照找
            pass
        # 從「前三名」行找英文 key
        for animal in ["dog", "cat", "bird", "sheep"]:
            from model import CLASS_ZH
            zh = CLASS_ZH[animal].split()[0]  # e.g. "狗"
            if zh in segment:
                return animal
    except Exception:
        pass
    return ""

