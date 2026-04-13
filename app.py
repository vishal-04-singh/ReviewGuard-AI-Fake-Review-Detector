"""
ReviewGuard – Flask API Server
Provides ML-powered review classification via REST endpoints.

Usage:
    python app.py

Endpoints:
    POST /classify      – Classify a single review
    POST /classify/bulk – Classify multiple reviews
    GET  /health        – Health check
    GET  /model/info    – Model metadata
"""

import os
import re
import json
import pickle
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow requests from the browser extension

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ReviewGuard")

# ---- Load model ----
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
META_PATH  = os.path.join(os.path.dirname(__file__), "model_meta.json")

model = None
model_meta = {}

def load_model():
    global model, model_meta
    if not os.path.exists(MODEL_PATH):
        logger.warning("Model not found. Run `python model_trainer.py` first.")
        return False
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            model_meta = json.load(f)
    logger.info("Model loaded. Accuracy: %.1f%%", model_meta.get("accuracy", 0) * 100)
    return True


# ---- NLP helpers ----
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s!?.,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_linguistic_features(text: str, rating: float = 0) -> dict:
    """Additional NLP-derived features returned alongside the ML prediction."""
    lower = text.lower()
    words = re.findall(r"\b\w+\b", lower)
    flags = []

    # Length check
    if len(words) < 8:
        flags.append("Very short review")
    
    # Caps ratio
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if caps_ratio > 0.35:
        flags.append("Excessive capitalisation")

    # Repetitive words
    from collections import Counter
    freq = Counter(words)
    most_common_count = freq.most_common(1)[0][1] if words else 0
    if most_common_count > 5 and len(words) < 40:
        flags.append(f"Repeated word: '{freq.most_common(1)[0][0]}'")

    # Spammy phrases
    spammy = [
        "best ever", "must buy", "100% recommend", "5 star",
        "best product ever", "perfect product", "highly recommend",
        "super quality", "amazing quality", "life changing"
    ]
    for phrase in spammy:
        if phrase in lower:
            flags.append(f'Spammy phrase: "{phrase}"')
            break

    # No personal pronouns
    personal = {"i", "my", "me", "we", "our", "myself", "bought", "ordered", "received"}
    if not any(w in personal for w in words):
        flags.append("No personal experience words")

    # Rating–sentiment mismatch
    neg_words = {"terrible","awful","horrible","worst","hate","useless","broke","broken","waste","scam","fake","defective"}
    has_neg = bool(neg_words & set(words))
    if has_neg and rating >= 4:
        flags.append("Negative language with high rating")

    # Spec dump
    spec_matches = re.findall(r"\d+\s*(gb|mb|mp|hz|mah|inch|cm|kg|watt)\b", lower)
    if len(spec_matches) > 3 and len(words) < 30:
        flags.append("Spec dump without personal opinion")

    return {"flags": flags, "word_count": len(words), "caps_ratio": round(caps_ratio, 3)}


# ---- Routes ----

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "version": "1.0.0"
    })


@app.route("/model/info", methods=["GET"])
def model_info():
    return jsonify(model_meta)


@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    text   = str(data.get("text", "")).strip()
    rating = float(data.get("rating", 0) or 0)

    if not text:
        return jsonify({"error": "Empty review text"}), 400

    # Fallback if model not loaded
    if model is None:
        return jsonify({"error": "Model not loaded. Run model_trainer.py first."}), 503

    # ML prediction
    try:
        proba     = model.predict_proba([text])[0]
        pred      = model.predict([text])[0]
        label     = "Suspicious" if pred == 1 else "Genuine"
        # confidence is how sure the model is of whichever class it picked
        confidence = float(max(proba) * 100)
    except Exception as e:
        logger.error("Prediction error: %s", e)
        return jsonify({"error": "Prediction failed"}), 500

    # Linguistic features
    ling = extract_linguistic_features(text, rating)

    # If ML is borderline (confidence < 60%), blend with heuristic flags
    if confidence < 60 and ling["flags"]:
        flag_boost = min(len(ling["flags"]) * 8, 20)
        if label == "Suspicious":
            confidence = min(confidence + flag_boost, 95)

    logger.info("Classified: %s (%.1f%%) | flags: %s", label, confidence, ling["flags"])

    return jsonify({
        "label":      label,
        "confidence": round(confidence, 1),
        "flags":      ling["flags"],
        "word_count": ling["word_count"],
        "method":     "ml"
    })


@app.route("/classify/bulk", methods=["POST"])
def classify_bulk():
    data = request.get_json(force=True, silent=True)
    if not data or "reviews" not in data:
        return jsonify({"error": "Provide 'reviews' array"}), 400

    reviews = data["reviews"]
    if not isinstance(reviews, list) or len(reviews) > 100:
        return jsonify({"error": "reviews must be an array of up to 100 items"}), 400

    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    results = []
    for rev in reviews:
        text   = str(rev.get("text", "")).strip()
        rating = float(rev.get("rating", 0) or 0)
        if not text:
            results.append({"error": "empty"})
            continue
        try:
            proba      = model.predict_proba([text])[0]
            pred       = model.predict([text])[0]
            label      = "Suspicious" if pred == 1 else "Genuine"
            confidence = float(max(proba) * 100)
            ling       = extract_linguistic_features(text, rating)
            results.append({
                "label": label, "confidence": round(confidence, 1),
                "flags": ling["flags"], "method": "ml"
            })
        except Exception as e:
            results.append({"error": str(e)})

    return jsonify({"results": results})


# ---- Entry point ----
if __name__ == "__main__":
    if not load_model():
        print("\n⚠️  Model not found. Training now…\n")
        from model_trainer import train_and_save
        train_and_save()
        load_model()
    print("\n🛡️  ReviewGuard API running on http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5001, debug=False)
