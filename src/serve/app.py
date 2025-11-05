import os, time, joblib, numpy as np
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "deployed_model/model.pkl")
FALLBACK = "artifacts/model.pkl"
if not os.path.exists(MODEL_PATH):
    if os.path.exists(FALLBACK):
        MODEL_PATH = FALLBACK
    else:
        raise FileNotFoundError(f"Model not found at {MODEL_PATH} or {FALLBACK}")
model = joblib.load(MODEL_PATH)


REQUESTS = Counter("api_requests_total", "Total API requests")
ERRORS = Counter("api_request_errors_total", "Total API errors")
PREDICTIONS = Counter("model_predictions_total", "Total predictions served")
LATENCY = Histogram("api_request_latency_seconds", "Request latency in seconds")
CONFIDENCE = Histogram("model_prediction_confidence", "Max predicted class probability")

@app.route("/predict", methods=["POST"])
def predict():
    start = time.time()
    REQUESTS.inc()
    try:
        data = request.get_json(force=True)
        features = data.get("features")
        if features is None:
            ERRORS.inc()
            return jsonify({"error": "JSON must include 'features' (13-length list)."}), 400

        # Wine has 13 numerical features. Enforce shape.
        x = np.array(features, dtype=float).reshape(1, -1)
        pred = int(model.predict(x)[0])

        conf = None
        # Many classifiers support predict_proba; if available, record confidence
        try:
            proba = model.predict_proba(x)[0]
            conf = float(np.max(proba))
            CONFIDENCE.observe(conf)
        except Exception:
            pass

        PREDICTIONS.inc()
        LATENCY.observe(time.time() - start)
        return jsonify({"prediction": pred, "confidence": conf})
    except Exception as e:
        ERRORS.inc()
        return jsonify({"error": str(e)}), 500

@app.route("/healthz")
def health():
    return "ok", 200

@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
