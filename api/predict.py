"""
Funzione serverless Vercel: carica model/model.joblib e predice la formulazione.

Endpoint: POST /api/predict
Body JSON atteso (le 14 feature del modello):
{
  "eta": 35, "sesso": "F", "fitzpatrick": "III",
  "elasticita": 0.70, "ph": 5.5, "idratazione": 55, "sebo": 60,
  "rossore": 0.20, "pigmentazione": 0.30, "rugosita": 0.30, "pori": 0.40,
  "sensibilita": 0.20, "acne": 1, "fotoaging": 3
}

Risposta:
{
  "formulazione_id": "F010",
  "compat": 88,
  "model": "Random Forest",
  "top3": [{"id": "F010", "prob": 0.24}, ...]
}
"""
import json
import os
from http.server import BaseHTTPRequestHandler

import joblib
import numpy as np

# Caricato UNA volta per container (riusato tra invocazioni a caldo).
_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_HERE, "..", "model", "model.joblib")
_ARTIFACT = joblib.load(_MODEL_PATH)


def _build_vector(payload):
    """Costruisce il vettore feature nell'ordine atteso dal modello."""
    a = _ARTIFACT
    # encoding categoriche, coerente con export_model.py
    sesso = str(payload.get("sesso", "")).strip()
    sesso_enc = a["sesso_classes"].index(sesso) if sesso in a["sesso_classes"] else 0
    fitz = str(payload.get("fitzpatrick", "")).strip()
    fitz_enc = a["fitzpatrick_map"].get(fitz, 3)  # default III se mancante

    encoded = {**{k: payload.get(k, 0) for k in payload}, "sesso": sesso_enc, "fitzpatrick": fitz_enc}
    row = []
    for f in a["feature_order"]:
        try:
            row.append(float(encoded.get(f, 0) or 0))
        except (TypeError, ValueError):
            row.append(0.0)
    return np.array([row])


def _predict(payload):
    a = _ARTIFACT
    X = a["scaler"].transform(_build_vector(payload))
    classes = [str(c) for c in a["model"].classes_]

    if a.get("has_proba"):
        proba = a["model"].predict_proba(X)[0]
        ranked = sorted(zip(classes, proba), key=lambda x: -x[1])
        fid = ranked[0][0]
        top3 = [{"id": c, "prob": round(float(p), 3)} for c, p in ranked[:3]]
        compat = int(round(ranked[0][1] * 100))
    else:
        fid = str(a["model"].predict(X)[0])
        top3 = [{"id": fid, "prob": 1.0}]
        compat = None

    return {"formulazione_id": fid, "compat": compat,
            "model": a.get("model_name", "model"), "top3": top3}


class handler(BaseHTTPRequestHandler):
    def _send(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode("utf-8"))

    def do_OPTIONS(self):
        self._send(204, {})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            self._send(200, _predict(payload))
        except Exception as e:  # noqa: BLE001 — superficie minima, errore tornato al client
            self._send(400, {"error": str(e)})

    def do_GET(self):
        # healthcheck rapido
        self._send(200, {"status": "ok", "model": _ARTIFACT.get("model_name"),
                         "cv_accuracy": _ARTIFACT.get("cv_accuracy")})
