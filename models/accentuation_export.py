"""
Standalone accentuation predictor (exported from Saga Ailoporosa ml/).
Requires: numpy

Usage:
    from accentuation_export import predict
    result = predict({"Soc": 0.5, "Act": 0.5, ...})
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_DATA_PATH = Path(__file__).with_name("accentuation_export.json")


def _load():
    with _DATA_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _softmax(scores: np.ndarray) -> np.ndarray:
    s = scores - scores.max()
    e = np.exp(s)
    return e / e.sum()


def predict(stats: dict, top_k: int = 3) -> dict:
    cfg = _load()
    features = cfg["features"]
    x = np.array([float(stats[k]) for k in features], dtype=np.float64)
    mean = np.array(cfg["scaler_mean"], dtype=np.float64)
    scale = np.array(cfg["scaler_scale"], dtype=np.float64)
    x_scaled = (x - mean) / scale
    coef = np.array(cfg["coefficients"], dtype=np.float64)
    intercept = np.array(cfg["intercept"], dtype=np.float64)
    scores = coef @ x_scaled + intercept
    probs = _softmax(scores)
    classes = cfg["classes"]
    ranked = sorted(zip(classes, probs.tolist()), key=lambda t: -t[1])
    pred = ranked[0][0]
    return {
        "predicted": pred,
        "confidence": float(ranked[0][1]),
        "top_k": [{"label": l, "probability": float(p)} for l, p in ranked[:top_k]],
        "features": {k: float(stats[k]) for k in features},
        "model_name": cfg.get("model_name"),
    }


if __name__ == "__main__":
    import sys
    vec = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {
        "Soc": 0.5, "Act": 0.5, "Emp": 0.5, "Anx": 0.5, "Ctrl": 0.5,
        "Imp": 0.5, "Ego": 0.5, "Rig": 0.5, "Neg": 0.5, "Adp": 0.5,
    }
    print(json.dumps(predict(vec), ensure_ascii=False, indent=2))
