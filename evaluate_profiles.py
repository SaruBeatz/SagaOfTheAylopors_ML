#!/usr/bin/env python3
"""Sanity-check: predict accentuation on 10 reference profile playthrough finals."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

from common import FEATURES, ML_DIR, parse_profile_final_vector

MODELS_DIR = ML_DIR / "models"
REPORTS_DIR = ML_DIR / "reports"
DEFAULT_MODEL = MODELS_DIR / "accentuation_classifier.joblib"


def predict(model_bundle: dict, vector: np.ndarray) -> tuple[str, float, dict]:
    model = model_bundle["model"]
    le = model_bundle["label_encoder"]
    X = vector.reshape(1, -1)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        idx = int(np.argmax(proba))
        pred = le.classes_[idx] if hasattr(le, "classes_") else model.classes_[idx]
        conf = float(proba[idx])
        dist = {le.classes_[i]: float(proba[i]) for i in range(len(proba))}
    else:
        pred = model.predict(X)[0]
        conf = 1.0
        dist = {str(pred): 1.0}
    return str(pred), conf, dist


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "-o", "--output", type=Path, default=REPORTS_DIR / "profile_sanity.json"
    )
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Model not found: {args.model}. Run train_compare.py first.", file=sys.stderr)
        return 1

    bundle = joblib.load(args.model)
    profiles = sorted(ML_DIR.glob("profile_*_playthrough.txt"))
    if len(profiles) != 10:
        print(f"Warning: expected 10 profile files, found {len(profiles)}", file=sys.stderr)

    rows = []
    ok = 0
    for path in profiles:
        expected, vector = parse_profile_final_vector(path)
        pred, conf, proba = predict(bundle, vector)
        match = pred == expected
        if match:
            ok += 1
        rows.append(
            {
                "file": path.name,
                "expected": expected,
                "predicted": pred,
                "confidence": conf,
                "match": match,
                "vector": {FEATURES[i]: float(vector[i]) for i in range(len(FEATURES))},
                "top3": dict(
                    sorted(proba.items(), key=lambda x: -x[1])[:3]
                ),
            }
        )
        status = "OK" if match else "MISS"
        print(f"[{status}] {expected:18s} -> {pred:18s} ({conf:.2%})  {path.name}")

    report = {
        "model": bundle.get("model_name", "unknown"),
        "passed": ok,
        "total": len(rows),
        "all_match": ok == len(rows),
        "profiles": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSanity: {ok}/{len(rows)} profiles matched -> {args.output}")

    # Merge into metrics.json if present
    metrics_path = REPORTS_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["profile_sanity"] = report
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return 0 if ok == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
