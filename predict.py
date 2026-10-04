#!/usr/bin/env python3
"""
Load the trained accentuation model and predict from a 10-parameter vector.

Examples:
  python predict.py --profile ml/profile_Тревожный_playthrough.txt
  python predict.py --json '{"Soc":0.36,"Act":0.22,"Emp":0.49,"Anx":1,"Ctrl":1,"Imp":0.31,"Ego":0.88,"Rig":0.75,"Neg":0.82,"Adp":0.58}'
  python predict.py --csv-row ml/data/synthetic.csv --index 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common import FEATURES, ML_DIR, dict_to_vector, parse_profile_final_vector

DEFAULT_MODEL = ML_DIR / "models" / "accentuation_classifier.joblib"


def load_bundle(model_path: Path | None = None) -> dict:
    path = model_path or DEFAULT_MODEL
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\nRun: python train_compare.py"
        )
    return joblib.load(path)


def predict_accentuation(
    bundle: dict,
    stats: dict[str, float] | np.ndarray,
    top_k: int = 3,
) -> dict:
    """
    Predict dominant accentuation and top-k probabilities.

    stats: dict with keys Soc..Adp or length-10 array in FEATURES order.
    Returns JSON-serializable dict.
    """
    if isinstance(stats, dict):
        vector = dict_to_vector(stats)
    else:
        vector = np.asarray(stats, dtype=np.float64).reshape(-1)
    if vector.shape[0] != len(FEATURES):
        raise ValueError(f"Expected {len(FEATURES)} features, got {vector.shape[0]}")

    model = bundle["model"]
    le = bundle["label_encoder"]
    X = vector.reshape(1, -1)

    pred = str(model.predict(X)[0])
    result = {
        "predicted": pred,
        "features": {FEATURES[i]: float(vector[i]) for i in range(len(FEATURES))},
        "model_name": bundle.get("model_name", "unknown"),
    }

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = list(le.classes_)
        ranked = sorted(
            zip(classes, [float(p) for p in proba]),
            key=lambda x: -x[1],
        )
        result["confidence"] = float(dict(ranked)[pred])
        result["top_k"] = [
            {"label": label, "probability": prob}
            for label, prob in ranked[:top_k]
        ]
        result["all_probabilities"] = {label: prob for label, prob in ranked}
    else:
        result["confidence"] = 1.0
        result["top_k"] = [{"label": pred, "probability": 1.0}]

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict accentuation from stats vector")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--json", type=str, help='Stats JSON, e.g. {"Soc":0.5,...}')
    parser.add_argument("--profile", type=Path, help="profile_*_playthrough.txt (uses final vector)")
    parser.add_argument("--csv-row", type=Path, help="Read one row from synthetic.csv")
    parser.add_argument("--index", type=int, default=0, help="Row index with --csv-row")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--expected", type=str, help="Optional label to check (exit 2 if mismatch)")
    args = parser.parse_args()

    bundle = load_bundle(args.model)

    if args.profile:
        expected_name, vector = parse_profile_final_vector(args.profile)
        stats = {FEATURES[i]: float(vector[i]) for i in range(len(FEATURES))}
        if not args.expected:
            args.expected = expected_name
    elif args.json:
        stats = json.loads(args.json)
    elif args.csv_row:
        df = pd.read_csv(args.csv_row)
        row = df.iloc[args.index]
        stats = {f: float(row[f]) for f in FEATURES}
        if not args.expected and "label" in row:
            args.expected = str(row["label"])
    else:
        print("Provide --json, --profile, or --csv-row", file=sys.stderr)
        return 1

    out = predict_accentuation(bundle, stats, top_k=args.top_k)
    print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.expected:
        ok = out["predicted"] == args.expected
        print(f"\nExpected: {args.expected}  ->  {'OK' if ok else 'MISMATCH'}", file=sys.stderr)
        return 0 if ok else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
