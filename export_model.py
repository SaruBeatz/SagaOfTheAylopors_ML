#!/usr/bin/env python3
"""
Export trained model for use outside this repo (no sklearn required for inference).

Creates:
  models/accentuation_export.json  — scaler + logistic weights + class names
  models/accentuation_export.py    — standalone predictor (stdlib + numpy only)

For full sklearn pipeline (any winner type), copy:
  models/accentuation_classifier.joblib
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import FEATURES, ML_DIR

DEFAULT_MODEL = ML_DIR / "models" / "accentuation_classifier.joblib"
EXPORT_JSON = ML_DIR / "models" / "accentuation_export.json"
EXPORT_PY = ML_DIR / "models" / "accentuation_export.py"


def extract_logistic_export(bundle: dict) -> dict:
    model = bundle["model"]
    if not isinstance(model, Pipeline):
        raise TypeError(f"Expected sklearn Pipeline, got {type(model)}")
    scaler = model.named_steps.get("scaler")
    clf = model.named_steps.get("clf")
    if not isinstance(scaler, StandardScaler) or not isinstance(clf, LogisticRegression):
        raise TypeError(
            "JSON export supports logistic_regression Pipeline only. "
            f"Got scaler={type(scaler)}, clf={type(clf)}. "
            "Use accentuation_classifier.joblib for other model types."
        )

    le = bundle["label_encoder"]
    classes = [str(c) for c in le.classes_]

    return {
        "format_version": 1,
        "model_name": bundle.get("model_name", "logistic_regression"),
        "features": FEATURES,
        "classes": classes,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "inference": (
            "1) x_scaled[i] = (x[i] - scaler_mean[i]) / scaler_scale[i]\n"
            "2) scores[k] = sum_j(coefficients[k][j] * x_scaled[j]) + intercept[k]\n"
            "3) probs = softmax(scores); predicted = argmax(probs)"
        ),
    }


def write_standalone_py(export_data: dict, out_path: Path) -> None:
    code = textwrap.dedent(
        '''\
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
        '''
    )
    out_path.write_text(code, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export model for other projects")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--json-out", type=Path, default=EXPORT_JSON)
    parser.add_argument("--py-out", type=Path, default=EXPORT_PY)
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    data = extract_logistic_export(bundle)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_standalone_py(data, args.py_out)

    print(f"Exported JSON -> {args.json_out}")
    print(f"Exported standalone module -> {args.py_out}")
    print("\nCopy to another project:")
    print(f"  - {args.json_out.name}")
    print(f"  - {args.py_out.name}")
    print("  - pip install numpy")
    print("\nOr use full sklearn bundle:")
    print(f"  - {args.model.name}  (+ pip install scikit-learn joblib)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
