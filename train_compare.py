#!/usr/bin/env python3
"""Train and compare accentuation classifiers; save winner and reports."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

from common import ACCENTUATIONS, FEATURES, ML_DIR, parse_target_profiles

DATA_DIR = ML_DIR / "data"
REPORTS_DIR = ML_DIR / "reports"
MODELS_DIR = ML_DIR / "models"
DEFAULT_CSV = DATA_DIR / "synthetic.csv"


class NearestCentroidClassifier(BaseEstimator, ClassifierMixin):
    """Predict class by minimum L2 distance to target centroids."""

    def __init__(self, centroids: dict | None = None):
        self.centroids = centroids or {}
        self.classes_ = np.array(ACCENTUATIONS)

    def fit(self, X, y):
        self.classes_ = np.array(sorted(set(y)))
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        k = len(self.classes_)
        dists = np.zeros((n, k), dtype=np.float64)
        for j, cls in enumerate(self.classes_):
            c = self.centroids.get(cls)
            if c is None:
                dists[:, j] = np.inf
            else:
                dists[:, j] = np.linalg.norm(X - c, axis=1)
        # Convert distances to pseudo-probabilities (closer = higher)
        inv = 1.0 / (dists + 1e-6)
        row_sums = inv.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return inv / row_sums


def load_dataset(csv_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    for col in FEATURES + ["label", "path_id"]:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in {csv_path}")
    X = df[FEATURES].values.astype(np.float64)
    y = df["label"].values
    groups = df["path_id"].values
    return df, X, y, groups


def build_models(centroids: dict[str, np.ndarray]) -> dict:
    return {
        "nearest_centroid": NearestCentroidClassifier(centroids),
        "knn_5": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance")),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs"),
                ),
            ]
        ),
        "linear_svm_calibrated": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    CalibratedClassifierCV(
                        LinearSVC(max_iter=3000, C=0.5, dual="auto"),
                        cv=3,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=6,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=5,
            learning_rate=0.08,
            max_iter=200,
            random_state=42,
        ),
        "decision_tree_interpret": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=42,
        ),
    }


def evaluate_model(name, model, X, y, groups, le: LabelEncoder) -> dict:
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("Need at least 2 unique path_id groups for CV")
    gkf = GroupKFold(n_splits=n_splits)
    y_pred = cross_val_predict(model, X, y, groups=groups, cv=gkf, method="predict")
    try:
        y_proba = cross_val_predict(
            model, X, y, groups=groups, cv=gkf, method="predict_proba"
        )
        ll = log_loss(y, y_proba, labels=le.classes_)
    except Exception:
        y_proba = None
        ll = None

    return {
        "model": name,
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro")),
        "log_loss": float(ll) if ll is not None else None,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "confusion_matrix": confusion_matrix(y, y_pred, labels=le.classes_).tolist(),
        "classification_report": classification_report(y, y_pred, labels=le.classes_),
    }


def plot_confusion(cm: np.ndarray, labels: list, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=7,
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"Dataset not found: {args.csv}. Run generate_synthetic.py first.", file=sys.stderr)
        return 1

    df, X, y, groups = load_dataset(args.csv)
    le = LabelEncoder()
    le.fit(ACCENTUATIONS)
    y_enc = le.transform(y)

    targets = parse_target_profiles()
    centroids = {k: targets[k] for k in ACCENTUATIONS if k in targets}

    models = build_models(centroids)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    best_name = None
    best_f1 = -1.0
    best_model = None
    best_detail = None

    for name, model in models.items():
        print(f"Evaluating {name}...")
        detail = evaluate_model(name, model, X, y, groups, le)
        summary = {k: detail[k] for k in ("model", "accuracy", "macro_f1", "log_loss")}
        results.append(summary)
        print(f"  accuracy={summary['accuracy']:.4f} macro_f1={summary['macro_f1']:.4f}")

        cm = np.array(detail["confusion_matrix"])
        plot_confusion(
            cm,
            list(le.classes_),
            f"Confusion: {name}",
            REPORTS_DIR / f"confusion_{name}.png",
        )

        if summary["macro_f1"] > best_f1:
            best_f1 = summary["macro_f1"]
            best_name = name
            best_model = model
            best_detail = detail

    # Fit winner on full data
    best_model.fit(X, y)
    winner_path = MODELS_DIR / "accentuation_classifier.joblib"
    joblib.dump(
        {
            "model": best_model,
            "model_name": best_name,
            "label_encoder": le,
            "features": FEATURES,
            "centroids": centroids,
        },
        winner_path,
    )

    metrics = {
        "dataset_rows": len(df),
        "unique_paths": int(df["path_id"].nunique()),
        "comparison": results,
        "winner": {
            "name": best_name,
            "macro_f1_cv": best_f1,
            "accuracy_cv": best_detail["accuracy"],
            "log_loss_cv": best_detail["log_loss"],
            "model_path": str(winner_path),
        },
        "winner_classification_report": best_detail["classification_report"],
    }
    metrics_path = REPORTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Model comparison (GroupKFold CV) ===")
    for r in sorted(results, key=lambda x: -x["macro_f1"]):
        print(
            f"{r['model']:28s}  acc={r['accuracy']:.4f}  "
            f"macro_f1={r['macro_f1']:.4f}  log_loss={r['log_loss']}"
        )
    print(f"\nWinner: {best_name} (macro_f1={best_f1:.4f})")
    print(f"Saved model -> {winner_path}")
    print(f"Metrics -> {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
