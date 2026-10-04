#!/usr/bin/env python3
"""Charts for thesis presentation from reports/metrics.json (+ optional winner CM)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ML_DIR = Path(__file__).resolve().parent
REPORTS_DIR = ML_DIR / "reports"
METRICS_PATH = REPORTS_DIR / "metrics.json"

# App-inspired colors for slides
BG = "#0a0e27"
PARCHMENT = "#E8DCC8"
BROWN = "#5D4037"
GOLD = "#D4AF37"
ORANGE = "#F39C12"
CYAN = "#A8DADC"
HIGHLIGHT = "#D35400"

MODEL_LABELS = {
    "nearest_centroid": "Nearest centroid",
    "knn_5": "k-NN (k=5)",
    "logistic_regression": "Logistic Regression",
    "linear_svm_calibrated": "Linear SVM",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Hist. GB",
    "decision_tree_interpret": "Decision tree",
}


def _style_axes(ax, title: str) -> None:
    ax.set_facecolor("#1a1d3a")
    ax.figure.patch.set_facecolor(BG)
    ax.set_title(title, color=PARCHMENT, fontsize=14, pad=12)
    ax.tick_params(colors=PARCHMENT, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(BROWN)


def plot_model_comparison(metrics: dict, out_path: Path, metric: str = "macro_f1") -> None:
    rows = sorted(metrics["comparison"], key=lambda r: r[metric])
    names = [MODEL_LABELS.get(r["model"], r["model"]) for r in rows]
    values = [r[metric] for r in rows]
    winner = metrics["winner"]["name"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = [ORANGE if rows[i]["model"] == winner else CYAN for i in range(len(rows))]
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color=colors, edgecolor=BROWN, linewidth=0.8, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(
        "Macro F1 (GroupKFold CV)" if metric == "macro_f1" else "Accuracy (GroupKFold CV)",
        color=PARCHMENT,
    )
    _style_axes(
        ax,
        "Сравнение моделей" if metric == "macro_f1" else "Точность моделей (accuracy)",
    )
    for bar, val in zip(bars, values):
        ax.text(
            val + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            ha="left",
            color=PARCHMENT,
            fontsize=10,
        )
    note = (
        f"Датасет: {metrics['dataset_rows']} строк, "
        f"{metrics['unique_paths']} уникальных путей · победитель: "
        f"{MODEL_LABELS.get(winner, winner)}"
    )
    fig.text(0.5, 0.02, note, ha="center", color=GOLD, fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=150, facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_winner_confusion(metrics_path: Path) -> int:
    """Regenerate confusion matrix for CV winner only (needs dataset + sklearn)."""
    from train_compare import (  # noqa: PLC0415
        DEFAULT_CSV,
        build_models,
        evaluate_model,
        load_dataset,
        plot_confusion,
    )
    from sklearn.preprocessing import LabelEncoder
    from common import ACCENTUATIONS

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    winner = metrics["winner"]["name"]
    if not DEFAULT_CSV.exists():
        print(f"Dataset missing: {DEFAULT_CSV}. Run generate_synthetic.py first.", flush=True)
        return 1

    _, X, y, groups = load_dataset(DEFAULT_CSV)
    le = LabelEncoder()
    le.fit(ACCENTUATIONS)
    from common import parse_target_profiles

    targets = parse_target_profiles()
    centroids = {k: targets[k] for k in ACCENTUATIONS if k in targets}
    models = build_models(centroids)
    model = models[winner]
    detail = evaluate_model(winner, model, X, y, groups, le)
    cm = np.array(detail["confusion_matrix"])
    out = REPORTS_DIR / f"confusion_{winner}.png"
    plot_confusion(cm, list(le.classes_), f"Confusion matrix: {winner} (CV)", out)
    # Presentation-friendly copy
    pres_out = REPORTS_DIR / "presentation_confusion_matrix.png"
    plot_confusion(
        cm,
        list(le.classes_),
        "Матрица ошибок — Logistic Regression (CV)",
        pres_out,
    )
    print(f"Saved {out} and {pres_out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=Path,
        default=METRICS_PATH,
        help="Path to metrics.json",
    )
    parser.add_argument(
        "--confusion",
        action="store_true",
        help="Re-run CV for winner and save confusion matrix PNG",
    )
    args = parser.parse_args()

    if not args.metrics.exists():
        print(f"Not found: {args.metrics}. Run train_compare.py first.", flush=True)
        return 1

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))

    plot_model_comparison(
        metrics,
        REPORTS_DIR / "presentation_model_comparison_f1.png",
        metric="macro_f1",
    )
    plot_model_comparison(
        metrics,
        REPORTS_DIR / "presentation_model_comparison_accuracy.png",
        metric="accuracy",
    )

    if args.confusion:
        return plot_winner_confusion(args.metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
