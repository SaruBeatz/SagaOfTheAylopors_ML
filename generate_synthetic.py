#!/usr/bin/env python3
"""Generate synthetic training CSV from chapter JSON and target profiles."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    ACCENTUATIONS,
    FEATURES,
    ML_DIR,
    l2_distance,
    moderate_target,
    parse_profile_final_vector,
    parse_target_profiles,
)
from story_graph import load_all_chapters, simulate_playthrough

DATA_DIR = ML_DIR / "data"
DEFAULT_CSV = DATA_DIR / "synthetic.csv"

VARIANTS_PER_CLASS = 40
MODERATE_ALPHAS = (0.6, 0.75, 0.85)
MIXED_FRACTION = 0.10
REJECT_TAU_MARGIN = 1.15


def calibrate_tau(targets: dict, chapters: dict, rng: random.Random) -> float:
    """Max final L2 on reference profile replays + margin."""
    max_dist = 0.0
    for name, target in targets.items():
        result = simulate_playthrough(
            chapters, target, rng, epsilon=0.0, delta_noise=0.0, final_noise=0.0
        )
        max_dist = max(max_dist, result.final_distance)
    for path in ML_DIR.glob("profile_*_playthrough.txt"):
        try:
            prof_name, final_vec = parse_profile_final_vector(path)
            if prof_name in targets:
                max_dist = max(max_dist, l2_distance(final_vec, targets[prof_name]))
        except ValueError:
            pass
    return max(REJECT_TAU_MARGIN, max_dist * 1.2)


def generate_rows(
    targets: dict,
    chapters: dict,
    rng: random.Random,
    variants_per_class: int,
    tau: float,
) -> list[dict]:
    rows: list[dict] = []
    path_counter = 0
    other_classes = list(ACCENTUATIONS)

    for label in ACCENTUATIONS:
        base_target = targets[label]
        n_moderate = variants_per_class // 4
        n_mixed = max(1, int(variants_per_class * MIXED_FRACTION))
        n_extreme = variants_per_class - n_moderate - n_mixed

        for v in range(variants_per_class):
            path_counter += 1
            path_id = f"{label}_{v:03d}"

            if v < n_extreme:
                steer_target = base_target.copy()
                variant = "extreme"
                row_label = label
                epsilon = 0.10 + rng.uniform(0, 0.05)
            elif v < n_extreme + n_moderate:
                alpha = MODERATE_ALPHAS[(v - n_extreme) % len(MODERATE_ALPHAS)]
                steer_target = moderate_target(base_target, alpha)
                variant = f"moderate_a{alpha}"
                row_label = label
                epsilon = 0.14 + rng.uniform(0, 0.06)
            else:
                variant = "mixed"
                other = rng.choice([c for c in other_classes if c != label])
                steer_target = 0.7 * base_target + 0.3 * targets[other]
                steer_target = np.clip(steer_target, 0.0, 1.0)
                row_label = label
                epsilon = 0.18 + rng.uniform(0, 0.08)

            result = simulate_playthrough(
                chapters,
                steer_target,
                rng,
                epsilon=epsilon,
                delta_noise=0.02,
                final_noise=0.03,
            )

            if result.final_distance > tau:
                continue

            def add_row(vector: np.ndarray, chapter_end: bool, ch: int | None):
                row = {f: float(vector[i]) for i, f in enumerate(FEATURES)}
                row["label"] = row_label
                row["path_id"] = path_id
                row["chapter_end"] = chapter_end
                row["variant"] = variant
                row["chapter"] = ch if ch is not None else 0
                row["final_l2"] = result.final_distance
                rows.append(row)

            for ch_id in sorted(result.chapter_snapshots.keys()):
                add_row(result.chapter_snapshots[ch_id], True, ch_id)
            add_row(result.final_state, False, 0)

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic ML dataset")
    parser.add_argument(
        "-o", "--output", type=Path, default=DEFAULT_CSV, help="Output CSV path"
    )
    parser.add_argument(
        "-n",
        "--variants",
        type=int,
        default=VARIANTS_PER_CLASS,
        help="Variants per accentuation class",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    targets = parse_target_profiles()
    missing = [a for a in ACCENTUATIONS if a not in targets]
    if missing:
        print(f"Missing targets: {missing}", file=sys.stderr)
        return 1

    chapters = load_all_chapters()
    if len(chapters) != 7:
        print(f"Expected 7 chapters, got {len(chapters)}", file=sys.stderr)
        return 1

    tau = calibrate_tau(targets, chapters, rng)
    print(f"Reject threshold tau (L2): {tau:.4f}")

    rows = generate_rows(targets, chapters, rng, args.variants, tau)
    if not rows:
        print("No rows generated", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8")

    n_paths = df["path_id"].nunique()
    print(f"Wrote {len(df)} rows ({n_paths} paths) -> {args.output}")
    print(df.groupby("label")["path_id"].nunique().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
