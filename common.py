"""Shared constants and vector helpers for accentuation ML."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ML_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ML_DIR.parent
ASSETS_DIR = ML_DIR / "data" / "chapters"

FEATURES: List[str] = [
    "Soc", "Act", "Emp", "Anx", "Ctrl", "Imp", "Ego", "Rig", "Neg", "Adp"
]

JSON_EFFECT_TO_FEATURE = {
    "sociality": "Soc",
    "activity": "Act",
    "emotional_sensitivity": "Emp",
    "anxiety": "Anx",
    "self_control": "Ctrl",
    "impulsivity": "Imp",
    "ego_focus": "Ego",
    "rigidity": "Rig",
    "negative_affect": "Neg",
    "adaptability": "Adp",
}

SHORT_TO_JSON = {v: k for k, v in JSON_EFFECT_TO_FEATURE.items()}

ACCENTUATIONS: List[str] = [
    "Гипертим",
    "Возбудимый",
    "Эмотив",
    "Педантичный",
    "Тревожный",
    "Циклотим",
    "Демонстративный",
    "Неуравновешенный",
    "Дистим",
    "Экзальтированный",
]


def neutral_vector() -> np.ndarray:
    return np.full(len(FEATURES), 0.5, dtype=np.float64)


def clamp_vector(v: np.ndarray) -> np.ndarray:
    return np.clip(v, 0.0, 1.0)


def dict_to_vector(d: Dict[str, float]) -> np.ndarray:
    return np.array([float(d[k]) for k in FEATURES], dtype=np.float64)


def vector_to_dict(v: np.ndarray) -> Dict[str, float]:
    return {k: float(v[i]) for i, k in enumerate(FEATURES)}


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def parse_target_profiles(path: Path | None = None) -> Dict[str, np.ndarray]:
    path = path or ML_DIR / "ml_target_profiles.txt"
    text = path.read_text(encoding="utf-8")
    targets: Dict[str, np.ndarray] = {}
    current: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"\[(.+?)\]", line)
        if m:
            current = m.group(1).strip()
            continue
        if current and line.startswith("{"):
            targets[current] = dict_to_vector(json.loads(line))
    return targets


def moderate_target(target: np.ndarray, alpha: float) -> np.ndarray:
    """Blend target toward neutral: 0.5 + alpha * (target - 0.5)."""
    return clamp_vector(0.5 + alpha * (target - 0.5))


def effects_json_to_delta(effects: dict | None) -> np.ndarray:
    delta = np.zeros(len(FEATURES), dtype=np.float64)
    if not effects:
        return delta
    for json_key, val in effects.items():
        feat = JSON_EFFECT_TO_FEATURE.get(json_key)
        if feat:
            delta[FEATURES.index(feat)] = float(val)
    return delta


def apply_delta(state: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return clamp_vector(state + delta)


def parse_profile_final_vector(path: Path) -> Tuple[str, np.ndarray]:
    """Extract accentuation name and final vector from profile_*_playthrough.txt."""
    text = path.read_text(encoding="utf-8")
    name = ""
    m_name = re.search(r"#\s*Профиль:\s*(.+)", text)
    if m_name:
        name = m_name.group(1).strip()
    for line in reversed(text.splitlines()):
        m = re.search(r"Итоговые значения.*?:\s*(\{.+?\})", line)
        if m:
            return name, dict_to_vector(json.loads(m.group(1)))
    raise ValueError(f"No final vector in {path}")
