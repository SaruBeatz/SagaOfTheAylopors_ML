# Accentuation ML (Python v1)

Predicts one dominant Leonard accentuation from the 10 behavioral parameters (`Soc` … `Adp`).

## Setup

```bash
cd ml
pip install -r requirements.txt
```

## Pipeline

```bash
# 1. Synthetic dataset (real chapter JSON + target profiles)
python generate_synthetic.py

# 2. Train & compare models (GroupKFold by path_id)
python train_compare.py

# 3. Sanity check on 10 reference profile_* playthrough finals
python evaluate_profiles.py
```

## Outputs

| Path | Description |
|------|-------------|
| `data/synthetic.csv` | Training rows (final + per-chapter snapshots) |
| `models/accentuation_classifier.joblib` | Winner model + label encoder |
| `reports/metrics.json` | CV metrics, winner, profile sanity |
| `reports/confusion_*.png` | Confusion matrices per model |
| `reports/profile_sanity.json` | Per-profile prediction check |
| `reports/presentation_model_comparison_f1.png` | Bar chart: macro-F1 CV (all models) |
| `reports/presentation_model_comparison_accuracy.png` | Bar chart: accuracy CV |
| `reports/presentation_confusion_matrix.png` | Winner confusion matrix for slides |

### Charts for thesis / presentation

```bash
python plot_presentation.py              # from existing metrics.json
python plot_presentation.py --confusion  # also regenerate winner confusion PNG
```

## Modules

- `common.py` — feature names, target profiles, vector helpers
- `story_graph.py` — load `chapter_*.json`, simulate greedy paths
- `generate_synthetic.py` — build CSV with noise / moderate / mixed variants
- `train_compare.py` — centroid, kNN, LR, SVM, RF, HGB, decision tree
- `evaluate_profiles.py` — must-pass on `profile_*_playthrough.txt` finals

## v1 winner (typical run)

**Logistic Regression** (macro-F1 ~0.68 GroupKFold) — best balance of accuracy and exportability for a future Android phase.

---

## Где лежит модель

| Файл | Назначение |
|------|------------|
| **`models/accentuation_classifier.joblib`** | Полная модель (sklearn Pipeline + LabelEncoder). Нужны `joblib`, `scikit-learn`. |
| `models/accentuation_export.json` | Веса LR + scaler — после `python export_model.py` |
| `models/accentuation_export.py` | Автономный предиктор (только numpy) для других проектов |

Появляется после `python train_compare.py`. Если файла нет — сначала обучите модель.

---

## Как протестировать

### 1. Быстрая проверка на 10 эталонных профилях

```bash
python evaluate_profiles.py
```

Ожидается: `Sanity: 10/10 profiles matched`. Отчёт: `reports/profile_sanity.json`.

### 2. Один вектор вручную

```bash
python predict.py --profile ml/profile_Тревожный_playthrough.txt
```

или JSON (те же ключи, что в игре / Firestore):

```bash
python predict.py --json "{\"Soc\":0.36,\"Act\":0.22,\"Emp\":0.49,\"Anx\":1,\"Ctrl\":1,\"Imp\":0.31,\"Ego\":0.88,\"Rig\":0.75,\"Neg\":0.82,\"Adp\":0.58}"
```

С проверкой ожидаемой метки (код выхода 2 при ошибке):

```bash
python predict.py --profile ml/profile_Гипертим_playthrough.txt --expected Гипертим
```

### 3. Строка из синтетического датасета

```bash
python predict.py --csv-row data/synthetic.csv --index 100
```

### 4. Метрики и confusion matrix

`reports/metrics.json`, картинки `reports/confusion_*.png`.

---

## Экспорт в другие проекты

### Вариант A — без sklearn (рекомендуется для JS/Java/другого Python)

```bash
python export_model.py
```

Скопируйте в другой репозиторий:

- `models/accentuation_export.json`
- `models/accentuation_export.py`

```bash
cd /path/to/other/project
pip install numpy
python accentuation_export.py "{\"Soc\":0.5,\"Act\":0.5,...}"
```

Или в коде:

```python
from accentuation_export import predict
result = predict({"Soc": 0.36, "Act": 0.22, "Emp": 0.49, "Anx": 1.0, ...})
print(result["predicted"], result["confidence"])
print(result["top_k"])
```

В Java/Kotlin можно реализовать ту же формулу по полям JSON (`scaler_mean`, `coefficients`, `intercept`).

### Вариант B — полный sklearn bundle

Скопируйте `accentuation_classifier.joblib` + зависимости из `requirements.txt` (минимум: `scikit-learn`, `joblib`, `numpy`).

```python
import joblib
import numpy as np

bundle = joblib.load("accentuation_classifier.joblib")
features = bundle["features"]  # Soc, Act, ...
x = np.array([[0.36, 0.22, 0.49, 1.0, 1.0, 0.31, 0.88, 0.75, 0.82, 0.58]])
pred = bundle["model"].predict(x)[0]
proba = bundle["model"].predict_proba(x)[0]
print(pred, proba.max())
```

### Вариант C — ONNX (будущая фаза Android)

Для v1 не экспортируется автоматически; для LR можно добавить `skl2onnx` отдельно. Пока проще `accentuation_export.json` в нативный код.

---

## Соответствие полей игры

| CSV / JSON short | Firestore / `PlayerProgress` |
|------------------|------------------------------|
| Soc | sociality |
| Act | activity |
| Emp | emotionalSensitivity |
| Anx | anxiety |
| Ctrl | selfControl |
| Imp | impulsivity |
| Ego | egoFocus |
| Rig | rigidity |
| Neg | negativeAffect |
| Adp | adaptability |

Все значения в диапазоне **0.0–1.0** (старт игры 0.5).

---

## Android integration

- Model asset: `app/src/main/assets/ml/accentuation_export.json`
- Inference: `app/.../ml/AccentuationPredictor.java` (no sklearn on device)
- Chapter end: hidden predict + Firestore `predictions` on sync (`PlaythroughRepository`)
- Game end: `FinalScreenActivity` shows primary accentuation + description
- Backfill existing Firestore users: `scripts/firestore_backfill_predictions.py` (see `scripts/README_firestore_predictions.md`)
