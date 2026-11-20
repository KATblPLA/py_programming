"""Модуль для работы с моделью: загрузка, предсказание, графики, LaTeX.

Модель читается из .pkl один раз при импорте модуля и живёт в памяти.
"""
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# Пути проекта (родитель папки app/)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "Linreg_sklearn_ver_1.6.1_pickle_ver_4.0.pkl"
DATA_PATH = BASE_DIR / "data" / "subsample_2.csv"


def _load_model(path: Path = MODEL_PATH):
    """Загружает модель из pickle-файла.
    Принимает путь к .pkl (по умолчанию MODEL_PATH).
    Возвращает объект sklearn-модели."""
    with open(path, "rb") as f:
        return pickle.load(f)


MODEL = _load_model()

FEATURES: list[str] = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "waterfront",
    "condition", "grade", "sqft_above", "sqft_basement", "yr_built",
    "yr_renovated", "lat", "long", "sqft_living15", "sqft_lot15",
]
TARGET: str = "price_ln"

_coef_len = int(np.asarray(MODEL.coef_).ravel().shape[0])
assert _coef_len == len(FEATURES), (
    f"Модель ждёт {_coef_len} признаков, а FEATURES={len(FEATURES)}"
)


def _load_train_df() -> pd.DataFrame:
    """Читает обучающий CSV.
    Принимает ничего (путь берёт из DATA_PATH).
    Возвращает DataFrame с данными для scatter и R²."""
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    raise FileNotFoundError(
        f"Не найден датасет {DATA_PATH}. Положите subsample_2.csv в папку data/."
    )


TRAIN_DF = _load_train_df()


def correlation_matrix(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Считает матрицу корреляций Пирсона.
    Принимает DataFrame (или None — тогда TRAIN_DF).
    Возвращает квадратную матрицу (n_features+1) x (n_features+1)."""
    df = TRAIN_DF if df is None else df
    return df[FEATURES + [TARGET]].corr(method="pearson")


def _top_features_by_corr(k: int = 3) -> list[str]:
    """Находит k признаков с наибольшей |корреляцией| с TARGET.
    Принимает число k (по умолчанию 3).
    Возвращает список имён признаков по убыванию |корреляции|."""
    corr = correlation_matrix()
    target_corr = corr[TARGET].drop(labels=[TARGET])
    return list(target_corr.abs().sort_values(ascending=False).index[:k])


CORR_MATRIX: pd.DataFrame = correlation_matrix()
TOP_FEATURES: list[str] = _top_features_by_corr(3)


def _compute_r2() -> float:
    """Считает R² модели на обучающей выборке.
    Принимает ничего.
    Возвращает R² как float."""
    X = TRAIN_DF[FEATURES].values
    y = TRAIN_DF[TARGET].values
    return float(r2_score(y, MODEL.predict(X)))


meta: dict = {
    "r2_test": _compute_r2(),
    "target": TARGET,
    "n_features": len(FEATURES),
    "model_path": str(MODEL_PATH),
    "model_class": type(MODEL).__name__,
    "n_train": int(len(TRAIN_DF)),
    "top_features_corr": {
        f: round(float(CORR_MATRIX.loc[f, TARGET]), 4) for f in TOP_FEATURES
    },
}


def predict(values: list[float]) -> float:
    """Предсказывает log(price) для одного дома.
    Принимает список из 15 значений в порядке FEATURES.
    Возвращает pred_ln (float); бросает ValueError при неверной длине."""
    x = np.asarray(values, dtype=float).reshape(1, -1)
    if x.shape[1] != len(FEATURES):
        raise ValueError(f"Ожидалось {len(FEATURES)} признаков, получено {x.shape[1]}")
    return float(MODEL.predict(x)[0])


def _scatter_for_feature(feature: str, values: list[float], pred_ln: float) -> plt.Figure:
    """Строит scatter по одному признаку с точкой пользователя и линией регрессии.
    Принимает имя признака, вектор значений пользователя и pred_ln.
    Возвращает matplotlib Figure."""
    idx = FEATURES.index(feature)
    x_user = float(values[idx])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(TRAIN_DF[feature], TRAIN_DF[TARGET],
               s=18, alpha=0.45, label="Обучающая выборка")
    ax.scatter([x_user], [pred_ln], s=160, color="red", marker="*",
               edgecolor="black", zorder=5, label="Ваша точка")

    # Линия регрессии: остальные признаки фиксированы на средних
    xs = np.linspace(TRAIN_DF[feature].min(), TRAIN_DF[feature].max(), 100)
    grid = np.tile(TRAIN_DF[FEATURES].mean().values, (len(xs), 1))
    grid[:, idx] = xs
    ys = MODEL.predict(grid)
    ax.plot(xs, ys, color="green", lw=2, label="Линия регрессии")

    ax.set_xlabel(feature)
    ax.set_ylabel(TARGET)
    ax.set_title(f"{TARGET} vs {feature}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def make_scatters(values: list[float], pred_ln: float) -> list[plt.Figure]:
    """Строит scatter для всех TOP_FEATURES.
    Принимает вектор значений пользователя и pred_ln.
    Возвращает список Figure (по одному на признак)."""
    return [_scatter_for_feature(f, values, pred_ln) for f in TOP_FEATURES]


def latex_equation() -> str:
    """Собирает уравнение регрессии в LaTeX.
    Принимает ничего.
    Возвращает строку вида $$ y = ... $$ с коэффициентами до 2 знаков."""
    coefs = np.asarray(MODEL.coef_).ravel()
    intercept = float(np.asarray(MODEL.intercept_).ravel()[0])

    parts: list[str] = []
    for name, c in zip(FEATURES, coefs):
        c_r = round(float(c), 2)
        if c_r == 0:
            continue
        sign = "+" if c_r >= 0 else "-"
        parts.append(f"{sign} {abs(c_r)}\\,x_{{\\mathrm{{{name}}}}}")

    if not parts:
        return rf"$$ y = {round(intercept, 2)} $$"
    first = parts[0].lstrip("+ ").strip()
    body = first + "".join(" " + p for p in parts[1:])
    sign = "+" if intercept >= 0 else "-"
    return rf"$$ y = {body} {sign} {abs(round(intercept, 2))} $$"