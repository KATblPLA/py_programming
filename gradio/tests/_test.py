"""
Тесты модели и функций func_module.

Запуск (из корня проекта):
    pytest -v
"""
import matplotlib.figure
import numpy as np
import pytest

import func_module as fm

__author__ = "Volosikova M.A."


# ============================================================
# Модель
# ============================================================
class TestModelLoaded:
    def test_model_not_none(self):
        assert fm.MODEL is not None

    def test_model_has_coef(self):
        assert hasattr(fm.MODEL, "coef_")

    def test_model_has_predict(self):
        assert hasattr(fm.MODEL, "predict")

    def test_model_file_exists(self):
        assert fm.MODEL_PATH.exists(), f"Нет файла модели {fm.MODEL_PATH}"


class TestFeaturesMatchModel:
    @pytest.mark.parametrize("n_expected", [15])
    def test_features_count(self, n_expected):
        n_model = int(np.asarray(fm.MODEL.coef_).ravel().shape[0])
        assert n_model == len(fm.FEATURES) == n_expected


# ============================================================
# predict
# ============================================================
class TestPredict:
    def test_predict_matches_model(self):
        """fm.predict должен совпадать с MODEL.predict на фиксированном входе."""
        x = np.ones(len(fm.FEATURES))
        expected = float(fm.MODEL.predict(x.reshape(1, -1))[0])
        assert fm.predict(x.tolist()) == pytest.approx(expected, abs=1e-8)

    @pytest.mark.parametrize("fill", [0.0, 1.0, -1.0, 100.0])
    def test_predict_matches_model_various_fill(self, fill):
        """Согласованность обёртки и модели на разных константных входах."""
        x = np.full(len(fm.FEATURES), fill)
        expected = float(fm.MODEL.predict(x.reshape(1, -1))[0])
        assert fm.predict(x.tolist()) == pytest.approx(expected, abs=1e-8)

    def test_predict_on_means_is_finite(self):
        """Предсказание на средних значениях — конечное число."""
        vals = [fm.TRAIN_DF[f].mean() for f in fm.FEATURES]
        assert np.isfinite(fm.predict(vals))

    def test_predict_returns_float(self):
        """Обёртка возвращает встроенный float, а не numpy-тип."""
        vals = [fm.TRAIN_DF[f].mean() for f in fm.FEATURES]
        assert type(fm.predict(vals)) is float

    @pytest.mark.parametrize("length", [
        len(fm.FEATURES) - 1,
        len(fm.FEATURES) + 1,
        0,
        1,
    ])
    def test_predict_wrong_length(self, length):
        with pytest.raises(ValueError):
            fm.predict([0.0] * length)


# ============================================================
# R² и метаданные
# ============================================================
class TestMeta:
    def test_r2_in_range(self):
        assert 0.0 <= fm.meta["r2_test"] <= 1.0

    def test_meta_has_required_keys(self):
        for key in ("r2_test", "target", "n_features", "model_path", "n_train"):
            assert key in fm.meta

    def test_meta_n_features(self):
        assert fm.meta["n_features"] == len(fm.FEATURES)


# ============================================================
# Данные
# ============================================================
class TestData:
    def test_train_csv_exists(self):
        assert fm.DATA_PATH.exists(), f"Нет файла {fm.DATA_PATH}"

    def test_train_df_not_empty(self):
        assert len(fm.TRAIN_DF) > 0

    def test_train_df_has_all_columns(self):
        for col in fm.FEATURES + [fm.TARGET]:
            assert col in fm.TRAIN_DF.columns


# ============================================================
# Матрица корреляций и топ-признаки
# ============================================================
class TestCorrelation:
    def test_corr_matrix_shape(self):
        n = len(fm.FEATURES) + 1
        assert fm.CORR_MATRIX.shape == (n, n)

    def test_corr_matrix_symmetric(self):
        assert np.allclose(fm.CORR_MATRIX.values, fm.CORR_MATRIX.values.T)

    def test_corr_matrix_diag_ones(self):
        assert np.allclose(np.diag(fm.CORR_MATRIX.values), 1.0)


class TestTopFeatures:
    def test_top_features_match_manual(self):
        corr = fm.CORR_MATRIX[fm.TARGET].drop(labels=[fm.TARGET]).abs()
        expected = list(corr.sort_values(ascending=False).index[:len(fm.TOP_FEATURES)])
        assert fm.TOP_FEATURES == expected

    @pytest.mark.parametrize("name", ["grade", "sqft_living"])
    def test_top_features_contain_key(self, name):
        """На текущем датасете эти признаки в топ-3 по корреляции."""
        assert name in fm.TOP_FEATURES

    def test_top_features_length(self):
        assert len(fm.TOP_FEATURES) == 3


# ============================================================
# LaTeX-уравнение
# ============================================================
class TestLatexEquation:
    def test_format(self):
        eq = fm.latex_equation()
        assert eq.startswith("$$") and eq.endswith("$$")

    def test_contains_y(self):
        assert "y =" in fm.latex_equation()

    def test_terms_count(self):
        """По слагаемому на каждый ненулевой коэффициент."""
        eq = fm.latex_equation()
        n_terms = sum(1 for c in np.ravel(fm.MODEL.coef_) if round(float(c), 2) != 0)
        assert eq.count("\\,x_") == n_terms


# ============================================================
# Scatter-графики
# ============================================================
class TestMakeScatters:
    def _vals_and_pred(self):
        vals = [fm.TRAIN_DF[f].mean() for f in fm.FEATURES]
        return vals, fm.predict(vals)

    def test_returns_list_of_figures(self):
        vals, pred = self._vals_and_pred()
        figs = fm.make_scatters(vals, pred)
        assert isinstance(figs, list)
        assert all(isinstance(f, matplotlib.figure.Figure) for f in figs)

    def test_count_matches_top_features(self):
        vals, pred = self._vals_and_pred()
        assert len(fm.make_scatters(vals, pred)) == len(fm.TOP_FEATURES)

    def test_has_regression_line(self):
        vals, pred = self._vals_and_pred()
        for fig in fm.make_scatters(vals, pred):
            ax = fig.axes[0]
            assert len(ax.lines) >= 1, "нет линии регрессии"

    def test_has_point_cloud(self):
        vals, pred = self._vals_and_pred()
        for fig in fm.make_scatters(vals, pred):
            ax = fig.axes[0]
            assert any(hasattr(c, "get_offsets") for c in ax.collections), "нет облака точек"

    def test_has_user_point(self):
        """Точка пользователя — это вторая коллекция на графике."""
        vals, pred = self._vals_and_pred()
        for fig in fm.make_scatters(vals, pred):
            ax = fig.axes[0]
            assert len(ax.collections) >= 2, "нет точки пользователя"