'''
модуль с функциями для заданий для обучения Python
'''

import pytest
import numpy as np
import Python_Edu_module as pem

__author__ = "Volosikova M.A."


class TestCalcBonus:
    @pytest.mark.parametrize("salary, percent, expected", [
        (100,  3,  3.0),
        (1000, 80, 800.0),
        (800,  5,  40.0),
    ])
    def test_calc_bonus(self, salary, percent, expected):
        assert pem.calc_bonus(salary, percent) == expected


class TestCheck:
    @pytest.mark.parametrize("value, expected", [
        (-10, -10),
        (0,    0),
    ])
    def test_check_returns_value(self, value, expected):
        assert pem.check(value) == expected

    @pytest.mark.parametrize("value", [4, 5])
    def test_check_returns_none(self, value):
        assert pem.check(value) is None


class TestCountOdd:
    @pytest.mark.parametrize("start, end, expected", [
        (0,    100, 50),
        (0,    0,   0),
        (1,    1,   1),
        (1,    2,   1),
        (-100, 100, 100),
    ])
    def test_count_odd_ok(self, start, end, expected):
        assert pem.count_odd(start, end) == expected

    @pytest.mark.parametrize("start, end", [
        (100, 0),      # start > end
    ])
    def test_count_odd_raises(self, start, end):
        with pytest.raises(ValueError):
            pem.count_odd(start, end)


class TestCountEven:
    @pytest.mark.parametrize("start, end, expected", [
        (0,    100, 51),
        (0,    0,   1),
        (1,    1,   0),
        (1,    2,   1),
        (-100, 100, 101),
    ])
    def test_count_even_ok(self, start, end, expected):
        assert pem.count_even(start, end) == expected

    def test_count_even_invalid_range(self):
        with pytest.raises(ValueError):
            pem.count_even(100, 0)


class TestG_Arr:
    @pytest.mark.parametrize("min_, max_, length, is_odd", [
        (0,    100, 10, 1),
        (-100, 100, 50, 1),
        (30,   50,  5,  1),
        (0,    100, 10, 0),
        (-100, 100, 50, 0),
        (30,   50,  5,  0),
    ])
    def test_g_arr_ok(self, min_, max_, length, is_odd):
        result = pem.g_arr(min_, max_, length, is_odd)
        assert len(result) == length
        assert all(i % 2 == int(is_odd) for i in result)

    @pytest.mark.parametrize("min_, max_, length, is_odd", [
        (100,  0,  50, 1),   # min > max
        (-10,  10, 30, 1),   # недостаточно нечётных
        (-10,  10, 30, 0),   # недостаточно чётных
    ])
    def test_g_arr_raises(self, min_, max_, length, is_odd):
        with pytest.raises(ValueError):
            pem.g_arr(min_, max_, length, is_odd)


class TestG_Set:
    @pytest.mark.parametrize("min_, max_, length, is_odd", [
        (0,    100, 10, 1),
        (-100, 100, 50, 1),
        (30,   50,  5,  1),
        (0,    100, 10, 0),
        (-100, 100, 50, 0),
        (30,   50,  5,  0),
    ])
    def test_g_set_ok(self, min_, max_, length, is_odd):
        result = pem.g_set(min_, max_, length, is_odd)
        assert len(result) == length
        assert all(i % 2 == int(is_odd) for i in result)

    @pytest.mark.parametrize("min_, max_, length, is_odd", [
        (100,  0,  50, 1),   # min > max
        (-10,  10, 30, 1),   # недостаточно нечётных
        (-10,  10, 30, 0),   # недостаточно чётных
    ])
    def test_g_set_raises(self, min_, max_, length, is_odd):
        with pytest.raises(ValueError):
            pem.g_set(min_, max_, length, is_odd)


class TestG_Ndarr:
    @pytest.mark.parametrize("min_, max_, length, is_odd", [
        (0,    100, 10, 1),
        (-100, 100, 50, 1),
        (30,   50,  5,  1),
        (0,    100, 10, 0),
        (-100, 100, 50, 0),
        (30,   50,  5,  0),
    ])
    def test_g_ndarr_ok(self, min_, max_, length, is_odd):
        result = pem.g_ndarr(min_, max_, length, is_odd)

        assert isinstance(result, np.ndarray)
        assert len(result) == length
        assert np.all(result % 2 == int(is_odd))
        assert np.all((result >= min_) & (result <= max_))

    @pytest.mark.parametrize("min_, max_, length, is_odd", [
        (100,  0,  50, 1),   # min > max
        (-10,  10, 30, 1),   # недостаточно нечётных
        (-10,  10, 30, 0),   # недостаточно чётных
    ])
    def test_g_ndarr_raises(self, min_, max_, length, is_odd):
        with pytest.raises(ValueError):
            pem.g_ndarr(min_, max_, length, is_odd)

    
class TestToCsv:
    @pytest.mark.parametrize("filename, data", [
        (1, "123"),
        ("1", []),
        ("1", 1) 
    ])
    def test_g_set_raises(self, filename, data):
        with pytest.raises(ValueError):
            pem.to_csv(filename, data)


class TestMySigmoid:
    @pytest.mark.parametrize("data", [
        None,
        "abc",
        b"abc",
        [],
        np.array([]),
    ])
    def test_raises(self, data):
        with pytest.raises(ValueError):
            pem.my_sigmoid(data)

    @pytest.mark.parametrize("x, expected", [
        (0,    0.5),
        (100,  1.0),          # ≈1
        (-100, 0.0),          # ≈0
    ])
    def test_scalar(self, x, expected):
        assert pem.my_sigmoid(x) == pytest.approx(expected, abs=1e-6)

    def test_ndarray(self):
        arr = np.array([-100, 0, 100])
        result = pem.my_sigmoid(arr)

        assert isinstance(result, np.ndarray)
        assert result.shape == arr.shape
        assert np.all(result >= 0) and np.all(result <= 1)
        np.testing.assert_allclose(
            result,
            np.array([0.0, 0.5, 1.0]),
            atol=1e-6,
        )