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


class TestCountNumbers:
    @pytest.mark.parametrize("min_, max_, is_odd, expected", [
        # базовые случаи: весь диапазон
        (0,   100, True,  50),   # нечётные: 1..99    → 50
        (0,   100, False, 51),   # чётные:   0..100   → 51
        (0,   0,   True,  0),    # только 0, нечётных нет
        (0,   0,   False, 1),    # только 0, чётное есть
        (1,   1,   True,  1),    # только 1, нечётное есть
        (1,   1,   False, 0),    # только 1, чётных нет
        # границы совпадают с чётностью
        (2,   2,   False, 1),    # одно чётное
        (2,   3,   True,  1),    # только 3
        (2,   3,   False, 1),    # только 2
        # отрицательные
        (-100, 100, True,  100), # -99..99  → 100
        (-100, 100, False, 101), # -100..100 → 101
        (-10,  10,  True,  10),  # -9..9   → 10
        (-10,  10,  False, 11),  # -10..10 → 11
        # случай, когда чисел нужной чётности нет вообще
        (5,   5,   False, 0),    # только 5, чётных нет
        (6,   6,   True,  0),    # только 6, нечётных нет
        # маленькие диапазоны
        (3,   4,   True,  1),
        (3,   4,   False, 1),
        (10,  11,  True,  1),
        (10,  11,  False, 1),
    ])
    def test_count_numbers_ok(self, min_, max_, is_odd, expected):
        assert pem.count_numbers(min_, max_, is_odd) == expected

    @pytest.mark.parametrize("min_, max_", [
        (100, 0),
        (10,  9),
        (1,   -1),
    ])
    def test_count_numbers_invalid_range(self, min_, max_):
        # count_numbers сам не проверяет min_ > max_,
        # поэтому проверяем согласованность с count_odd/count_even
        # (в g_* проверка диапазона идёт отдельно)
        with pytest.raises(ValueError):
            # если count_numbers не бросает — тест упадёт,
            # что сигнализирует о несоответствии контракта
            pem.g_arr(min_, max_, 1, True)

    def test_count_numbers_invalid_range(self):
        with pytest.raises(ValueError):
            pem.count_numbers(100, 0, True)


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


class TestG_Ndarr_Fast:
    @pytest.mark.parametrize("min_, max_, length, is_odd", [
        (0,    100, 10, True),
        (-100, 100, 50, True),
        (30,   50,  5,  True),
        (0,    100, 10, False),
        (-100, 100, 50, False),
        (30,   50,  5,  False),
    ])
    def test_g_ndarr_fast_ok(self, min_, max_, length, is_odd):
        result = pem.g_ndarr_fast(min_, max_, length, is_odd)

        assert isinstance(result, np.ndarray)
        assert result.shape == (length,)
        assert np.all(result % 2 == int(is_odd))
        assert np.all((result >= min_) & (result <= max_))

    @pytest.mark.parametrize("min_, max_, length, is_odd", [
        (0,    100, 10, True),
        (-100, 100, 50, True),
        (0,    100, 10, False),
        (30,   50,  5,  False),
    ])
    def test_g_ndarr_fast_unique(self, min_, max_, length, is_odd):
        result = pem.g_ndarr_fast(min_, max_, length, is_odd)
        assert len(np.unique(result)) == length

    @pytest.mark.parametrize("min_, max_, length, is_odd", [
        (100,  0,  50, True),    # min > max
        (-10,  10, 30, True),    # недостаточно нечётных
        (-10,  10, 30, False),   # недостаточно чётных
        (5,    5,  1,  False),   # единственное число чётное, нужны нечётные
        (6,    6,  1,  True),    # единственное число нечётное, нужны чётные
        (0,    0,  2,  False),   # чётное только одно, а нужно два
    ])
    def test_g_ndarr_fast_raises(self, min_, max_, length, is_odd):
        with pytest.raises(ValueError):
            pem.g_ndarr_fast(min_, max_, length, is_odd)

    def test_g_ndarr_fast_full_range(self):
        # граничный случай: запрашиваем ровно все числа нужной чётности
        result = pem.g_ndarr_fast(1, 9, 5, True)   # 1,3,5,7,9
        assert sorted(result.tolist()) == [1, 3, 5, 7, 9]

    def test_g_ndarr_fast_negative_step(self):
        # проверка с отрицательным is_odd через int()
        result = pem.g_ndarr_fast(0, 10, 3, 0)
        assert np.all(result % 2 == 0)

    def test_g_ndarr_fast_dtype(self):
        result = pem.g_ndarr_fast(0, 100, 5, True)
        assert np.issubdtype(result.dtype, np.integer)

    
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