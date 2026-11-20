'''
модуль с функциями для заданий для обучения Python
'''

import random
import numpy as np
import csv
from collections.abc import Iterable

__author__ = "Volosikova M.A."


def calc_bonus(money, percent: float):
    '''
    считает надбавку, умножает количество зп на процент
    '''
    return money * percent * 0.01


def check(i: int):
    '''
    проверяет число на четность и кратность 5
    '''
    if i % 2 == 0 and i % 5 == 0:
        return i


def count_odd(min_: int, max_: int) -> int:
    '''
    считает количество нечетных чисел в диапазоне
    '''
    if min_ > max_:
        raise ValueError("Неправильно заданный диапазон (min_ > max_)")
    return (max_ + 1) // 2 - min_ // 2


def count_even(min_: int, max_: int) -> int:
    '''
    считает количество четных чисел в диапазоне
    '''
    if min_ > max_:
            raise ValueError("Неправильно заданный диапазон (min_ > max_)")
    return max_ // 2 - (min_ - 1) // 2


def count_numbers(min_: int, max_: int, is_odd: bool) -> int:
    '''
        считает количество четных и нечетных чисел в диапазоне
    '''
    if min_ > max_:
            raise ValueError("Неправильно заданный диапазон (min_ > max_)")
    
    # считает начальное число диапазона
    # пример: если число четное, а требуется нечетное, то прибавляется 1 
    start = min_ + ((min_ % 2) != int(is_odd))
    if start > max_:
        return 0
    return (max_ - start) // 2 + 1


def g_arr(min_: int, max_: int, lenght: int, is_odd: bool) -> list[int]:
    '''
    генератор четных или нечетных (is_odd) чисел в диапазоне (min_, max_)
    '''
    # проверка диапазона
    if min_>max_:
        raise ValueError("Неправильно заданный диапазон (min_ >= max_)")
    
    # проверка, достаточно ли в диапазоне уникальных чисел
    if count_numbers(min_, max_, is_odd) < lenght:
        raise ValueError ("В диапазоне нет достаточного количества чисел")
        
    # результат - массив
    result = []

    # пока не будет сгенерировано достаточно уникальных чисел
    while len(result) < lenght:
        # генерация
        cur = random.randint(min_, max_)
        if cur % 2 == int(is_odd) and cur not in result:
            result.append(cur)
    return result


def g_set(min_: int, max_: int, lenght: int, is_odd: bool) -> set[int]:
    '''
    генератор четных или нечетных (is_odd) чисел в диапазоне (min_, max_)
    Множество (set) в Python устроено на основе хеш-таблиц (hash tables)
    оиск за O(1): Благодаря хешам Python сразу знает, в какую ячейку памяти посмотреть
    '''
    # проверка диапазона
    if min_>max_:
        raise ValueError("Неправильно заданный диапазон (min_ >= max_)")
    
    # проверка, достаточно ли в диапазоне уникальных чисел
    if count_numbers(min_, max_, is_odd) < lenght:
        raise ValueError ("В диапазоне нет достаточного количества чисел")
        
    # результат - множество
    # множество в Python устроено на базе хеш-таблицы 
    result = set()

    # пока не будет сгенерировано достаточно уникальных чисел. set - множество уникальных чисел
    while len(result) < lenght:
        # генерация
        cur = random.randint(min_, max_)
        if cur % 2 == int(is_odd):
            result.add(cur)
    return result


def g_ndarr(min_: int, max_: int, lenght: int, is_odd: bool) -> np.array:
    '''
    генератор четных или нечетных (is_odd) чисел в диапазоне (min_, max_)
    '''
    # проверка диапазона
    if min_ > max_:
        raise ValueError("Неправильно заданный диапазон (min_ >= max_)")
    
    # проверка, достаточно ли в диапазоне уникальных чисел
    if count_numbers(min_, max_, is_odd) < lenght:
        raise ValueError ("В диапазоне нет достаточного количества чисел")

  
    # результат - массив
    result = np.empty(lenght, dtype=int)

    if min_%2 != int(is_odd):
        min_ +=1
    result[:] = random.sample(range(min_, max_, 2), lenght)

    return result


def g_ndarr_fast(min_, max_, lenght, is_odd):
    '''
        генератор четных или нечетных (is_odd) чисел в диапазоне (min_, max_)
    '''
    if min_ > max_:
        raise ValueError("Неправильно заданный диапазон")
    
    rng = np.random.default_rng()
  
    total = count_numbers(min_, max_, is_odd)
    if total < lenght:
        raise ValueError("В диапазоне нет достаточного количества чисел")

    # выбрать lenght случайных индексов из последовательности [0..total] без повторений (replace=False).
    idx = rng.choice(total, size=lenght, replace=False)

    # если нижняя граница диапазона нечетная, делаем четной и наоборот
    if min_%2 != int(is_odd):
        min_ +=1

    # преобразовать индексы в числа
    # 2 - шаг для четности
    return min_ + idx * 2 

    
def to_csv(filename: str, data) -> None:
    '''
    функция выводит коллекцию в файл
    бросает исключения если: data нет (None) или пустая, это строка, коллекция не итерируется
    '''
    if data is None:
        raise ValueError("data не может быть None")
    if isinstance(data, (str, bytes)):
        raise ValueError("data не может быть строкой")
    if not isinstance(data, Iterable):
        raise ValueError("data должна быть итерируемой")
    if len(data) == 0:
        raise ValueError("data не может быть пустой")

    with open(filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for item in data:
            writer.writerow([item])


def my_sigmoid(data):
    '''
    функция сигмоиды применяется к data, возвращает результат в диапазоне [0,1]
    может применяться к скалярам или ndarray - функция применяется к каждому элементу
    '''
    if data is None:
        raise ValueError("data не может быть None")
    if isinstance(data, (str, bytes)):
        raise ValueError("data не может быть строкой")
    if isinstance(data, Iterable) and len(data) == 0:
            raise ValueError("data не может быть пустой")
    
    return 1 /(1 + np.exp(-data))
