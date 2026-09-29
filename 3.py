import Python_Edu_module as pe
import numpy as np

arr: np.array = pe.g_ndarr(-100, 100, 10, 0)

print(arr)

# форматирование вывода массива, чтобы не было экспоненциальной формы, кругление до 4 цифр после запятой,
# np.set_printoptions(suppress=True, precision=4)

print(pe.my_sigmoid(arr))