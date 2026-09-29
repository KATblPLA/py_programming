# ВАРИАНТ 2
__author__ = "Volosikova Maria"


from Python_Edu_module import check 


'''
Вывести все четные числа кратные пяти в интервале от 2 до 100 включительно.
https://github.com/VetrovSV/Programming/blob/master/plans/ML/tasks/tasks_programming.md
'''
      

# вывод условия
print ('Вывести все четные числа кратные пяти в интервале от {n} до {m} включительно.')


          
# бесконечный цикл выполнения программы
while(1):
    print ( '\n\nВведите нижнюю границу: ')

    # ввод строки через input,
    # преобразование строки в число для мат. операций
    
    # проверка на любые, чтобы программа не ломалась,
    # если пользователь дурачок и ввел не число
    try:
        n = int(input())
    except Exception as e:
        print ('Нужно ввести число. Ошибка: {e}')


    print ( '\n\nВведите верхнюю границу: ')
    # ввод строки через input,
    # преобразование строки в число для мат. операций
    # проверка, чтобы программа не ломалась,
    # если пользователь дурачок и ввел не число
    try:
        m = int(input())
    except Exception as e:
        print ('Нужно ввести число. Ошибка: {e}')
        
    # in range(start, stop) перебирает до stop-1
    for cur in range (n, m+1):
        if check(cur) is not None:
            # вывод
            print (cur)
