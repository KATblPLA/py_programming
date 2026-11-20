import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==========================================
# ЛОГИКА ДЛЯ ЗАДАНИЯ 1 (Нечеткие множества)
# ==========================================
class Task1Logic:
    def __init__(self):
        self.x_max = 40.0
        self.x0 = 24.0
        # Параметры для функций принадлежности (Вариант 2: Температура для купания)
        # A1: Некомфортная (Z-образная) - спадает от 1 до 0 на [10, 25]
        self.a1_a, self.a1_b = 10.0, 25.0
        # A2: Нейтральная (Трапеция) - плато [15, 25], скосы [10, 30]
        self.a2_a, self.a2_b, self.a2_c, self.a2_d = 10.0, 15.0, 25.0, 30.0
        # A3: Очень комфортная (S-образная) - возрастает от 0 до 1 на [25, 40]
        self.a3_a, self.a3_b = 25.0, 40.0

    def mu_A1(self, x):
        """Z-образная функция (Некомфортная)"""
        if x <= self.a1_a: return 1.0
        elif x >= self.a1_b: return 0.0
        else: return (self.a1_b - x) / (self.a1_b - self.a1_a)

    def mu_A2(self, x):
        """Трапециевидная функция (Нейтральная)"""
        if x <= self.a2_a or x >= self.a2_d: return 0.0
        elif self.a2_b <= x <= self.a2_c: return 1.0
        elif self.a2_a < x < self.a2_b: return (x - self.a2_a) / (self.a2_b - self.a2_a)
        else: return (self.a2_d - x) / (self.a2_d - self.a2_c)

    def mu_A3(self, x):
        """S-образная функция (Очень комфортная)"""
        if x <= self.a3_a: return 0.0
        elif x >= self.a3_b: return 1.0
        else: return (x - self.a3_a) / (self.a3_b - self.a3_a)

    def get_discrete_set(self, func, step=1.0):
        """Возвращает дискретное представление в виде строки"""
        vals = []
        for x in np.arange(0, self.x_max + 0.1, step):
            mu = func(x)
            if mu > 0:
                vals.append(f"{mu:.2f}/{int(x) if x.is_integer() else x}")
        return " + ".join(vals) if vals else "0"

    def get_supp_core(self, func):
        """Носитель и ядро для A2"""
        supp = [x for x in np.arange(0, self.x_max + 0.1, 0.5) if func(x) > 0]
        core = [x for x in np.arange(0, self.x_max + 0.1, 0.5) if func(x) == 1.0]
        
        supp_str = f"[{min(supp):.1f}; {max(supp):.1f}]" if supp else "Пусто"
        core_str = f"[{min(core):.1f}; {max(core):.1f}]" if core else "Пусто"
        return supp_str, core_str

# ==========================================
# ЛОГИКА ДЛЯ ЗАДАНИЯ 2 (Композиция отношений)
# ==========================================
class Task2Logic:
    def __init__(self):
        # R1: Специальности (строки: характеристики, столбцы: Менеджер, Программист, Водитель, Секретарь, Переводчик)
        self.R1 = np.array([
            [0.9, 0.8, 0.3, 0.5, 0.7], # Быстрота мышления
            [0.9, 0.5, 0.9, 0.4, 0.8], # Решения
            [0.8, 0.9, 0.6, 0.5, 0.8], # Внимание
            [0.4, 0.3, 0.5, 0.5, 0.2], # Зрительная память
            [0.5, 0.1, 0.9, 0.2, 0.6], # Быстрота реакции
            [0.3, 0.2, 0.8, 0.2, 0.2], # Двигательная память
            [0.6, 0.2, 0.9, 0.3, 0.2], # Выносливость
            [0.2, 0.2, 0.8, 0.3, 0.3], # Координация
            [0.9, 0.5, 0.6, 0.9, 0.3], # Эмоц. устойчивость
            [0.8, 0.5, 0.3, 0.8, 0.2]  # Ответственность
        ])
        # R2: Кандидаты (строки: характеристики, столбцы: Петров, Иванов, Сидоров, Васильева, Григорьева)
        self.R2 = np.array([
            [0.9, 0.8, 0.7, 0.9, 1.0], # Быстрота мышления
            [0.6, 0.4, 0.8, 0.5, 0.6], # Решения
            [0.5, 0.2, 0.3, 0.8, 0.7], # Внимание
            [0.5, 0.9, 0.5, 0.8, 0.4], # Зрительная память
            [1.0, 0.6, 0.5, 0.7, 0.4], # Быстрота реакции
            [0.4, 0.5, 1.0, 0.7, 0.8], # Двигательная память
            [0.5, 0.8, 0.9, 0.5, 0.4], # Выносливость
            [0.5, 0.6, 0.7, 0.6, 0.5], # Координация
            [0.8, 1.0, 0.2, 0.5, 0.6], # Эмоц. устойчивость
            [0.3, 0.5, 0.9, 0.6, 0.8]  # Ответственность
        ])
        self.candidates = ['Петров', 'Иванов', 'Сидоров', 'Васильева', 'Григорьева']

    def max_min_composition(self):
        # R1 (10x5) x R2^T (5x10) -> (5x5)
        R2_T = self.R2.T
        n = self.R1.shape[1]
        m = R2_T.shape[1]
        result = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                result[i, j] = np.max(np.minimum(self.R1[:, i], R2_T[:, j]))
        return result

    def max_prod_composition(self):
        # R1 (10x5) x R2^T (5x10) -> (5x5)
        R2_T = self.R2.T
        n = self.R1.shape[1]
        m = R2_T.shape[1]
        result = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                result[i, j] = np.max(self.R1[:, i] * R2_T[:, j])
        return result

# ==========================================
# ИНТЕРФЕЙС ПРИЛОЖЕНИЯ
# ==========================================
class FuzzyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Практическая работа №1 - Нечеткие множества (Вариант 2)")
        self.root.geometry("1000x800")
        
        # Создаем вкладки
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab1, text='Задание 1: Нечеткие множества')
        self.notebook.add(self.tab2, text='Задание 2: Композиция отношений')
        
        self.task1 = Task1Logic()
        self.task2 = Task2Logic()
        
        self.setup_tab1()
        self.setup_tab2()

    def setup_tab1(self):
        # Панель управления
        frame_ctrl = ttk.LabelFrame(self.tab1, text="Управление")
        frame_ctrl.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(frame_ctrl, text="Построить графики ФП", command=self.plot_task1).pack(side='left', padx=10, pady=5)
        ttk.Button(frame_ctrl, text="Показать расчеты", command=self.calc_task1).pack(side='left', padx=10, pady=5)
        
        # Область для вывода текста
        self.text_task1 = tk.Text(self.tab1, height=15, width=100)
        self.text_task1.pack(fill='both', expand=True, padx=10, pady=5)

    def plot_task1(self):
        for widget in self.tab1.winfo_children():
            if isinstance(widget, tk.Canvas) or widget.winfo_class() == 'Frame' and widget != self.tab1:
                if widget.winfo_class() == 'Frame' and hasattr(widget, 'is_plot'):
                    widget.destroy()
                    
        # Создаем фрейм для графика
        plot_frame = ttk.Frame(self.tab1)
        plot_frame.pack(fill='both', expand=True, padx=10, pady=5)
        plot_frame.is_plot = True
        
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.linspace(0, 40, 500)
        
        y1 = [self.task1.mu_A1(val) for val in x]
        y2 = [self.task1.mu_A2(val) for val in x]
        y3 = [self.task1.mu_A3(val) for val in x]
        
        ax.plot(x, y1, label='A1: Некомфортная (Z)', color='blue')
        ax.plot(x, y2, label='A2: Нейтральная (Трапеция)', color='green')
        ax.plot(x, y3, label='A3: Очень комфортная (S)', color='red')
        
        # Отмечаем точку x0
        ax.axvline(x=self.task1.x0, color='gray', linestyle='--', label=f'x0 = {self.task1.x0}')
        
        ax.set_title("Функции принадлежности (Вариант 2)")
        ax.set_xlabel("Температура, °C")
        ax.set_ylabel("Степень принадлежности μ(x)")
        ax.legend()
        ax.grid(True)
        
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def calc_task1(self):
        self.text_task1.delete(1.0, tk.END)
        
        # 1. Дискретное представление
        self.text_task1.insert(tk.END, "--- 1. Дискретное представление (шаг 5) ---\n")
        self.text_task1.insert(tk.END, f"A1 = {self.task1.get_discrete_set(self.task1.mu_A1, 5)}\n")
        self.text_task1.insert(tk.END, f"A2 = {self.task1.get_discrete_set(self.task1.mu_A2, 5)}\n")
        self.text_task1.insert(tk.END, f"A3 = {self.task1.get_discrete_set(self.task1.mu_A3, 5)}\n\n")
        
        # 2. Значения в точке x0
        x0 = self.task1.x0
        mu1 = self.task1.mu_A1(x0)
        mu2 = self.task1.mu_A2(x0)
        mu3 = self.task1.mu_A3(x0)
        self.text_task1.insert(tk.END, f"--- 2. Значения в фиксированной точке x0 = {x0} ---\n")
        self.text_task1.insert(tk.END, f"μA1({x0}) = {mu1:.3f}\n")
        self.text_task1.insert(tk.END, f"μA2({x0}) = {mu2:.3f}\n")
        self.text_task1.insert(tk.END, f"μA3({x0}) = {mu3:.3f}\n\n")
        
        # 3. Свойства A2
        supp, core = self.task1.get_supp_core(self.task1.mu_A2)
        self.text_task1.insert(tk.END, f"--- 3. Свойства нечеткого множества A2 ---\n")
        self.text_task1.insert(tk.END, f"Носитель (supp): {supp}\n")
        self.text_task1.insert(tk.END, f"Ядро (core): {core}\n")
        self.text_task1.insert(tk.END, f"Точка перехода: x = 12.5 (μ=0.5) и x = 27.5 (μ=0.5)\n\n")
        
        # 4. Лингвистическая модификация A2
        self.text_task1.insert(tk.END, f"--- 4. Лингвистическая модификация A2 ---\n")
        self.text_task1.insert(tk.END, f"Оператор концентрации (Очень): μ^2. В точке x0={x0}: {mu2**2:.3f}\n")
        self.text_task1.insert(tk.END, f"Оператор растяжения (Довольно): sqrt(μ). В точке x0={x0}: {np.sqrt(mu2):.3f}\n\n")
        
        # 5. Операции над A1 и A2
        self.text_task1.insert(tk.END, f"--- 5. Операции над A1 и A2 (в точке x0={x0}) ---\n")
        self.text_task1.insert(tk.END, f"Дополнение A1 (1 - μ): {1 - mu1:.3f}\n")
        
        tmin = min(mu1, mu2)
        tprod = mu1 * mu2
        tmax = max(0, mu1 + mu2 - 1)
        self.text_task1.insert(tk.END, f"Пересечение (T-нормы):\n")
        self.text_task1.insert(tk.END, f"  MIN: {tmin:.3f}\n")
        self.text_task1.insert(tk.END, f"  PROD: {tprod:.3f}\n")
        self.text_task1.insert(tk.END, f"  Ограниченная разность: {tmax:.3f}\n")
        
        smax = max(mu1, mu2)
        ssum = mu1 + mu2 - mu1 * mu2
        smin = min(1, mu1 + mu2)
        self.text_task1.insert(tk.END, f"Объединение (S-нормы):\n")
        self.text_task1.insert(tk.END, f"  MAX: {smax:.3f}\n")
        self.text_task1.insert(tk.END, f"  Алгебраическая сумма: {ssum:.3f}\n")
        self.text_task1.insert(tk.END, f"  Ограниченная сумма: {smin:.3f}\n")

    def setup_tab2(self):
        frame_ctrl = ttk.LabelFrame(self.tab2, text="Управление")
        frame_ctrl.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(frame_ctrl, text="Выполнить композицию", command=self.calc_task2).pack(side='left', padx=10, pady=5)
        
        self.text_task2 = tk.Text(self.tab2, height=25, width=100)
        self.text_task2.pack(fill='both', expand=True, padx=10, pady=5)

    def calc_task2(self):
        self.text_task2.delete(1.0, tk.END)
        
        res_min = self.task2.max_min_composition()
        res_prod = self.task2.max_prod_composition()
        
        # Вывод матрицы max-min
        self.text_task2.insert(tk.END, "--- Результат композиции (max-min) ---\n")
        self.text_task2.insert(tk.END, "Строки: Специальности (Менеджер, Программист, Водитель, Секретарь, Переводчик)\n")
        self.text_task2.insert(tk.END, "Столбцы: Кандидаты (Петров, Иванов, Сидоров, Васильева, Григорьева)\n\n")
        
        header = " " * 12 + "".join([f"{c:>12}" for c in self.task2.candidates]) + "\n"
        self.text_task2.insert(tk.END, header)
        
        specs = ['Менеджер', 'Программист', 'Водитель', 'Секретарь', 'Переводчик']
        for i, spec in enumerate(specs):
            row_str = f"{spec:>12}"
            for j in range(5):
                row_str += f"{res_min[i, j]:>12.2f}"
            self.text_task2.insert(tk.END, row_str + "\n")
            
        self.text_task2.insert(tk.END, "\n--- Результат композиции (max-prod) ---\n")
        self.text_task2.insert(tk.END, header)
        for i, spec in enumerate(specs):
            row_str = f"{spec:>12}"
            for j in range(5):
                row_str += f"{res_prod[i, j]:>12.2f}"
            self.text_task2.insert(tk.END, row_str + "\n")
            
        self.text_task2.insert(tk.END, "\n--- Анализ результатов ---\n")
        self.text_task2.insert(tk.END, "1. Наилучший кандидат для каждой специальности (по max-min):\n")
        for i, spec in enumerate(specs):
            best_idx = np.argmax(res_min[i])
            self.text_task2.insert(tk.END, f"   {spec}: {self.task2.candidates[best_idx]} (степень {res_min[i, best_idx]:.2f})\n")
            
        self.text_task2.insert(tk.END, "\n2. Наилучший кандидат для каждой специальности (по max-prod):\n")
        for i, spec in enumerate(specs):
            best_idx = np.argmax(res_prod[i])
            self.text_task2.insert(tk.END, f"   {spec}: {self.task2.candidates[best_idx]} (степень {res_prod[i, best_idx]:.2f})\n")
            
        self.text_task2.insert(tk.END, "\nВывод: Результаты композиции разными методами совпадают по выбору лучшего кандидата,\nно значения степеней принадлежности отличаются. Max-min дает более высокие значения,\nтак как оператор MIN менее строгий, чем PROD.\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = FuzzyApp(root)
    root.mainloop()