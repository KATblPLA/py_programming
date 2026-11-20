import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox

# ==================== ЦЕЛЕВАЯ ФУНКЦИЯ И ГРАДИЕНТ (вариант 1.2) ====================
def f(x, y):
    return (x**4/4 - x**3/3 - 5*x**2 - 8*x +
            y**4/4 - y**3/3 - 17*y**2/2 - 15*y)

def grad(x, y):
    gx = x**3 - x**2 - 10*x - 8
    gy = y**3 - y**2 - 17*y - 15
    return np.array([gx, gy], dtype=float)

# ==================== СТАЦИОНАРНЫЕ ТОЧКИ (вычисляемые) ====================
STATIONARY = [(x, y) for x in (-2, -1, 4) for y in (-3, -1, 5)]

def classify(x, y, eps=1e-9):
    fxx = 3*x*x - 2*x - 10
    fyy = 3*y*y - 2*y - 17
    if fxx > eps and fyy > eps:
        return "min"
    if fxx < -eps and fyy < -eps:
        return "max"
    return "saddle"

def stationary_table():
    rows = [(x, y, f(x, y), classify(x, y)) for (x, y) in STATIONARY]
    rows.sort(key=lambda r: r[2])
    return rows

# ==================== МЕТОДЫ ВЫБОРА ШАГА ====================
def step_line_search(x, y, d):
    fx = f(x, y)
    T = 0.01
    for _ in range(60):
        if f(x - T*d[0], y - T*d[1]) < fx:
            T *= 2.0
        else:
            break
    if T < 1e-10:
        return 1e-10
    phi = lambda t: f(x - t*d[0], y - t*d[1])
    a, b = 0.0, T
    gr = (np.sqrt(5) - 1) / 2
    c = b - gr*(b-a); e = a + gr*(b-a)
    fc, fe = phi(c), phi(e)
    for _ in range(90):
        if fc < fe:
            b, e, fe = e, c, fc
            c = b - gr*(b-a); fc = phi(c)
        else:
            a, c, fc = c, e, fe
            e = a + gr*(b-a); fe = phi(e)
        if b - a < 1e-10:
            break
    return (a + b) / 2

def step_armijo(x, y, d, alpha0=1.0, c1=1e-4, rho=0.5, max_iter=60):
    fx = f(x, y)
    a = alpha0
    d2 = np.dot(d, d)
    for _ in range(max_iter):
        if f(x - a*d[0], y - a*d[1]) <= fx - c1*a*d2:
            return a
        a *= rho
    return a

def step_greedy(x, y, d, alpha0=0.0, alpha_max=5.0, h=0.05):
    fx = f(x, y)
    a = max(alpha0, h) if alpha0 > 0 else h
    while a > 1e-10 and f(x - a*d[0], y - a*d[1]) >= fx:
        a *= 0.5
    if a <= 1e-10:
        return 1e-8
    best = a
    while best < alpha_max:
        a_next = best + h
        if f(x - a_next*d[0], y - a_next*d[1]) < fx:
            best = a_next
        else:
            break
    return best

def step_bb1_guarded(x, y, d, x_prev, y_prev, alpha_prev,
                     c1=1e-4, rho=0.5, alpha_min=1e-5, alpha_max=5.0):
    g = grad(x, y)
    fx = f(x, y)
    d2 = np.dot(d, d)
    if x_prev is None:
        a = 0.05
    else:
        s = np.array([x - x_prev, y - y_prev])
        yv = g - grad(x_prev, y_prev)
        sy = np.dot(s, yv); ss = np.dot(s, s)
        a = ss / sy if sy > 1e-12 * ss else 0.5 * alpha_prev
    a = float(np.clip(a, alpha_min, alpha_max))
    for _ in range(60):
        if f(x - a*d[0], y - a*d[1]) <= fx - c1*a*d2:
            return a
        a *= rho
    return a

# ==================== ГРАДИЕНТНЫЙ СПУСК ====================
TOL = 1e-7   # внутренняя точность (не выносим в интерфейс)

def gradient_descent(method, x0, y0, max_iter=500):
    x, y = float(x0), float(y0)
    path = [(x, y)]
    alphas = []
    x_prev, y_prev = None, None
    alpha_prev = 0.01
    for k in range(max_iter):
        g = grad(x, y)
        if np.linalg.norm(g) < TOL:
            break
        d = g
        if method == "line_search":
            a = step_line_search(x, y, d)
        elif method == "armijo":
            a = step_armijo(x, y, d, alpha0=1.0)
        elif method == "greedy":
            a = step_greedy(x, y, d, alpha0=alpha_prev)
        elif method == "bb":
            a = step_bb1_guarded(x, y, d, x_prev, y_prev, alpha_prev)
        else:
            raise ValueError(method)
        if not np.isfinite(a) or a <= 0:
            break
        xn, yn = x - a*d[0], y - a*d[1]
        if not (np.isfinite(xn) and np.isfinite(yn)):
            break
        x_prev, y_prev = x, y
        x, y = xn, yn
        alpha_prev = a
        alphas.append(a)
        path.append((x, y))
        if np.hypot(x - x_prev, y - y_prev) < TOL:
            break
    return np.array(path), np.array(alphas)

# ==================== ГРАФИЧЕСКИЙ ИНТЕРФЕЙС ====================
class App:
    DEF_X = (-5.5, 7.0)
    DEF_Y = (-6.5, 8.0)

    def __init__(self, root):
        self.root = root
        root.title("Градиентный спуск — вариант 2")
        root.geometry("1400x760")

        ctrl = ttk.Frame(root, padding=8)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(ctrl, text="Метод шага:").grid(row=0, column=0, sticky="w", pady=2)
        self.method_var = tk.StringVar(value="bb")
        methods = [
            ("Линейный поиск (точный)", "line_search"),
            ("Армихо", "armijo"),
            ("Жадный водитель", "greedy"),
            ("Барзилай–Борвейн (BB1+Armijo)", "bb"),
        ]
        for i, (txt, val) in enumerate(methods):
            ttk.Radiobutton(ctrl, text=txt, value=val,
                            variable=self.method_var).grid(row=1+i, column=0, sticky="w")

        ttk.Separator(ctrl).grid(row=5, column=0, columnspan=2, sticky="ew", pady=6)

        ttk.Label(ctrl, text="x0:").grid(row=6, column=0, sticky="w")
        self.x0_var = tk.StringVar(value="0.0")
        ttk.Entry(ctrl, textvariable=self.x0_var, width=10).grid(row=6, column=1, sticky="w")

        ttk.Label(ctrl, text="y0:").grid(row=7, column=0, sticky="w")
        self.y0_var = tk.StringVar(value="0.0")
        ttk.Entry(ctrl, textvariable=self.y0_var, width=10).grid(row=7, column=1, sticky="w")

        ttk.Label(ctrl, text="Макс. итераций:").grid(row=8, column=0, sticky="w")
        self.iter_var = tk.StringVar(value="500")
        ttk.Entry(ctrl, textvariable=self.iter_var, width=10).grid(row=8, column=1, sticky="w")

        ttk.Button(ctrl, text="Запустить", command=self.run).grid(
            row=9, column=0, columnspan=2, sticky="ew", pady=8)

        ttk.Separator(ctrl).grid(row=10, column=0, columnspan=2, sticky="ew", pady=6)

        self.info = tk.Text(ctrl, width=48, height=28, font=("Consolas", 9))
        self.info.grid(row=11, column=0, columnspan=2, sticky="nsew")
        ctrl.rowconfigure(11, weight=1)

        self.fig = Figure(figsize=(9.5, 7), dpi=100)
        self.ax1 = self.fig.add_subplot(131)
        self.ax2 = self.fig.add_subplot(132)
        self.ax3 = self.fig.add_subplot(133)
        self.fig.subplots_adjust(left=0.06, right=0.98, top=0.94,
                                 bottom=0.1, wspace=0.30)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.draw_all(x_lo=self.DEF_X[0], x_hi=self.DEF_X[1],
                      y_lo=self.DEF_Y[0], y_hi=self.DEF_Y[1])

    def draw_levels(self, ax, x_lo, x_hi, y_lo, y_hi, levels_count=35):
        ax.clear()
        xs = np.linspace(x_lo, x_hi, 400)
        ys = np.linspace(y_lo, y_hi, 400)
        X, Y = np.meshgrid(xs, ys)
        Z = f(X, Y)
        lo = np.percentile(Z, 2)
        hi = np.percentile(Z, 92)
        if hi <= lo:
            hi = lo + 1.0
        levels = np.linspace(lo, hi, levels_count)
        ax.contour(X, Y, Z, levels=levels, cmap="viridis", linewidths=0.8)
        ax.contourf(X, Y, Z, levels=60, cmap="viridis", alpha=0.35)
        for (px, py, fp, typ) in stationary_table():
            if x_lo <= px <= x_hi and y_lo <= py <= y_hi:
                marker = "*" if typ == "min" else ("x" if typ == "max" else "d")
                ax.plot(px, py, marker, color="white", ms=9, markeredgecolor="k")
                ax.annotate(f"{typ}", (px, py), textcoords="offset points",
                            xytext=(4, 4), fontsize=7, color="white",
                            bbox=dict(boxstyle="round", fc="black",
                                      alpha=0.55, lw=0))
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.grid(alpha=0.3)

    def draw_all(self, x_lo, x_hi, y_lo, y_hi):
        self.draw_levels(self.ax1, x_lo, x_hi, y_lo, y_hi)
        self.ax1.set_title("Линии уровня и траектория")

        self.ax2.clear()
        self.ax2.set_title("Шаг alpha_k по итерациям")
        self.ax2.set_xlabel("k"); self.ax2.set_ylabel("alpha_k")
        self.ax2.grid(alpha=0.3)

        self.ax3.clear()
        self.ax3.set_title("Крупный план финиша")
        self.ax3.grid(alpha=0.3)

        self.canvas.draw()

    def run(self):
        try:
            x0 = float(self.x0_var.get())
            y0 = float(self.y0_var.get())
            max_it = int(self.iter_var.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Проверьте ввод чисел.")
            return

        method = self.method_var.get()
        path, alphas = gradient_descent(method, x0, y0, max_iter=max_it)
        xf, yf = path[-1]

        x_lo = min(self.DEF_X[0], path[:, 0].min()) - 1.0
        x_hi = max(self.DEF_X[1], path[:, 0].max()) + 1.0
        y_lo = min(self.DEF_Y[0], path[:, 1].min()) - 1.0
        y_hi = max(self.DEF_Y[1], path[:, 1].max()) + 1.0

        # ---- общий вид ----
        self.draw_levels(self.ax1, x_lo, x_hi, y_lo, y_hi)
        self.ax1.set_title("Линии уровня и траектория")
        self.ax1.plot(path[:, 0], path[:, 1], "r.-", ms=3, lw=1.2, label="траектория")
        self.ax1.plot(path[0, 0], path[0, 1], "go", ms=8, label="старт")
        self.ax1.plot(xf, yf, "rX", ms=12, label="финиш")
        self.ax1.legend(loc="upper right", fontsize=7)

        # ---- шаги ----
        self.ax2.clear()
        self.ax2.set_title("Шаг alpha_k по итерациям")
        self.ax2.set_xlabel("k"); self.ax2.set_ylabel("alpha_k")
        self.ax2.grid(alpha=0.3)
        if len(alphas) > 0:
            self.ax2.semilogy(np.arange(1, len(alphas)+1), np.abs(alphas), "b.-")

        # ---- зум ----
        pad = 1.5
        self.draw_levels(self.ax3, xf - pad, xf + pad, yf - pad, yf + pad,
                         levels_count=25)
        self.ax3.set_title(f"Зум: ({xf:.3f}, {yf:.3f})")
        keep = max(2, len(path)//3)
        sub = path[-keep:]
        self.ax3.plot(sub[:, 0], sub[:, 1], "r.-", ms=4, lw=1.4)
        self.ax3.plot(path[0, 0], path[0, 1], "go", ms=8)
        self.ax3.plot(xf, yf, "rX", ms=12)

        self.canvas.draw()

        # ---- текстовая панель ----
        names = {"line_search": "Точный линейный поиск",
                 "armijo": "Армихо",
                 "greedy": "Жадный водитель",
                 "bb": "Барзилай–Борвейн (BB1 + Armijo)"}
        txt = f"Метод: {names[method]}\n"
        txt += f"Старт:  ({x0:.3f}, {y0:.3f}), f = {f(x0, y0):.4f}\n"
        txt += f"Финиш:  ({xf:.4f}, {yf:.4f}), f = {f(xf, yf):.6f}\n"
        txt += f"Итераций: {len(alphas)}\n"
        if len(alphas):
            txt += f"alpha_1 = {alphas[0]:.4g}, alpha_last = {alphas[-1]:.4g}\n"
            txt += f"alpha_min = {alphas.min():.4g}, alpha_max = {alphas.max():.4g}\n"
        txt += f"||grad|| = {np.linalg.norm(grad(xf, yf)):.2e}\n\n"

        rows = stationary_table()
        f_glob = rows[0][2]
        txt += "Стационарные точки (вычислено):\n"
        for (px, py, fp, typ) in rows:
            tag = ""
            if abs(fp - f_glob) < 1e-12:
                tag = "  <-- ГЛОБАЛЬНЫЙ МИНИМУМ"
            elif typ == "min":
                tag = "  (локальный min)"
            txt += f"  ({px:+.0f},{py:+.0f}): f = {fp:+.6f}  [{typ}]{tag}\n"

        txt += "\nБлизость финиша к стационарным точкам:\n"
        best = None
        for (px, py, fp, typ) in rows:
            d = np.hypot(xf - px, yf - py)
            txt += (f"  ({px:+.0f},{py:+.0f}) [{typ}]: "
                    f"dist = {d:.3e}, |f-fin - f*| = {abs(f(xf,yf)-fp):.3e}\n")
            if best is None or d < best[0]:
                best = (d, px, py, fp, typ)
        d_min, px, py, fp, typ = best
        txt += f"\nБлижайшая точка: ({px:+.0f},{py:+.0f}) [{typ}], dist = {d_min:.3e}\n"
        txt += f"f(финиш) - f(ближайшая) = {f(xf,yf) - fp:.3e}\n"
        txt += f"f(финиш) - f(глоб.мин)  = {f(xf,yf) - f_glob:.3e}\n"
        if d_min < 1e-3:
            txt += "=> ТОЧКА В СТАЦИОНАРНОЙ ТОЧКЕ (с точностью 1e-3)\n"
        elif d_min < 1e-1:
            txt += "=> точка очень близко к стационарной\n"
        else:
            txt += "=> спуск НЕ сошёлся к стационарной точке\n"

        self.info.delete("1.0", tk.END)
        self.info.insert(tk.END, txt)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()