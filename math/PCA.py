import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox

class PCAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PCA — Проекции на главные плоскости")
        self.root.geometry("1400x800")

        # ---------- Панель управления (слева) ----------
        ctrl = ttk.Frame(root, padding=10)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(ctrl, text="Настройки", font=("Arial", 10, "bold")).pack(pady=5)

        ttk.Label(ctrl, text="Seed:").pack(anchor="w")
        self.seed_var = tk.StringVar(value="42")
        ttk.Entry(ctrl, textvariable=self.seed_var, width=10).pack(anchor="w", pady=2)

        ttk.Label(ctrl, text="Кол-во точек:").pack(anchor="w")
        self.n_points_var = tk.StringVar(value="10")
        ttk.Entry(ctrl, textvariable=self.n_points_var, width=10).pack(anchor="w", pady=2)

        ttk.Button(ctrl, text="Пересчитать PCA", command=self.run_pca).pack(pady=10, fill=tk.X)

        ttk.Separator(ctrl).pack(fill=tk.X, pady=5)

        self.info_text = tk.Text(ctrl, width=46, height=35, font=("Consolas", 9))
        self.info_text.pack(fill=tk.BOTH, expand=True)

        # ---------- Графическая область (справа) ----------
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Вкладка 1: Три 2D-проекции (быстрая)
        self.tab_2d = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_2d, text="2D Проекции (PC1-PC2, PC1-PC3, PC2-PC3)")
        
        self.fig2d = Figure(figsize=(8, 8), dpi=100)
        # Создаем 3 подграфика в сетке 2x2 (один квадрант оставим пустым или используем для легенды)
        self.ax_12 = self.fig2d.add_subplot(221)
        self.ax_13 = self.fig2d.add_subplot(222)
        self.ax_23 = self.fig2d.add_subplot(223)
        # 224 оставим пустым, чтобы не перегружать
        self.fig2d.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08, wspace=0.3, hspace=0.3)
        
        self.canvas2d = FigureCanvasTkAgg(self.fig2d, master=self.tab_2d)
        self.canvas2d.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Вкладка 2: 3D-график (для общего понимания)
        self.tab_3d = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_3d, text="3D Исходные данные")
        
        self.fig3d = Figure(figsize=(7, 7), dpi=100)
        self.ax3d = self.fig3d.add_subplot(111, projection='3d')
        self.fig3d.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)
        
        self.canvas3d = FigureCanvasTkAgg(self.fig3d, master=self.tab_3d)
        self.canvas3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Первый расчет
        self.run_pca()

    # ========================================================
    def generate_data(self, n, seed):
        rng = np.random.default_rng(seed)
        v1 = np.array([1.0, 1.5, 2.0]); v1 = v1 / np.linalg.norm(v1)
        t = rng.uniform(-5, 5, size=n)
        points = np.outer(t, v1) + rng.normal(0, 0.5, size=(n, 3))
        return points

    # ========================================================
    def run_pca(self):
        try:
            seed = int(self.seed_var.get())
            n = int(self.n_points_var.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Seed и кол-во точек — целые числа.")
            return
        if n < 3:
            messagebox.showerror("Ошибка", "Нужно хотя бы 3 точки.")
            return

        # 1. Данные и PCA
        X = self.generate_data(n, seed)
        mean_X = X.mean(axis=0)
        Xc = X - mean_X
        C = np.cov(Xc, rowvar=False)

        eigvals, eigvecs = np.linalg.eigh(C)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        Z = Xc @ eigvecs
        total = eigvals.sum()
        expl = eigvals / total * 100

        # 2. Отрисовка трех проекций
        self.draw_projections(Z, eigvecs, expl, n)

        # 3. Отрисовка 3D
        self.draw_3d(X, mean_X, eigvecs, n)

        # 4. Текст
        self.update_text(X, mean_X, C, eigvals, eigvecs, Z, expl)

    # ========================================================
    def draw_projections(self, Z, eigvecs, expl, n):
        """Рисует три 2D-проекции с векторами признаков (biplot)."""
        # Очистка осей
        self.ax_12.clear(); self.ax_13.clear(); self.ax_23.clear()
        
        # Данные для стрелок (нагрузки)
        # Нагрузка = собственный вектор * sqrt(собственное значение)
        # Для проекции PC1-PC2 нам нужны 1-й и 2-й столбцы
        lam = np.sqrt(np.maximum(Z.var(axis=0, ddof=1), 1e-12))
        loadings = eigvecs * lam  # 3x3: строки — признаки, столбцы — PC
        
        colors = ['crimson', 'darkorange', 'purple']
        labels = ['x', 'y', 'z']

        # ---- 1. PC1-PC2 ----
        ax = self.ax_12
        ax.scatter(Z[:, 0], Z[:, 1], s=50, color='steelblue', edgecolors='k', zorder=3)
        for i in range(n):
            ax.annotate(str(i+1), (Z[i, 0], Z[i, 1]), textcoords="offset points", xytext=(4, 4), fontsize=7)
        
        # Стрелки (biplot)
        for j in range(3):
            dx, dy = loadings[j, 0], loadings[j, 1]
            ax.annotate("", xy=(dx, dy), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=colors[j], lw=1.5))
            ax.text(dx*1.1, dy*1.1, labels[j], color=colors[j], fontsize=10, fontweight="bold")
            
        ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
        ax.set_xlabel(f"PC1 ({expl[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({expl[1]:.1f}%)")
        ax.set_title("Проекция PC1–PC2")
        ax.grid(alpha=0.3)
        ax.set_aspect('equal', adjustable='datalim')

        # ---- 2. PC1-PC3 ----
        ax = self.ax_13
        ax.scatter(Z[:, 0], Z[:, 2], s=50, color='steelblue', edgecolors='k', zorder=3)
        for i in range(n):
            ax.annotate(str(i+1), (Z[i, 0], Z[i, 2]), textcoords="offset points", xytext=(4, 4), fontsize=7)
            
        for j in range(3):
            dx, dy = loadings[j, 0], loadings[j, 2]
            ax.annotate("", xy=(dx, dy), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=colors[j], lw=1.5))
            ax.text(dx*1.1, dy*1.1, labels[j], color=colors[j], fontsize=10, fontweight="bold")
            
        ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
        ax.set_xlabel(f"PC1 ({expl[0]:.1f}%)")
        ax.set_ylabel(f"PC3 ({expl[2]:.1f}%)")
        ax.set_title("Проекция PC1–PC3")
        ax.grid(alpha=0.3)
        ax.set_aspect('equal', adjustable='datalim')

        # ---- 3. PC2-PC3 ----
        ax = self.ax_23
        ax.scatter(Z[:, 1], Z[:, 2], s=50, color='steelblue', edgecolors='k', zorder=3)
        for i in range(n):
            ax.annotate(str(i+1), (Z[i, 1], Z[i, 2]), textcoords="offset points", xytext=(4, 4), fontsize=7)
            
        for j in range(3):
            dx, dy = loadings[j, 1], loadings[j, 2]
            ax.annotate("", xy=(dx, dy), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=colors[j], lw=1.5))
            ax.text(dx*1.1, dy*1.1, labels[j], color=colors[j], fontsize=10, fontweight="bold")
            
        ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
        ax.set_xlabel(f"PC2 ({expl[1]:.1f}%)")
        ax.set_ylabel(f"PC3 ({expl[2]:.1f}%)")
        ax.set_title("Проекция PC2–PC3")
        ax.grid(alpha=0.3)
        ax.set_aspect('equal', adjustable='datalim')

        self.canvas2d.draw_idle()

    # ========================================================
    def draw_3d(self, X, mean_X, eigvecs, n):
        ax = self.ax3d
        ax.clear()

        # Точки
        ax.plot(X[:, 0], X[:, 1], X[:, 2], 'o', color='steelblue', ms=6, label='точки')

        # Центр
        ax.plot([mean_X[0]], [mean_X[1]], [mean_X[2]], 'X', color='black', ms=10)

        # Оси PC
        scale = np.max(np.abs(X - mean_X)) * 1.2
        colors = ['red', 'green', 'purple']
        for i in range(3):
            v = eigvecs[:, i]
            ax.plot([mean_X[0], mean_X[0] + v[0]*scale],
                    [mean_X[1], mean_X[1] + v[1]*scale],
                    [mean_X[2], mean_X[2] + v[2]*scale],
                    color=colors[i], lw=2.5, label=f'PC{i+1}')

        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title('Исходные данные и новые оси')
        ax.grid(False)
        ax.legend(loc='upper right', fontsize=8, frameon=False)

        self.canvas3d.draw_idle()

    # ========================================================
    def update_text(self, X, mean_X, C, eigvals, eigvecs, Z, expl):
        self.info_text.delete("1.0", tk.END)
        txt = "--- ИСХОДНЫЕ ДАННЫЕ X ---\n"
        txt += str(np.round(X, 3)) + "\n\n"

        txt += "--- СРЕДНИЕ ---\n"
        txt += str(np.round(mean_X, 3)) + "\n\n"

        txt += "--- КОВАРИАЦИОННАЯ МАТРИЦА C ---\n"
        
        txt += str(np.round(C, 4)) + "\n\n"

        txt += "--- СОБСТВЕННЫЕ ЗНАЧЕНИЯ (дисперсия) ---\n"
        for i, v in enumerate(eigvals):
            txt += f"λ{i+1} = {v:.5f}   ({expl[i]:.2f}%)\n"

        txt += "\n--- СОБСТВЕННЫЕ ВЕКТОРЫ (PC1, PC2, PC3) ---\n"
        txt += str(np.round(eigvecs, 4)) + "\n\n"

        txt += "--- ПРОЕКЦИИ Z ---\n"
        txt += str(np.round(Z, 3)) + "\n"

        self.info_text.insert(tk.END, txt)


if __name__ == "__main__":
    root = tk.Tk()
    app = PCAApp(root)
    root.mainloop()