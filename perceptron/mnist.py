# gradio_app.py
# Запуск: python gradio_app.py   (НЕ через streamlit!)
# Зависимости: gradio, numpy, scikit-learn, matplotlib, torch

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
import gradio as gr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=ConvergenceWarning)

PIC_DIR = "pic"
DATASET_DIR = "dataset"
os.makedirs(PIC_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# -----------------------------
# Загрузка данных
# -----------------------------
def load_mnist():
    arff_path = os.path.join(DATASET_DIR, "mnist_784.arff")
    if not os.path.exists(arff_path):
        raise FileNotFoundError(
            f"Скачайте mnist_784.arff в папку {DATASET_DIR}/\n"
            "Ссылка: https://www.openml.org/data/download/52667/mnist_784.arff"
        )
    mnist = fetch_openml(
        "mnist_784", version=1, as_frame=False,
        parser="liac-arff", data_home=DATASET_DIR,
    )
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(int)
    return X, y

print("Загрузка MNIST...")
X_global, y_global = load_mnist()
print("MNIST загружен.")

X_train_global, X_val_global, y_train_global, y_val_global = train_test_split(
    X_global, y_global, test_size=0.2, random_state=42, stratify=y_global
)

ALL_CLASSES = np.arange(10)

# -----------------------------
# Вспомогательные функции
# -----------------------------
def split_by_classes(X, y, classes):
    mask = np.isin(y, classes)
    return X[mask], y[mask]

def build_tasks(split_mode, repeat_mode):
    if split_mode == "0-4|5-9":
        groups = [list(range(0, 5)), list(range(5, 10))]
    else:
        groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    if repeat_mode == "once":
        return groups
    elif repeat_mode == "twice":
        return groups * 2
    elif repeat_mode == "cycle10":
        seq = []
        for _ in range(10):
            seq.extend(groups)
        return seq
    return groups

# -----------------------------
# PyTorch: динамическая сеть
# -----------------------------
class DynamicMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=(256, 128), num_classes=10):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.feature = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, num_classes)

    def forward(self, x):
        return self.classifier(self.feature(x))

    def expand_output(self, new_num_classes):
        old = self.classifier.out_features
        if new_num_classes <= old:
            return
        new_layer = nn.Linear(self.classifier.in_features, new_num_classes)
        with torch.no_grad():
            new_layer.weight[:old] = self.classifier.weight
            new_layer.bias[:old] = self.classifier.bias
        self.classifier = new_layer

def train_torch_epoch(model, X, y, optimizer, criterion, batch_size=256, device="cpu"):
    model.train()
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(dataset)

def predict_torch(model, X, device="cpu"):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X).to(device)).argmax(dim=1).cpu().numpy()

def evaluate_torch(model, Xa_te, ya_te, Xb_te, yb_te, X_val_global, y_val_global, X_test_all, y_test_all, device="cpu"):
    a = accuracy_score(ya_te, predict_torch(model, Xa_te, device)) if len(ya_te) > 0 else 0.0
    b = accuracy_score(yb_te, predict_torch(model, Xb_te, device)) if len(yb_te) > 0 else 0.0
    g = accuracy_score(y_val_global, predict_torch(model, X_val_global, device)) if len(y_val_global) > 0 else 0.0
    pred_all = predict_torch(model, X_test_all, device) if len(y_test_all) > 0 else []
    f1v = f1_score(y_test_all, pred_all, labels=ALL_CLASSES, average=None, zero_division=0) if len(y_test_all) > 0 else np.zeros(10)
    return a, b, g, f1v

# -----------------------------
# Основная функция (sklearn)
# -----------------------------
def run_experiment_sklearn(method, split_mode, repeat_mode, n_epochs, replay_frac, hidden_size, random_state):
    rng = np.random.RandomState(int(random_state))
    tasks = build_tasks(split_mode, repeat_mode)

    X_test_all = X_val_global
    y_test_all = y_val_global

    group_a = tasks[0] if len(tasks) > 0 else []
    group_b = tasks[1] if len(tasks) > 1 else []

    Xa_te, ya_te = split_by_classes(X_val_global, y_val_global, group_a)
    Xb_te, yb_te = split_by_classes(X_val_global, y_val_global, group_b)

    task_data = []
    mlp = MLPClassifier(
        hidden_layer_sizes=(int(hidden_size),),
        batch_size=10000,
        random_state=int(random_state),
        max_iter=1,
        warm_start=True,
    )

    epochs_list, acc_A, acc_B, acc_gl = [], [], [], []
    f1_per_class = {d: [] for d in range(10)}
    global_epoch = 0
    classes_seen = np.array([], dtype=int)

    for task_idx, task_classes in enumerate(tasks):
        X_task, y_task = split_by_classes(X_train_global, y_train_global, task_classes)
        if len(X_task) == 0:
            continue
        classes_seen = np.union1d(classes_seen, task_classes).astype(int)
        task_data.append((X_task, y_task))

        for ep in range(int(n_epochs)):
            if method == "no_replay" or task_idx == 0:
                X_fit, y_fit = X_task, y_task
            elif method == "simple_replay":
                replay_X, replay_y = [], []
                n_total_old = sum(len(d[0]) for d in task_data[:-1])
                n_replay = int(n_total_old * float(replay_frac))
                if n_replay > 0:
                    for X_old, y_old in task_data[:-1]:
                        take = max(1, int(len(X_old) / n_total_old * n_replay))
                        take = min(take, len(X_old))
                        idx = rng.choice(len(X_old), size=take, replace=False)
                        replay_X.append(X_old[idx]); replay_y.append(y_old[idx])
                    X_fit = np.vstack([X_task] + replay_X)
                    y_fit = np.concatenate([y_task] + replay_y)
                else:
                    X_fit, y_fit = X_task, y_task
            elif method == "dynamic_replay":
                if len(f1_per_class[0]) == 0:
                    X_fit, y_fit = X_task, y_task
                else:
                    prev_f1 = {d: f1_per_class[d][-1] for d in range(10) if len(f1_per_class[d]) > 0}
                    seen_f1 = {d: v for d, v in prev_f1.items() if d in classes_seen and d not in task_classes}
                    if not seen_f1:
                        X_fit, y_fit = X_task, y_task
                    else:
                        med = np.median(list(seen_f1.values()))
                        weak_classes = [d for d, v in seen_f1.items() if v <= med]
                        if not weak_classes:
                            X_fit, y_fit = X_task, y_task
                        else:
                            replay_X, replay_y = [], []
                            n_old_total = sum(len(d[0]) for d in task_data[:-1])
                            n_replay = int(n_old_total * float(replay_frac))
                            for X_old, y_old in task_data[:-1]:
                                mask = np.isin(y_old, weak_classes)
                                if mask.sum() == 0:
                                    continue
                                X_w, y_w = X_old[mask], y_old[mask]
                                take = max(1, int(len(X_w) / n_old_total * n_replay)) if n_old_total > 0 else 0
                                take = min(take, len(X_w))
                                if take > 0:
                                    idx = rng.choice(len(X_w), size=take, replace=False)
                                    replay_X.append(X_w[idx]); replay_y.append(y_w[idx])
                            if replay_X:
                                X_fit = np.vstack([X_task] + replay_X)
                                y_fit = np.concatenate([y_task] + replay_y)
                            else:
                                X_fit, y_fit = X_task, y_task

            perm = rng.permutation(len(X_fit))
            X_fit, y_fit = X_fit[perm], y_fit[perm]

            mlp.partial_fit(X_fit, y_fit, classes=ALL_CLASSES)

            a = accuracy_score(ya_te, mlp.predict(Xa_te)) if len(ya_te) > 0 else 0.0
            b = accuracy_score(yb_te, mlp.predict(Xb_te)) if len(yb_te) > 0 else 0.0
            g = accuracy_score(y_val_global, mlp.predict(X_val_global)) if len(y_val_global) > 0 else 0.0
            pred_all = mlp.predict(X_test_all) if len(y_test_all) > 0 else []
            f1v = f1_score(y_test_all, pred_all, labels=ALL_CLASSES, average=None, zero_division=0) if len(y_test_all) > 0 else np.zeros(10)

            global_epoch += 1
            epochs_list.append(global_epoch)
            acc_A.append(a); acc_B.append(b); acc_gl.append(g)
            for d in range(10):
                f1_per_class[d].append(f1v[d])

    return epochs_list, acc_A, acc_B, acc_gl, f1_per_class, tasks, group_a, group_b, global_epoch

# -----------------------------
# Основная функция (PyTorch)
# -----------------------------
def run_experiment_torch(method, split_mode, repeat_mode, n_epochs, replay_frac, hidden_dims_str, random_state):
    torch.manual_seed(int(random_state))
    np.random.seed(int(random_state))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(int(random_state))
    tasks = build_tasks(split_mode, repeat_mode)

    X_test_all = X_val_global
    y_test_all = y_val_global

    group_a = tasks[0] if len(tasks) > 0 else []
    group_b = tasks[1] if len(tasks) > 1 else []

    Xa_te, ya_te = split_by_classes(X_val_global, y_val_global, group_a)
    Xb_te, yb_te = split_by_classes(X_val_global, y_val_global, group_b)

    hidden_dims = tuple(int(x.strip()) for x in hidden_dims_str.split(",") if x.strip().isdigit())
    if not hidden_dims:
        hidden_dims = (256, 128)

    model = DynamicMLP(input_dim=784, hidden_dims=hidden_dims, num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    task_data = []
    epochs_list, acc_A, acc_B, acc_gl = [], [], [], []
    f1_per_class = {d: [] for d in range(10)}
    global_epoch = 0
    classes_seen = np.array([], dtype=int)

    for task_idx, task_classes in enumerate(tasks):
        X_task, y_task = split_by_classes(X_train_global, y_train_global, task_classes)
        if len(X_task) == 0:
            continue
        classes_seen = np.union1d(classes_seen, task_classes).astype(int)
        task_data.append((X_task, y_task))

        for ep in range(int(n_epochs)):
            if method == "no_replay" or task_idx == 0:
                X_fit, y_fit = X_task, y_task
            elif method == "simple_replay":
                replay_X, replay_y = [], []
                n_total_old = sum(len(d[0]) for d in task_data[:-1])
                n_replay = int(n_total_old * float(replay_frac))
                if n_replay > 0:
                    for X_old, y_old in task_data[:-1]:
                        take = max(1, int(len(X_old) / n_total_old * n_replay))
                        take = min(take, len(X_old))
                        idx = rng.choice(len(X_old), size=take, replace=False)
                        replay_X.append(X_old[idx]); replay_y.append(y_old[idx])
                    X_fit = np.vstack([X_task] + replay_X)
                    y_fit = np.concatenate([y_task] + replay_y)
                else:
                    X_fit, y_fit = X_task, y_task
            elif method == "dynamic_replay":
                if len(f1_per_class[0]) == 0:
                    X_fit, y_fit = X_task, y_task
                else:
                    prev_f1 = {d: f1_per_class[d][-1] for d in range(10) if len(f1_per_class[d]) > 0}
                    seen_f1 = {d: v for d, v in prev_f1.items() if d in classes_seen and d not in task_classes}
                    if not seen_f1:
                        X_fit, y_fit = X_task, y_task
                    else:
                        med = np.median(list(seen_f1.values()))
                        weak_classes = [d for d, v in seen_f1.items() if v <= med]
                        if not weak_classes:
                            X_fit, y_fit = X_task, y_task
                        else:
                            replay_X, replay_y = [], []
                            n_old_total = sum(len(d[0]) for d in task_data[:-1])
                            n_replay = int(n_old_total * float(replay_frac))
                            for X_old, y_old in task_data[:-1]:
                                mask = np.isin(y_old, weak_classes)
                                if mask.sum() == 0:
                                    continue
                                X_w, y_w = X_old[mask], y_old[mask]
                                take = max(1, int(len(X_w) / n_old_total * n_replay)) if n_old_total > 0 else 0
                                take = min(take, len(X_w))
                                if take > 0:
                                    idx = rng.choice(len(X_w), size=take, replace=False)
                                    replay_X.append(X_w[idx]); replay_y.append(y_w[idx])
                            if replay_X:
                                X_fit = np.vstack([X_task] + replay_X)
                                y_fit = np.concatenate([y_task] + replay_y)
                            else:
                                X_fit, y_fit = X_task, y_task

            perm = rng.permutation(len(X_fit))
            X_fit, y_fit = X_fit[perm], y_fit[perm]

            train_torch_epoch(model, X_fit, y_fit, optimizer, criterion, device=device)

            a, b, g, f1v = evaluate_torch(model, Xa_te, ya_te, Xb_te, yb_te, X_val_global, y_val_global, X_test_all, y_test_all, device)
            global_epoch += 1
            epochs_list.append(global_epoch)
            acc_A.append(a); acc_B.append(b); acc_gl.append(g)
            for d in range(10):
                f1_per_class[d].append(f1v[d])

    return epochs_list, acc_A, acc_B, acc_gl, f1_per_class, tasks, group_a, group_b, global_epoch

# -----------------------------
# Построение графиков и отчёт
# -----------------------------
def make_plots_and_report(epochs_list, acc_A, acc_B, acc_gl, f1_per_class, tasks, group_a, group_b, global_epoch, n_epochs, method, split_mode, repeat_mode, backend, replay_frac, size_info):
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(epochs_list, acc_A, "o-", label=f"A ({group_a})")
    if len(group_b) > 0:
        ax1.plot(epochs_list, acc_B, "o-", label=f"B ({group_b})")
    ax1.plot(epochs_list, acc_gl, "o-", label="Global")
    boundary = 0
    for t_idx in range(len(tasks)):
        boundary += int(n_epochs)
        if t_idx < len(tasks) - 1:
            ax1.axvline(x=boundary, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Эпоха"); ax1.set_ylabel("Accuracy"); ax1.set_ylim(0, 1.05)
    ax1.legend(); ax1.grid(alpha=0.3)
    ax1.set_title(f"Accuracy: {method} | {backend} | split={split_mode} | repeat={repeat_mode}")
    plt.tight_layout()
    path_acc = os.path.join(PIC_DIR, f"acc_{backend}_{method}_{split_mode}_{repeat_mode}.png")
    plt.savefig(path_acc, dpi=120); plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(14, 8))
    cmap = plt.get_cmap("tab10")
    for d in range(10):
        ax2.plot(epochs_list, f1_per_class[d], color=cmap(d), linewidth=2, label=f"цифра {d}")
    boundary = 0
    for t_idx in range(len(tasks)):
        boundary += int(n_epochs)
        if t_idx < len(tasks) - 1:
            ax2.axvline(x=boundary, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Эпоха"); ax2.set_ylabel("F1"); ax2.set_ylim(0, 1.05)
    ax2.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_title(f"Per-class F1: {method} | {backend} | split={split_mode} | repeat={repeat_mode}")
    plt.tight_layout()
    path_f1 = os.path.join(PIC_DIR, f"f1_{backend}_{method}_{split_mode}_{repeat_mode}.png")
    plt.savefig(path_f1, dpi=120, bbox_inches="tight"); plt.close(fig2)

    report = f"Бэкенд: {backend}\nМетод: {method}\nРазбиение: {split_mode}\nПовторение: {repeat_mode}\n"
    report += f"Эпох на задачу: {n_epochs}\nДоля подмешивания: {replay_frac}\n{size_info}\nВсего эпох: {global_epoch}\n\n"
    report += "F1 в конце (по классам):\n"
    for d in range(10):
        if len(f1_per_class[d]) > 0:
            report += f"  цифра {d}: {f1_per_class[d][-1]:.3f}\n"
        else:
            report += f"  цифра {d}: N/A\n"
    return path_acc, path_f1, report

# -----------------------------
# Gradio-обёртка
# -----------------------------
def run_from_ui(method, backend, split_mode, repeat_mode, n_epochs, replay_frac, sklearn_hidden, torch_hidden, random_state):
    try:
        if backend == "scikit-learn":
            epochs_list, acc_A, acc_B, acc_gl, f1_per_class, tasks, group_a, group_b, global_epoch = run_experiment_sklearn(
                method, split_mode, repeat_mode, int(n_epochs), float(replay_frac), int(sklearn_hidden), int(random_state)
            )
            size_info = f"Скрытый слой (sklearn): {sklearn_hidden}"
        else:
            epochs_list, acc_A, acc_B, acc_gl, f1_per_class, tasks, group_a, group_b, global_epoch = run_experiment_torch(
                method, split_mode, repeat_mode, int(n_epochs), float(replay_frac), torch_hidden, int(random_state)
            )
            size_info = f"Слои (PyTorch): {torch_hidden}"
        return make_plots_and_report(epochs_list, acc_A, acc_B, acc_gl, f1_per_class, tasks, group_a, group_b, global_epoch, n_epochs, method, split_mode, repeat_mode, backend, replay_frac, size_info)
    except Exception as e:
        return None, None, f"Ошибка: {e}"

# -----------------------------
# Интерфейс Gradio
# -----------------------------
with gr.Blocks(title="Смягчение катастрофического забывания") as demo:
    gr.Markdown("# Смягчение катастрофического забывания на MNIST")
    gr.Markdown("Выберите метод, бэкенд (scikit-learn / PyTorch) и параметры обучения.")

    with gr.Tabs():
        # ---------- Простое подмешивание ----------
        with gr.Tab("Простое подмешивание"):
            with gr.Row():
                with gr.Column():
                    sr_backend = gr.Radio(choices=["scikit-learn", "PyTorch"], value="scikit-learn", label="Бэкенд")
                    sr_split = gr.Radio(choices=["0-4|5-9", "0-1|2-3|4-5|6-7|8-9"], value="0-4|5-9", label="Разбиение классов")
                    sr_repeat = gr.Radio(choices=["once", "twice", "cycle10"], value="once", label="Повторение")
                    sr_epochs = gr.Slider(1, 50, value=30, step=1, label="Эпох на задачу")
                    sr_replay = gr.Slider(0.0, 0.5, value=0.1, step=0.05, label="Доля подмешивания")
                    sr_sk_hidden = gr.Slider(50, 500, value=100, step=50, label="Скрытый слой (sklearn)")
                    sr_torch_hidden = gr.Textbox(value="256,128", label="Слои PyTorch (через запятую)")
                    sr_seed = gr.Number(value=42, label="Random seed")
                    sr_btn = gr.Button("Запустить", variant="primary")
                with gr.Column():
                    sr_acc_img = gr.Image(label="Accuracy")
                    sr_f1_img = gr.Image(label="F1 по классам")
                    sr_text = gr.Textbox(label="Отчёт", lines=15)
            sr_btn.click(run_from_ui,
                inputs=[gr.State("simple_replay"), sr_backend, sr_split, sr_repeat, sr_epochs, sr_replay, sr_sk_hidden, sr_torch_hidden, sr_seed],
                outputs=[sr_acc_img, sr_f1_img, sr_text])

        # ---------- Динамическое подмешивание ----------
        with gr.Tab("Динамическое подмешивание (забываемые классы)"):
            with gr.Row():
                with gr.Column():
                    dr_backend = gr.Radio(choices=["scikit-learn", "PyTorch"], value="scikit-learn", label="Бэкенд")
                    dr_split = gr.Radio(choices=["0-4|5-9", "0-1|2-3|4-5|6-7|8-9"], value="0-4|5-9", label="Разбиение классов")
                    dr_repeat = gr.Radio(choices=["once", "twice", "cycle10"], value="once", label="Повторение")
                    dr_epochs = gr.Slider(1, 50, value=30, step=1, label="Эпох на задачу")
                    dr_replay = gr.Slider(0.0, 0.5, value=0.1, step=0.05, label="Доля подмешивания")
                    dr_sk_hidden = gr.Slider(50, 500, value=100, step=50, label="Скрытый слой (sklearn)")
                    dr_torch_hidden = gr.Textbox(value="256,128", label="Слои PyTorch (через запятую)")
                    dr_seed = gr.Number(value=42, label="Random seed")
                    dr_btn = gr.Button("Запустить", variant="primary")
                with gr.Column():
                    dr_acc_img = gr.Image(label="Accuracy")
                    dr_f1_img = gr.Image(label="F1 по классам")
                    dr_text = gr.Textbox(label="Отчёт", lines=15)
            dr_btn.click(run_from_ui,
                inputs=[gr.State("dynamic_replay"), dr_backend, dr_split, dr_repeat, dr_epochs, dr_replay, dr_sk_hidden, dr_torch_hidden, dr_seed],
                outputs=[dr_acc_img, dr_f1_img, dr_text])

        # ---------- Без подмешивания ----------
        with gr.Tab("Без подмешивания (baseline)"):
            with gr.Row():
                with gr.Column():
                    nr_backend = gr.Radio(choices=["scikit-learn", "PyTorch"], value="scikit-learn", label="Бэкенд")
                    nr_split = gr.Radio(choices=["0-4|5-9", "0-1|2-3|4-5|6-7|8-9"], value="0-4|5-9", label="Разбиение классов")
                    nr_repeat = gr.Radio(choices=["once", "twice", "cycle10"], value="once", label="Повторение")
                    nr_epochs = gr.Slider(1, 50, value=30, step=1, label="Эпох на задачу")
                    nr_sk_hidden = gr.Slider(50, 500, value=100, step=50, label="Скрытый слой (sklearn)")
                    nr_torch_hidden = gr.Textbox(value="256,128", label="Слои PyTorch (через запятую)")
                    nr_seed = gr.Number(value=42, label="Random seed")
                    nr_btn = gr.Button("Запустить", variant="primary")
                with gr.Column():
                    nr_acc_img = gr.Image(label="Accuracy")
                    nr_f1_img = gr.Image(label="F1 по классам")
                    nr_text = gr.Textbox(label="Отчёт", lines=15)
            nr_btn.click(run_from_ui,
                inputs=[gr.State("no_replay"), nr_backend, nr_split, nr_repeat, nr_epochs, gr.State(0.0), nr_sk_hidden, nr_torch_hidden, nr_seed],
                outputs=[nr_acc_img, nr_f1_img, nr_text])

if __name__ == "__main__":
    demo.launch()