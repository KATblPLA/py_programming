# app.py
# Инкрементальное обучение на MNIST: смягчение катастрофического забывания
# Запуск: python app.py
# Зависимости: gradio, numpy, scikit-learn, matplotlib, os

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

warnings.filterwarnings("ignore", category=ConvergenceWarning)

PIC_DIR = "pic"
DATASET_DIR = "dataset"
os.makedirs(PIC_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# -----------------------------
# Загрузка данных (один раз)
# -----------------------------
def load_mnist():
    arff_path = os.path.join(DATASET_DIR, "mnist_784.arff")
    if not os.path.exists(arff_path):
        raise FileNotFoundError(
            f"Скачайте mnist_784.arff в папку {DATASET_DIR}/\n"
            "Ссылка: https://www.openml.org/data/download/52667/mnist_784.arff"
        )
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        parser="liac-arff",
        data_home=DATASET_DIR,
    )
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(int)
    return X, y

X_global, y_global = load_mnist()

# Глобальное разбиение: 80% train, 20% val (стратифицированное)
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
    """
    Возвращает список списков классов (последовательность задач).
    split_mode: "0-4|5-9" или "0-1|2-3|4-5|6-7|8-9"
    repeat_mode: "once", "twice", "cycle10"
    """
    if split_mode == "0-4|5-9":
        groups = [list(range(0, 5)), list(range(5, 10))]
    else:
        groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    if repeat_mode == "once":
        return groups
    elif repeat_mode == "twice":
        return groups * 2
    elif repeat_mode == "cycle10":
        # 10 повторений всей последовательности
        seq = []
        for _ in range(10):
            seq.extend(groups)
        return seq
    else:
        return groups

def evaluate_model(model, Xa_te, ya_te, Xb_te, yb_te, X_val_global, y_val_global, X_test_all, y_test_all):
    a = accuracy_score(ya_te, model.predict(Xa_te)) if len(ya_te) > 0 else 0.0
    b = accuracy_score(yb_te, model.predict(Xb_te)) if len(yb_te) > 0 else 0.0
    g = accuracy_score(y_val_global, model.predict(X_val_global)) if len(y_val_global) > 0 else 0.0
    pred_all = model.predict(X_test_all) if len(y_test_all) > 0 else []
    f1v = f1_score(y_test_all, pred_all, labels=ALL_CLASSES, average=None, zero_division=0) if len(y_test_all) > 0 else np.zeros(10)
    return a, b, g, f1v

# -----------------------------
# Основная функция обучения
# -----------------------------
def run_experiment(
    method,
    split_mode,
    repeat_mode,
    n_epochs,
    replay_frac,
    hidden_size,
    random_state,
):
    """
    method: "simple_replay" | "dynamic_replay" | "no_replay"
    Возвращает: пути к двум графикам, текстовый отчёт
    """
    rng = np.random.RandomState(int(random_state))
    tasks = build_tasks(split_mode, repeat_mode)

    # Подготовка тестовых наборов для каждой группы классов
    # Для простоты: тест по всем классам из val_global
    X_test_all = X_val_global
    y_test_all = y_val_global

    # Для accuracy A и B используем первые два класса из первой группы и второй группы
    # (для отображения на графике)
    # Определяем группы для отображения: первая группа и вторая группа (если есть)
    group_a = tasks[0] if len(tasks) > 0 else []
    group_b = tasks[1] if len(tasks) > 1 else []

    Xa_te, ya_te = split_by_classes(X_val_global, y_val_global, group_a)
    Xb_te, yb_te = split_by_classes(X_val_global, y_val_global, group_b)

    # Пул старых данных для replay (из train)
    # Для каждой задачи будем хранить данные всех предыдущих задач
    task_data = []  # список (X_task, y_task)

    # Инициализация модели
    mlp = MLPClassifier(
        hidden_layer_sizes=(int(hidden_size),),
        batch_size=10000,
        random_state=int(random_state),
        max_iter=1,  # будет вызываться partial_fit
        warm_start=True,
    )

    epochs_list = []
    acc_A, acc_B, acc_gl = [], [], []
    f1_per_class = {d: [] for d in range(10)}
    global_epoch = 0
    classes_seen = np.array([], dtype=int)

    # Проходим по задачам
    for task_idx, task_classes in enumerate(tasks):
        X_task, y_task = split_by_classes(X_train_global, y_train_global, task_classes)
        if len(X_task) == 0:
            continue

        # Обновляем seen classes
        new_classes = np.setdiff1d(task_classes, classes_seen)
        classes_seen = np.union1d(classes_seen, task_classes).astype(int)

        # Сохраняем данные задачи для будущего replay
        task_data.append((X_task, y_task))

        for ep in range(int(n_epochs)):
            # Формируем обучающую выборку для текущей эпохи
            if method == "no_replay" or task_idx == 0:
                X_fit, y_fit = X_task, y_task
            elif method == "simple_replay":
                # Случайный replay из всех предыдущих задач
                replay_X = []
                replay_y = []
                n_total_old = sum(len(d[0]) for d in task_data[:-1])
                n_replay = int(n_total_old * float(replay_frac))
                if n_replay > 0:
                    # Простое случайное подмешивание из всех старых данных
                    for X_old, y_old in task_data[:-1]:
                        take = max(1, int(len(X_old) / n_total_old * n_replay))
                        take = min(take, len(X_old))
                        idx = rng.choice(len(X_old), size=take, replace=False)
                        replay_X.append(X_old[idx])
                        replay_y.append(y_old[idx])
                    X_old_mix = np.vstack(replay_X) if replay_X else np.empty((0, X_task.shape[1]))
                    y_old_mix = np.concatenate(replay_y) if replay_y else np.empty(0, dtype=int)
                    X_fit = np.vstack([X_task, X_old_mix])
                    y_fit = np.concatenate([y_task, y_old_mix])
                else:
                    X_fit, y_fit = X_task, y_task
            elif method == "dynamic_replay":
                # Динамическое подмешивание: только те классы, которые сильнее забываются
                # Оцениваем забывание по F1 на предыдущем шаге (если есть история)
                if len(global_epoch) == 0 or len(f1_per_class[0]) == 0:
                    X_fit, y_fit = X_task, y_task
                else:
                    # Определяем классы, чей F1 упал сильнее всего с предыдущего шага
                    # Берём последние значения F1 для каждого класса
                    prev_f1 = {d: f1_per_class[d][-1] for d in range(10) if len(f1_per_class[d]) > 0}
                    if not prev_f1:
                        X_fit, y_fit = X_task, y_task
                    else:
                        # Классы, которые уже видели, но F1 низкий
                        # Выбираем классы с F1 ниже медианы среди виденных
                        seen_f1 = {d: v for d, v in prev_f1.items() if d in classes_seen and d not in task_classes}
                        if not seen_f1:
                            X_fit, y_fit = X_task, y_task
                        else:
                            med = np.median(list(seen_f1.values()))
                            weak_classes = [d for d, v in seen_f1.items() if v <= med]
                            if not weak_classes:
                                X_fit, y_fit = X_task, y_task
                            else:
                                # Собираем данные только для weak_classes из старых задач
                                replay_X = []
                                replay_y = []
                                n_old_total = sum(len(d[0]) for d in task_data[:-1])
                                n_replay = int(n_old_total * float(replay_frac))
                                for X_old, y_old in task_data[:-1]:
                                    mask = np.isin(y_old, weak_classes)
                                    if mask.sum() == 0:
                                        continue
                                    X_w = X_old[mask]
                                    y_w = y_old[mask]
                                    take = max(1, int(len(X_w) / n_old_total * n_replay)) if n_old_total > 0 else 0
                                    take = min(take, len(X_w))
                                    if take > 0:
                                        idx = rng.choice(len(X_w), size=take, replace=False)
                                        replay_X.append(X_w[idx])
                                        replay_y.append(y_w[idx])
                                if replay_X:
                                    X_old_mix = np.vstack(replay_X)
                                    y_old_mix = np.concatenate(replay_y)
                                    X_fit = np.vstack([X_task, X_old_mix])
                                    y_fit = np.concatenate([y_task, y_old_mix])
                                else:
                                    X_fit, y_fit = X_task, y_task

            # Перемешиваем
            perm = rng.permutation(len(X_fit))
            X_fit, y_fit = X_fit[perm], y_fit[perm]

            # partial_fit
            if global_epoch == 0:
                mlp.partial_fit(X_fit, y_fit, classes=ALL_CLASSES)
            else:
                mlp.partial_fit(X_fit, y_fit)

            # Оценка
            a, b, g, f1v = evaluate_model(
                mlp, Xa_te, ya_te, Xb_te, yb_te, X_val_global, y_val_global, X_test_all, y_test_all
            )
            global_epoch += 1
            epochs_list.append(global_epoch)
            acc_A.append(a)
            acc_B.append(b)
            acc_gl.append(g)
            for d in range(10):
                f1_per_class[d].append(f1v[d])

    # -----------------------------
    # Построение графиков
    # -----------------------------
    # График 1: accuracy
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(epochs_list, acc_A, "o-", label=f"A ({group_a})")
    if len(group_b) > 0:
        ax1.plot(epochs_list, acc_B, "o-", label=f"B ({group_b})")
    ax1.plot(epochs_list, acc_gl, "o-", label="Global")
    # Вертикальные линии на границах задач
    boundary = 0
    for t_idx, t in enumerate(tasks):
        boundary += int(n_epochs)
        if t_idx < len(tasks) - 1:
            ax1.axvline(x=boundary, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Эпоха")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_title(f"Accuracy: {method} | split={split_mode} | repeat={repeat_mode}")
    plt.tight_layout()
    path_acc = os.path.join(PIC_DIR, f"acc_{method}_{split_mode}_{repeat_mode}.png")
    plt.savefig(path_acc, dpi=120)
    plt.close(fig1)

    # График 2: per-class F1
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    cmap = plt.get_cmap("tab10")
    for d in range(10):
        ax2.plot(epochs_list, f1_per_class[d], color=cmap(d), linewidth=2, label=f"цифра {d}")
    boundary = 0
    for t_idx, t in enumerate(tasks):
        boundary += int(n_epochs)
        if t_idx < len(tasks) - 1:
            ax2.axvline(x=boundary, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Эпоха")
    ax2.set_ylabel("F1")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_title(f"Per-class F1: {method} | split={split_mode} | repeat={repeat_mode}")
    plt.tight_layout()
    path_f1 = os.path.join(PIC_DIR, f"f1_{method}_{split_mode}_{repeat_mode}.png")
    plt.savefig(path_f1, dpi=120, bbox_inches="tight")
    plt.close(fig2)

    # -----------------------------
    # Текстовый отчёт
    # -----------------------------
    report = f"Метод: {method}\n"
    report += f"Разбиение: {split_mode}\n"
    report += f"Повторение: {repeat_mode}\n"
    report += f"Эпох на задачу: {n_epochs}\n"
    report += f"Доля replay: {replay_frac}\n"
    report += f"Скрытый слой: {hidden_size}\n"
    report += f"Всего эпох: {global_epoch}\n\n"
    report += "F1 в конце (по классам):\n"
    for d in range(10):
        if len(f1_per_class[d]) > 0:
            report += f"  цифра {d}: {f1_per_class[d][-1]:.3f}\n"
        else:
            report += f"  цифра {d}: N/A\n"

    return path_acc, path_f1, report

# -----------------------------
# Gradio интерфейс
# -----------------------------
def run_from_ui(method, split_mode, repeat_mode, n_epochs, replay_frac, hidden_size, random_state):
    try:
        path_acc, path_f1, report = run_experiment(
            method, split_mode, repeat_mode,
            int(n_epochs), float(replay_frac), int(hidden_size), int(random_state)
        )
        return path_acc, path_f1, report
    except Exception as e:
        return None, None, f"Ошибка: {e}"

with gr.Blocks(title="Catastrophic Forgetting Mitigation") as demo:
    gr.Markdown("# Смягчение катастрофического забывания на MNIST")
    gr.Markdown("Выберите метод, разбиение классов и параметры обучения.")

    with gr.Tabs():
        with gr.Tab("Simple Replay"):
            with gr.Row():
                with gr.Column():
                    sr_split = gr.Radio(
                        choices=["0-4|5-9", "0-1|2-3|4-5|6-7|8-9"],
                        value="0-4|5-9",
                        label="Разбиение классов"
                    )
                    sr_repeat = gr.Radio(
                        choices=["once", "twice", "cycle10"],
                        value="once",
                        label="Повторение последовательности"
                    )
                    sr_epochs = gr.Slider(1, 50, value=30, step=1, label="Эпох на задачу")
                    sr_replay = gr.Slider(0.0, 0.5, value=0.1, step=0.05, label="Доля replay")
                    sr_hidden = gr.Slider(50, 500, value=100, step=50, label="Нейронов в скрытом слое")
                    sr_seed = gr.Number(value=42, label="Random seed")
                    sr_btn = gr.Button("Запустить", variant="primary")
                with gr.Column():
                    sr_acc_img = gr.Image(label="Accuracy")
                    sr_f1_img = gr.Image(label="Per-class F1")
                    sr_text = gr.Textbox(label="Отчёт", lines=15)
            sr_btn.click(
                run_from_ui,
                inputs=[gr.State("simple_replay"), sr_split, sr_repeat, sr_epochs, sr_replay, sr_hidden, sr_seed],
                outputs=[sr_acc_img, sr_f1_img, sr_text]
            )

        with gr.Tab("Dynamic Replay (забываемые классы)"):
            with gr.Row():
                with gr.Column():
                    dr_split = gr.Radio(
                        choices=["0-4|5-9", "0-1|2-3|4-5|6-7|8-9"],
                        value="0-4|5-9",
                        label="Разбиение классов"
                    )
                    dr_repeat = gr.Radio(
                        choices=["once", "twice", "cycle10"],
                        value="once",
                        label="Повторение последовательности"
                    )
                    dr_epochs = gr.Slider(1, 50, value=30, step=1, label="Эпох на задачу")
                    dr_replay = gr.Slider(0.0, 0.5, value=0.1, step=0.05, label="Доля replay")
                    dr_hidden = gr.Slider(50, 500, value=100, step=50, label="Нейронов в скрытом слое")
                    dr_seed = gr.Number(value=42, label="Random seed")
                    dr_btn = gr.Button("Запустить", variant="primary")
                with gr.Column():
                    dr_acc_img = gr.Image(label="Accuracy")
                    dr_f1_img = gr.Image(label="Per-class F1")
                    dr_text = gr.Textbox(label="Отчёт", lines=15)
            dr_btn.click(
                run_from_ui,
                inputs=[gr.State("dynamic_replay"), dr_split, dr_repeat, dr_epochs, dr_replay, dr_hidden, dr_seed],
                outputs=[dr_acc_img, dr_f1_img, dr_text]
            )

        with gr.Tab("No Replay (baseline)"):
            with gr.Row():
                with gr.Column():
                    nr_split = gr.Radio(
                        choices=["0-4|5-9", "0-1|2-3|4-5|6-7|8-9"],
                        value="0-4|5-9",
                        label="Разбиение классов"
                    )
                    nr_repeat = gr.Radio(
                        choices=["once", "twice", "cycle10"],
                        value="once",
                        label="Повторение последовательности"
                    )
                    nr_epochs = gr.Slider(1, 50, value=30, step=1, label="Эпох на задачу")
                    nr_hidden = gr.Slider(50, 500, value=100, step=50, label="Нейронов в скрытом слое")
                    nr_seed = gr.Number(value=42, label="Random seed")
                    nr_btn = gr.Button("Запустить", variant="primary")
                with gr.Column():
                    nr_acc_img = gr.Image(label="Accuracy")
                    nr_f1_img = gr.Image(label="Per-class F1")
                    nr_text = gr.Textbox(label="Отчёт", lines=15)
            nr_btn.click(
                run_from_ui,
                inputs=[gr.State("no_replay"), nr_split, nr_repeat, nr_epochs, gr.State(0.0), nr_hidden, nr_seed],
                outputs=[nr_acc_img, nr_f1_img, nr_text]
            )

if __name__ == "__main__":
    demo.launch()