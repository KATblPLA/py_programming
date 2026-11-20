import gradio as gr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
import warnings
import io

warnings.filterwarnings("ignore", category=ConvergenceWarning)

global_data = {'X': None, 'y': None}

def load_mnist_from_internet():
    try:
        print("Loading MNIST...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, cache=True)
        X = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(int)
        global_data['X'] = X
        global_data['y'] = y
        return "Loaded {} samples, classes: {}".format(len(X), np.unique(y))
    except Exception as e:
        return "Error: {}".format(str(e))

def load_csv_file(file):
    if file is None:
        return "No file selected"
    try:
        df = pd.read_csv(file.name)
        y = df.iloc[:, 0].values.astype(int)
        X = df.iloc[:, 1:].values.astype(np.float32) / 255.0
        global_data['X'] = X
        global_data['y'] = y
        return "Loaded {} samples, classes: {}".format(len(X), np.unique(y))
    except Exception as e:
        return "Error: {}".format(str(e))

def prepare_data_simple(X, y):
    mask_04 = y < 5
    mask_59 = y >= 5
    
    X_04, y_04 = X[mask_04], y[mask_04]
    X_59, y_59 = X[mask_59], y[mask_59]
    
    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(X_04, y_04, test_size=0.2, random_state=42, stratify=y_04)
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(X_59, y_59, test_size=0.2, random_state=42, stratify=y_59)
    X_val, _, y_val, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return {
        'Xa_tr': Xa_tr, 'Xa_te': Xa_te, 'ya_tr': ya_tr, 'ya_te': ya_te,
        'Xb_tr': Xb_tr, 'Xb_te': Xb_te, 'yb_tr': yb_tr, 'yb_te': yb_te,
        'X_val': X_val, 'y_val': y_val,
        'X_test_all': np.vstack([Xa_te, Xb_te]),
        'y_test_all': np.concatenate([ya_te, yb_te])
    }

def run_method_1(hidden_units, batch_size, epochs, replay_frac):
    if global_data['X'] is None:
        return None, None, {"status": "Load data first!"}
    
    X, y = global_data['X'], global_data['y']
    data = prepare_data_simple(X, y)
    
    mlp = MLPClassifier(hidden_layer_sizes=(hidden_units,), batch_size=batch_size, random_state=42)
    N_EPOCHS = int(epochs)
    REPLAY_FRAC = replay_frac
    rng = np.random.RandomState(42)
    ALL_CLASSES = np.arange(10)
    
    X_old_pool, y_old_pool = data['Xa_tr'], data['ya_tr']
    N_OLD = len(X_old_pool)
    
    epochs_list, acc_A, acc_B, acc_gl = [], [], [], []
    f1_per_class = {d: [] for d in range(10)}
    
    def evaluate():
        a = accuracy_score(data['ya_te'], mlp.predict(data['Xa_te']))
        b = accuracy_score(data['yb_te'], mlp.predict(data['Xb_te']))
        g = accuracy_score(data['y_val'], mlp.predict(data['X_val']))
        return a, b, g
    
    def compute_f1():
        pred = mlp.predict(data['X_test_all'])
        return f1_score(data['y_test_all'], pred, labels=ALL_CLASSES, average=None, zero_division=0)
    
    for i in range(N_EPOCHS):
        if i == 0:
            mlp.partial_fit(data['Xa_tr'], data['ya_tr'], classes=ALL_CLASSES)
        else:
            mlp.partial_fit(data['Xa_tr'], data['ya_tr'])
        
        a, b, g = evaluate()
        f1v = compute_f1()
        epochs_list.append(len(epochs_list) + 1)
        acc_A.append(a)
        acc_B.append(b)
        acc_gl.append(g)
        for d in range(10):
            f1_per_class[d].append(f1v[d])
    
    n_replay = int(N_OLD * REPLAY_FRAC)
    for i in range(N_EPOCHS):
        idx = rng.choice(N_OLD, size=n_replay, replace=False)
        X_mix = np.vstack([data['Xb_tr'], X_old_pool[idx]])
        y_mix = np.concatenate([data['yb_tr'], y_old_pool[idx]])
        perm = rng.permutation(len(X_mix))
        mlp.partial_fit(X_mix[perm], y_mix[perm])
        
        a, b, g = evaluate()
        f1v = compute_f1()
        epochs_list.append(len(epochs_list) + 1)
        acc_A.append(a)
        acc_B.append(b)
        acc_gl.append(g)
        for d in range(10):
            f1_per_class[d].append(f1v[d])
    
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    N = len(acc_A) // 2
    ax1.axvspan(0.5, N + 0.5, color="lightblue", alpha=0.2)
    ax1.axvspan(N + 0.5, 2*N + 0.5, color="lightcoral", alpha=0.2)
    ax1.plot(epochs_list, acc_A, "o-", label="A (0-4)", markersize=4)
    ax1.plot(epochs_list, acc_B, "s-", label="B (5-9)", markersize=4)
    ax1.plot(epochs_list, acc_gl, "D-", label="Global", markersize=4, linewidth=2)
    ax1.axvline(x=N + 0.5, color="black", linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_title("Method 1: Simple Replay")
    plt.tight_layout()
    
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("tab10")
    ax2.axvspan(0.5, N + 0.5, color="lightblue", alpha=0.2)
    ax2.axvspan(N + 0.5, 2*N + 0.5, color="lightcoral", alpha=0.2)
    ax2.axvline(x=N + 0.5, color="black", linestyle="--")
    for d in range(10):
        ax2.plot(epochs_list, f1_per_class[d], color=cmap(d), linewidth=2, label="digit {}".format(d))
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(alpha=0.3)
    ax2.set_title("Method 1: F1 per Class")
    plt.tight_layout()
    
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', dpi=100, bbox_inches='tight')
    buf1.seek(0)
    
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', dpi=100, bbox_inches='tight')
    buf2.seek(0)
    
    plt.close(fig1)
    plt.close(fig2)
    
    f1_s1 = [f1_per_class[d][N_EPOCHS - 1] for d in range(10)]
    f1_s2 = [f1_per_class[d][-1] for d in range(10)]
    
    metrics = {
        "status": "Done! Acc A: {:.3f}, Acc B: {:.3f}".format(acc_A[-1], acc_B[-1]),
        "F1_Stage1": {str(d): round(f1_s1[d], 3) for d in range(10)},
        "F1_Stage2": {str(d): round(f1_s2[d], 3) for d in range(10)},
        "Delta": {str(d): round(f1_s2[d]-f1_s1[d], 3) for d in range(10)}
    }
    
    return buf1.getvalue(), buf2.getvalue(), metrics

def run_method_2(hidden_units, batch_size, epochs, replay_frac):
    if global_data['X'] is None:
        return None, None, {"status": "Load data first!"}
    
    X, y = global_data['X'], global_data['y']
    data = prepare_data_simple(X, y)
    
    mlp = MLPClassifier(hidden_layer_sizes=(hidden_units,), batch_size=batch_size, random_state=42)
    N_EPOCHS = int(epochs)
    TOTAL_REPLAY_FRAC = replay_frac
    rng = np.random.RandomState(42)
    ALL_CLASSES = np.arange(10)
    
    X_old_pool, y_old_pool = data['Xa_tr'], data['ya_tr']
    N_OLD = len(X_old_pool)
    
    epochs_list, acc_A, acc_B, acc_gl = [], [], [], []
    f1_per_class = {d: [] for d in range(10)}
    
    def evaluate():
        a = accuracy_score(data['ya_te'], mlp.predict(data['Xa_te']))
        b = accuracy_score(data['yb_te'], mlp.predict(data['Xb_te']))
        g = accuracy_score(data['y_val'], mlp.predict(data['X_val']))
        return a, b, g
    
    def compute_f1():
        pred = mlp.predict(data['X_test_all'])
        return f1_score(data['y_test_all'], pred, labels=ALL_CLASSES, average=None, zero_division=0)
    
    def get_forgetting_scores(f1_history):
        scores = {}
        for d in range(5):
            if len(f1_history[d]) >= 2:
                peak_f1 = max(f1_history[d][:N_EPOCHS])
                current_f1 = f1_history[d][-1]
                forgetting = peak_f1 - current_f1
                scores[d] = max(forgetting, 0)
            else:
                scores[d] = 0
        return scores
    
    for i in range(N_EPOCHS):
        if i == 0:
            mlp.partial_fit(data['Xa_tr'], data['ya_tr'], classes=ALL_CLASSES)
        else:
            mlp.partial_fit(data['Xa_tr'], data['ya_tr'])
        
        a, b, g = evaluate()
        f1v = compute_f1()
        epochs_list.append(len(epochs_list) + 1)
        acc_A.append(a)
        acc_B.append(b)
        acc_gl.append(g)
        for d in range(10):
            f1_per_class[d].append(f1v[d])
    
    total_replay_samples = int(N_OLD * TOTAL_REPLAY_FRAC)
    
    for i in range(N_EPOCHS):
        forgetting = get_forgetting_scores(f1_per_class)
        total_forgetting = sum(forgetting.values())
        
        if total_forgetting > 0:
            replay_counts = {}
            for d in range(5):
                proportion = forgetting[d] / total_forgetting
                replay_counts[d] = int(total_replay_samples * proportion)
            
            current_total = sum(replay_counts.values())
            if current_total < total_replay_samples:
                diff = total_replay_samples - current_total
                max_class = max(forgetting, key=forgetting.get)
                replay_counts[max_class] += diff
        
        replay_indices = []
        for d in range(5):
            if replay_counts.get(d, 0) > 0:
                class_mask = y_old_pool == d
                class_indices = np.where(class_mask)[0]
                n_samples = min(replay_counts[d], len(class_indices))
                if n_samples > 0:
                    selected = rng.choice(class_indices, size=n_samples, replace=False)
                    replay_indices.extend(selected)
        
        if replay_indices:
            X_replay = X_old_pool[replay_indices]
            y_replay = y_old_pool[replay_indices]
            X_mix = np.vstack([data['Xb_tr'], X_replay])
            y_mix = np.concatenate([data['yb_tr'], y_replay])
        else:
            X_mix, y_mix = data['Xb_tr'], data['yb_tr']
        
        perm = rng.permutation(len(X_mix))
        mlp.partial_fit(X_mix[perm], y_mix[perm])
        
        a, b, g = evaluate()
        f1v = compute_f1()
        epochs_list.append(len(epochs_list) + 1)
        acc_A.append(a)
        acc_B.append(b)
        acc_gl.append(g)
        for d in range(10):
            f1_per_class[d].append(f1v[d])
    
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    N = len(acc_A) // 2
    ax1.axvspan(0.5, N + 0.5, color="lightblue", alpha=0.2)
    ax1.axvspan(N + 0.5, 2*N + 0.5, color="lightcoral", alpha=0.2)
    ax1.plot(epochs_list, acc_A, "o-", label="A (0-4)", markersize=4)
    ax1.plot(epochs_list, acc_B, "s-", label="B (5-9)", markersize=4)
    ax1.plot(epochs_list, acc_gl, "D-", label="Global", markersize=4, linewidth=2)
    ax1.axvline(x=N + 0.5, color="black", linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_title("Method 2: Forgetting-based Replay")
    plt.tight_layout()
    
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("tab10")
    ax2.axvspan(0.5, N + 0.5, color="lightblue", alpha=0.2)
    ax2.axvspan(N + 0.5, 2*N + 0.5, color="lightcoral", alpha=0.2)
    ax2.axvline(x=N + 0.5, color="black", linestyle="--")
    for d in range(10):
        ax2.plot(epochs_list, f1_per_class[d], color=cmap(d), linewidth=2, label="digit {}".format(d))
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(alpha=0.3)
    ax2.set_title("Method 2: F1 per Class")
    plt.tight_layout()
    
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', dpi=100, bbox_inches='tight')
    buf1.seek(0)
    
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', dpi=100, bbox_inches='tight')
    buf2.seek(0)
    
    plt.close(fig1)
    plt.close(fig2)
    
    f1_s1 = [f1_per_class[d][N_EPOCHS - 1] for d in range(10)]
    f1_s2 = [f1_per_class[d][-1] for d in range(10)]
    
    metrics = {
        "status": "Done! Acc A: {:.3f}, Acc B: {:.3f}".format(acc_A[-1], acc_B[-1]),
        "F1_Stage1": {str(d): round(f1_s1[d], 3) for d in range(10)},
        "F1_Stage2": {str(d): round(f1_s2[d], 3) for d in range(10)},
        "Delta": {str(d): round(f1_s2[d]-f1_s1[d], 3) for d in range(10)}
    }
    
    return buf1.getvalue(), buf2.getvalue(), metrics

def run_method_3(hidden_units, batch_size, epochs_stage1, epochs_stage2, min_replay, max_replay):
    if global_data['X'] is None:
        return None, None, {"status": "Load data first!"}
    
    X, y = global_data['X'], global_data['y']
    
    mask_04 = y < 5
    mask_5 = y == 5
    mask_69 = y >= 6
    
    X_04, y_04 = X[mask_04], y[mask_04]
    X_5, y_5 = X[mask_5], y[mask_5]
    X_69, y_69 = X[mask_69], y[mask_69]
    
    X_04_tr, X_04_te, y_04_tr, y_04_te = train_test_split(X_04, y_04, test_size=0.2, random_state=42, stratify=y_04)
    X_5_tr, X_5_te, y_5_tr, y_5_te = train_test_split(X_5, y_5, test_size=0.2, random_state=42)
    X_69_tr, X_69_te, y_69_tr, y_69_te = train_test_split(X_69, y_69, test_size=0.2, random_state=42, stratify=y_69)
    
    X_test_all = np.vstack([X_04_te, X_5_te, X_69_te])
    y_test_all = np.concatenate([y_04_te, y_5_te, y_69_te])
    
    X_old_pool, y_old_pool = X_04_tr, y_04_tr
    N_OLD = len(X_old_pool)
    
    mlp = MLPClassifier(hidden_layer_sizes=(hidden_units,), batch_size=batch_size, random_state=42)
    N_EPOCHS_STAGE1 = int(epochs_stage1)
    N_EPOCHS_STAGE2 = int(epochs_stage2)
    rng = np.random.RandomState(42)
    ALL_CLASSES = np.arange(10)
    
    epochs_list, acc_04, acc_5, acc_69, acc_gl = [], [], [], [], []
    f1_per_class = {d: [] for d in range(10)}
    
    def evaluate():
        a = accuracy_score(y_04_te, mlp.predict(X_04_te))
        b = accuracy_score(y_5_te, mlp.predict(X_5_te))
        c = accuracy_score(y_69_te, mlp.predict(X_69_te))
        g = accuracy_score(y_test_all, mlp.predict(X_test_all))
        return a, b, c, g
    
    def compute_f1():
        pred = mlp.predict(X_test_all)
        return f1_score(y_test_all, pred, labels=ALL_CLASSES, average=None, zero_division=0)
    
    def get_dynamic_replay_frac(epoch, total_epochs):
        progress = epoch / total_epochs
        frac = max_replay - (max_replay - min_replay) * progress
        return frac
    
    for i in range(N_EPOCHS_STAGE1):
        if i == 0:
            mlp.partial_fit(X_04_tr, y_04_tr, classes=ALL_CLASSES)
        else:
            mlp.partial_fit(X_04_tr, y_04_tr)
        
        a, b, c, g = evaluate()
        f1v = compute_f1()
        epochs_list.append(len(epochs_list) + 1)
        acc_04.append(a)
        acc_5.append(b)
        acc_69.append(c)
        acc_gl.append(g)
        for d in range(10):
            f1_per_class[d].append(f1v[d])
    
    X_05_tr = np.vstack([X_04_tr, X_5_tr])
    y_05_tr = np.concatenate([y_04_tr, y_5_tr])
    
    for i in range(N_EPOCHS_STAGE2):
        replay_frac = get_dynamic_replay_frac(i, N_EPOCHS_STAGE2)
        n_replay = int(N_OLD * replay_frac)
        
        idx = rng.choice(N_OLD, size=n_replay, replace=False)
        X_mix = np.vstack([X_05_tr, X_old_pool[idx]])
        y_mix = np.concatenate([y_05_tr, y_old_pool[idx]])
        perm = rng.permutation(len(X_mix))
        mlp.partial_fit(X_mix[perm], y_mix[perm])
        
        a, b, c, g = evaluate()
        f1v = compute_f1()
        epochs_list.append(len(epochs_list) + 1)
        acc_04.append(a)
        acc_5.append(b)
        acc_69.append(c)
        acc_gl.append(g)
        for d in range(10):
            f1_per_class[d].append(f1v[d])
    
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    N1 = N_EPOCHS_STAGE1
    ax1.axvspan(0.5, N1 + 0.5, color="lightblue", alpha=0.2)
    ax1.axvspan(N1 + 0.5, N1 + N_EPOCHS_STAGE2 + 0.5, color="lightcoral", alpha=0.2)
    ax1.plot(epochs_list, acc_04, "o-", label="0-4", markersize=4)
    ax1.plot(epochs_list, acc_5, "s-", label="5", markersize=4)
    ax1.plot(epochs_list, acc_69, "^-", label="6-9", markersize=4)
    ax1.plot(epochs_list, acc_gl, "D-", label="Global", markersize=4, linewidth=2)
    ax1.axvline(x=N1 + 0.5, color="black", linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_title("Method 3: Dynamic Replay")
    plt.tight_layout()
    
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("tab10")
    ax2.axvspan(0.5, N1 + 0.5, color="lightblue", alpha=0.2)
    ax2.axvspan(N1 + 0.5, N1 + N_EPOCHS_STAGE2 + 0.5, color="lightcoral", alpha=0.2)
    ax2.axvline(x=N1 + 0.5, color="black", linestyle="--")
    for d in range(10):
        ax2.plot(epochs_list, f1_per_class[d], color=cmap(d), linewidth=2, label="digit {}".format(d))
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(alpha=0.3)
    ax2.set_title("Method 3: F1 per Class")
    plt.tight_layout()
    
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', dpi=100, bbox_inches='tight')
    buf1.seek(0)
    
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', dpi=100, bbox_inches='tight')
    buf2.seek(0)
    
    plt.close(fig1)
    plt.close(fig2)
    
    f1_s1 = [f1_per_class[d][N_EPOCHS_STAGE1 - 1] for d in range(10)]
    f1_s2 = [f1_per_class[d][-1] for d in range(10)]
    
    metrics = {
        "status": "Done! Acc 0-4: {:.3f}, Acc 5: {:.3f}".format(acc_04[-1], acc_5[-1]),
        "F1_Stage1": {str(d): round(f1_s1[d], 3) for d in range(10)},
        "F1_Stage2": {str(d): round(f1_s2[d], 3) for d in range(10)},
        "Delta": {str(d): round(f1_s2[d]-f1_s1[d], 3) for d in range(10)}
    }
    
    return buf1.getvalue(), buf2.getvalue(), metrics

with gr.Blocks(title="Catastrophic Forgetting Mitigation", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Методы смягчения катастрофического забывания")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Загрузка данных")
            
            data_source = gr.Radio(
                choices=["MNIST (internet)", "Local CSV"],
                value="MNIST (internet)",
                label="Data source"
            )
            
            mnist_btn = gr.Button("Load MNIST from internet", variant="primary")
            csv_upload = gr.File(label="Or upload CSV file", file_types=[".csv"])
            
            load_status = gr.Textbox(label="Status", interactive=False)
    
    with gr.Tabs():
        with gr.TabItem("1. Simple Replay"):
            gr.Markdown("### Method 1: Simple Random Replay")
            
            with gr.Row():
                with gr.Column():
                    m1_hidden = gr.Slider(50, 200, value=100, step=10, label="Hidden units")
                    m1_batch = gr.Dropdown([100, 1000, 5000, 10000], value=10000, label="Batch size")
                with gr.Column():
                    m1_epochs = gr.Slider(10, 50, value=20, step=5, label="Epochs per stage")
                    m1_replay = gr.Slider(0.05, 0.30, value=0.10, step=0.05, label="Replay fraction")
            
            m1_run_btn = gr.Button("Run Method 1", variant="primary")
            
            with gr.Row():
                m1_plot1 = gr.Image(label="Accuracy")
                m1_plot2 = gr.Image(label="F1 per Class")
            
            m1_output = gr.JSON(label="Metrics")
        
        with gr.TabItem("2. Forgetting-based"):
            gr.Markdown("### Method 2: Forgetting-based Replay")
            
            with gr.Row():
                with gr.Column():
                    m2_hidden = gr.Slider(50, 200, value=100, step=10, label="Hidden units")
                    m2_batch = gr.Dropdown([100, 1000, 5000, 10000], value=10000, label="Batch size")
                with gr.Column():
                    m2_epochs = gr.Slider(10, 50, value=20, step=5, label="Epochs per stage")
                    m2_replay = gr.Slider(0.05, 0.30, value=0.10, step=0.05, label="Replay fraction")
            
            m2_run_btn = gr.Button("Run Method 2", variant="primary")
            
            with gr.Row():
                m2_plot1 = gr.Image(label="Accuracy")
                m2_plot2 = gr.Image(label="F1 per Class")
            
            m2_output = gr.JSON(label="Metrics")
        
        with gr.TabItem("3. Dynamic Replay"):
            gr.Markdown("### Method 3: Dynamic Replay")
            
            with gr.Row():
                with gr.Column():
                    m3_hidden = gr.Slider(50, 200, value=100, step=10, label="Hidden units")
                    m3_batch = gr.Dropdown([100, 1000, 5000, 10000], value=10000, label="Batch size")
                with gr.Column():
                    m3_epochs1 = gr.Slider(10, 40, value=20, step=5, label="Epochs stage 1")
                    m3_epochs2 = gr.Slider(10, 50, value=30, step=5, label="Epochs stage 2")
            
            with gr.Row():
                m3_min_replay = gr.Slider(0.02, 0.10, value=0.05, step=0.01, label="Min replay frac")
                m3_max_replay = gr.Slider(0.10, 0.30, value=0.15, step=0.01, label="Max replay frac")
            
            m3_run_btn = gr.Button("Run Method 3", variant="primary")
            
            with gr.Row():
                m3_plot1 = gr.Image(label="Accuracy")
                m3_plot2 = gr.Image(label="F1 per Class")
            
            m3_output = gr.JSON(label="Metrics")
    
    mnist_btn.click(
        fn=load_mnist_from_internet,
        outputs=[load_status]
    )
    
    csv_upload.change(
        fn=load_csv_file,
        inputs=[csv_upload],
        outputs=[load_status]
    )
    
    m1_run_btn.click(
        fn=run_method_1,
        inputs=[m1_hidden, m1_batch, m1_epochs, m1_replay],
        outputs=[m1_plot1, m1_plot2, m1_output]
    )
    
    m2_run_btn.click(
        fn=run_method_2,
        inputs=[m2_hidden, m2_batch, m2_epochs, m2_replay],
        outputs=[m2_plot1, m2_plot2, m2_output]
    )
    
    m3_run_btn.click(
        fn=run_method_3,
        inputs=[m3_hidden, m3_batch, m3_epochs1, m3_epochs2, m3_min_replay, m3_max_replay],
        outputs=[m3_plot1, m3_plot2, m3_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
