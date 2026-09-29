import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml

# 1. Реальные данные
d = fetch_openml('heart-disease', version=1, as_frame=True).frame
d.columns = d.columns.str.replace('\ufeff', '').str.strip()

# 2. Признаки и возраст
cols = ['age','sex','trestbps','chol','fbs','restecg',
        'thalach','exang','oldpeak','slope','ca','thal']
X = d[cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
age = pd.to_numeric(d['age'], errors='coerce').fillna(0).values

# 3. ОДНА задача для обеих групп: наличие болезни
y = (pd.to_numeric(d['target'], errors='coerce').fillna(0) > 0).astype(int).values

# 4. Разделение по возрасту + 80/20
old_mask   = age >= 55
young_mask = age < 55

Xo_tr, Xo_te, yo_tr, yo_te = train_test_split(
    X[old_mask], y[old_mask], test_size=0.2, random_state=42)
Xy_tr, Xy_te, yy_tr, yy_te = train_test_split(
    X[young_mask], y[young_mask], test_size=0.2, random_state=42)

# 5. Масштабирование (для MLP обязательно)
sc = StandardScaler().fit(np.vstack([Xo_tr, Xy_tr]))
Xo_tr, Xo_te = sc.transform(Xo_tr), sc.transform(Xo_te)
Xy_tr, Xy_te = sc.transform(Xy_tr), sc.transform(Xy_te)

# 6. MLP: обучение на стариках → ДОобучение на молодых → проверка на стариках
mlp = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    max_iter=500,
    warm_start=True,      # ← ключ: сохраняет веса между вызовами fit
    random_state=42,
)

# --- этап 1: обучение на стариках ---
mlp.fit(Xo_tr, yo_tr)
before = accuracy_score(yo_te, mlp.predict(Xo_te))
print(f"После обучения на стариках:  точность на стариках = {before:.3f}")

# --- этап 2: дообучение на молодых (те же веса, новая порция данных) ---
mlp.fit(Xy_tr, yy_tr)     # warm_start=True → продолжает с текущих весов
after = accuracy_score(yo_te, mlp.predict(Xo_te))
print(f"После дообучения на молодых: точность на стариках = {after:.3f}")

print(f"\nИзменение: {before:.3f} → {after:.3f} "
      f"({(after-before)*100:+.1f} п.п.)")