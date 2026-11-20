# Gradio-приложение: оценка стоимости дома

Веб-форма: 15 параметров дома → цена, `price_ln`, уравнение в LaTeX и три scatter-графика.

## Установка

```bash
pip install uv
```

```bash
uv pip install -r pyproject.toml --all-extras
```


## Запуск

```bash
.venv\Scripts\activate
python app/app.py
```

Или через VS Code: открыть `app/app.py` → ▶ / `F5`. Адрес — `http://127.0.0.1:7860`, остановка — `Ctrl+C`.

## Как работает Gradio

- **`gr.Blocks()`** — контейнер страницы; внутри `gr.Row` / `gr.Column` / `gr.Group` / `gr.Accordion`.
- **Входы** — `gr.Number` в порядке `mu.FEATURES`.
- **Выходы** — `gr.Number` (лог. цены), `gr.Textbox` (цена), `gr.Plot` (matplotlib `Figure`).
- **`gr.Markdown`** рендерит LaTeX между `$$...$$` — так показывается уравнение.
- **`submit_btn.click(fn=run, inputs=..., outputs=...)`** — при клике Gradio собирает значения полей в кортеж, вызывает `run(*values)` и раскладывает результат по `outputs` в том же порядке.

## Форматы данных

**`data/subsample_2.csv`** — обучающая выборка. Столбцы: `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `waterfront`, `condition`, `grade`, `sqft_above`, `sqft_basement`, `yr_built`, `yr_renovated`, `lat`, `long`, `sqft_living15`, `sqft_lot15` + целевая `price_ln = ln(price)`.

**Вход приложения** — 15 чисел в порядке `mu.FEATURES`; неверная длина → `ValueError`.
**Выход** — `price_ln` (float), цена строкой (`$1,234,567`), три `Figure`.

## Пример входных данных

Значения по умолчанию — «средний» дом: `bedrooms=3`, `bathrooms=2.25`, `sqft_living=1920`, `sqft_lot=7651`, `waterfront=0`, `condition=3`, `grade=7`, `sqft_above=1560`, `sqft_basement=0`, `yr_built=1975`, `yr_renovated=0`, `lat=47.56`, `long=-122.22`, `sqft_living15=1980`, `sqft_lot15=7600`.
Ожидаемо: `price_ln ≈ 12.9…13.0`, цена ≈ `$400 000 – $500 000`.

## Что показывают графики (scatter)

**Scatter** — облако точек: каждый дом из обучающей выборки (`X` — признак, `Y` — `price_ln`).
На графике: серые точки — выборка; красная звезда — ваша точка; зелёная линия — зависимость от одного признака при **средних** остальных.
