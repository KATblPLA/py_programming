"""Gradio-приложение для оценки стоимости дома.

Запуск:
    python app/app.py
"""
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import func_module as mu


LABELS = {
    "bedrooms":      {"label": "Спальни (шт.)"},
    "bathrooms":     {"label": "Ванные (шт.)"},
    "sqft_living":   {"label": "Жилая площадь (кв. футы)"},
    "sqft_lot":      {"label": "Площадь участка (кв. футы)"},
    "waterfront":    {"label": "Выход к воде (0/1)",            "info": "1 = есть, 0 = нет"},
    "condition":     {"label": "Состояние (1-5)",               "info": "1 = плохое, 5 = отличное"},
    "grade":         {"label": "Качество (1-13)",               "info": "1 - 13"},
    "sqft_above":    {"label": "Площадь над землёй (кв. футы)", "info": "площадь дома, этажей"},
    "sqft_basement": {"label": "Площадь под землей (кв. футы)", "info": "площадь подвала"},
    "yr_built":      {"label": "Год постройки",                 "info": "1900 - 2015"},
    "yr_renovated":  {"label": "Год ремонта",                   "info": "0 = не было"},
    "lat":           {"label": "Широта (lat)",                  "info": "47.15 - 47.78"},
    "long":          {"label": "Долгота (long)",                "info": "-122.52 - -121.32"},
    "sqft_living15": {"label": "Ср. жилая площадь 15 соседей",  "info": "площадь домов"},
    "sqft_lot15":    {"label": "Ср. площадь участков 15 соседей", "info": "площадь земельных участков"},
}

DEFAULTS = {
    "bedrooms": 3, "bathrooms": 2.25, "sqft_living": 1920, "sqft_lot": 7651,
    "waterfront": 0, "condition": 3, "grade": 7, "sqft_above": 1560,
    "sqft_basement": 0, "yr_built": 1975, "yr_renovated": 0,
    "lat": 47.56, "long": -122.22, "sqft_living15": 1980, "sqft_lot15": 7600,
}


def _label(name: str) -> str:
    """Русская подпись поля.
    Принимает имя признака.
    Возвращает строку-подпись (или само имя, если подписи нет)."""
    return LABELS.get(name, {}).get("label", name)


def _info(name: str) -> str:
    """Подсказка под полем.
    Принимает имя признака.
    Возвращает строку-подсказку (или пустую)."""
    return LABELS.get(name, {}).get("info", "")


def _default(name: str) -> float:
    """Значение по умолчанию.
    Принимает имя признака.
    Возвращает число из DEFAULTS (или 0.0)."""
    return DEFAULTS.get(name, 0.0)


def _scatter_for_ui(feature: str, values: list[float], pred_ln: float) -> plt.Figure:
    """Компактный scatter для UI.
    Принимает имя признака, вектор значений и pred_ln.
    Возвращает Figure с облаком, звездой пользователя и линией регрессии."""
    idx = mu.FEATURES.index(feature)
    x_user = float(values[idx])

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.scatter(mu.TRAIN_DF[feature], mu.TRAIN_DF[mu.TARGET],
               s=10, alpha=0.3, color="gray", label="Обучающая выборка")
    ax.scatter([x_user], [pred_ln], s=120, color="red", marker="*",
               edgecolor="black", zorder=5, label="Ваша точка")

    xs = np.linspace(mu.TRAIN_DF[feature].min(), mu.TRAIN_DF[feature].max(), 100)
    grid = np.tile(mu.TRAIN_DF[mu.FEATURES].mean().values, (len(xs), 1))
    grid[:, idx] = xs
    ax.plot(xs, mu.MODEL.predict(grid), color="green", lw=2, label="Линия регрессии")

    ax.set_xlabel(feature)
    ax.set_ylabel(mu.TARGET)
    ax.set_title(f"{mu.TARGET} vs {feature}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def run(*values):
    """Обработчик кнопки: считает цену и строит три графика.
    Принимает 15 значений из полей формы (в порядке FEATURES).
    Возвращает кортеж: (pred_ln, цена-строкой, Figure, Figure, Figure)."""
    values_list = list(values)

    pred_ln = mu.predict(values_list)                 # log(price)
    pred_price = float(np.exp(pred_ln))               # обратно в доллары
    price_str = f"${pred_price:,.0f}"

    figs_top = mu.make_scatters(values_list, pred_ln)
    fig_yr = _scatter_for_ui("yr_built", values_list, pred_ln)

    out = (pred_ln, price_str, figs_top[0], figs_top[1], fig_yr)

    # закрываем фигуры после отдачи в Gradio, чтобы не копить память
    for f in (figs_top[0], figs_top[1], fig_yr):
        plt.close(f)

    return out


input_components = [
    gr.Number(label=_label(name), info=_info(name), value=_default(name))
    for name in mu.FEATURES
]

out_log = gr.Number(label="Логарифм цены (price_ln)")
out_price = gr.Textbox(label="Примерная стоимость дома ($)")
out_plot1 = gr.Plot(label=f"Зависимость от {mu.TOP_FEATURES[0]}")
out_plot2 = gr.Plot(label=f"Зависимость от {mu.TOP_FEATURES[1]}")
out_plot_yr = gr.Plot(label="Зависимость от yr_built")


with gr.Blocks(theme=gr.themes.Citrus()) as demo:
    gr.Markdown(f"""
    # Оценка стоимости дома
    **R² на выборке:** `{mu.meta['r2_test']:.4f}` |
    **Топ-признаки:** `{', '.join(mu.TOP_FEATURES)}`
    """)

    # Уравнение регрессии — внутри Blocks, иначе не отрендерится
    
    gr.Markdown(mu.latex_equation())
    gr.Markdown(
            "Уравнение предсказывает `price_ln = ln(price)`. "
            "Чтобы получить цену в долларах, приложение берёт `exp(pred_ln)`."
        )

    with gr.Row():
        # ЛЕВЫЙ БЛОК — ввод
        with gr.Column(scale=1):
            gr.Markdown("### Параметры дома")
            with gr.Group():
                with gr.Row():
                    # левая колонка полей
                    with gr.Column():
                        for comp in input_components[:len(input_components) // 2 + 1]:
                            comp.render()
                    # правая колонка полей
                    with gr.Column():
                        for comp in input_components[len(input_components) // 2 + 1:]:
                            comp.render()

            submit_btn = gr.Button("Рассчитать стоимость", variant="primary", size="lg")

        # ПРАВЫЙ БЛОК — вывод
        with gr.Column(scale=1):
            gr.Markdown("### Результат")
            with gr.Row():
                with gr.Column():
                    out_log.render()
                with gr.Column():
                    out_price.render()

            gr.Markdown("### Графики зависимостей")
            with gr.Row():
                out_plot1.render()
                out_plot2.render()
            with gr.Row():
                out_plot_yr.render()

    # Порядок входов = порядок input_components (он же порядок FEATURES)
    submit_btn.click(
        fn=run,
        inputs=input_components,
        outputs=[out_log, out_price, out_plot1, out_plot2, out_plot_yr],
    )


if __name__ == "__main__":
    demo.launch()