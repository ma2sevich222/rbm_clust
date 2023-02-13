
import pandas as pd
import plotly.express as px

file_root = "outputs/lean_sel_parV32_rbm_GC_2019_2022_30min_04_10_2022"
filename = "intermedia_GC_2019_2022_30min.xlsx"
name = "Apate_V3.2"
final_df = pd.read_excel(f"{file_root}/{filename}")  # загружаем результаты  анализа

df_plot = final_df[
    [
        "values_0",
        "values_1",
        "# Trades",
        "patch",
        "hidden_units",
        "train_window",
        "train_backtest_window",
        "forward_window",

    ]
]

fig = px.parallel_coordinates(
    df_plot,
    color="values_0",
    labels={
        "values_0": "Net Profit ($)",
        "values_1": "Sharpe Ratio",
        "# Trades": "Trades",
        "patch": "patch(bars)",
        "hidden_units":"n_hidden_units",
        "train_window": "train_window (bars)",
        "train_backtest_window": "train_backtest_window",
        "forward_window": "forward_window (days)",
    },
    range_color=[df_plot["values_0"].min(), df_plot["values_0"].max()],
    color_continuous_scale=px.colors.sequential.Viridis,
    title=f"{name}_{filename[:-5]} "
)

fig.write_html(f"{name}_{filename[:-5]}.html")  # сохраняем в файл
fig.show()



