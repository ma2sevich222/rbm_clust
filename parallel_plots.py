#######################################################
# Copyright © 2021-2099 Ekosphere. All rights reserved
# Author: Evgeny Matusevich
# Contacts: <ma2sevich222@gmail.com>
# File: parallel.py
#######################################################
import pandas as pd
import plotly.express as px

file_root = "outputs/rbm_GC_2020_2022_15min_17_08_2022"
filename = "intermedia_GC_2020_2022_15min.xlsx"
name="RBM"
final_df = pd.read_excel(f"{file_root}/{filename}")  # загружаем результаты  анализа

df_plot = final_df[
    [
        "values_0",
        "values_1",
        "# Trades",
        "patch",
        "hidden_units",
        "train_window",
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
        "forward_window": "forward_window (bars)",
    },
    range_color=[df_plot["values_0"].min(), df_plot["values_0"].max()],
    color_continuous_scale=px.colors.sequential.Viridis,
    title=f"{name}_{filename[:-5]}",
)

fig.write_html(f"RBM_{filename[:-5]}.html")  # сохраняем в файл
fig.show()



