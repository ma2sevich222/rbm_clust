import pandas as pd
import plotly.express as px
from backtesting import Backtest
import backtesting._plotting as plt_backtesting
import os
from utilits.new_stop_loss_lazy_strategy import LazyStrategy


file_root = "outputs/lean_1_11_bt_switched_rbm_GC_2019_2022_30min_07_09_2022"
filename = "8_signals_GC_2019_2022_30min_train_window8800forward_window440_patch42.csv"
name = "stop loss selection"
out_root = f"select_stop_for_{filename[:-4]}"
os.mkdir(f"outputs/{out_root}")
Signals_df = pd.read_csv(f"{file_root}/{filename}")
N = 1000

plt_backtesting._MAX_CANDLES = 200_000
pd.pandas.set_option("display.max_columns", None)
pd.set_option("expand_frame_repr", False)
pd.options.display.expand_frame_repr = False
pd.set_option("display.precision", 2)


""" Откроем файл с разметкой нейронки """

Signals_df.set_index("Datetime", inplace=True)
Signals_df.index = pd.to_datetime(Signals_df.index)
Signals_df = Signals_df.sort_index()

""" Параметры тестирования """

deposit = 100000  # сумма одного контракта GC & CL
comm = 4.62  # GC - комиссия для золота
# comm = 4.52  # CL - комиссия для нейти

""" Тестирвоание """
df_stats = pd.DataFrame()

for i in range(0, N):
    stop_l = -i * 0.1
    bt = Backtest(
        Signals_df,
        strategy=LazyStrategy,
        cash=100000,
        commission_type="absolute",
        commission=comm,
        features_coeff=10,
        exclusive_orders=True,
    )
    if i == 0:
        stats = bt.run(clearing=True, time_frame=30)
    else:
        stats = bt.run(stop_loss=stop_l, take_profit=50.8, clearing=True, time_frame=15)
    """if (
            stats["Return (Ann.) [%]"] > 0
        ):  # будем показывать и сохранять только доходные разметки
            bt.plot(
                plot_volume=True,
                relative_equity=True,
                filename=f"{DESTINATION_ROOT}/{out_root}/bt_plot_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}_step{step}_{buy_before * step}_{sell_after * step}.html",
            )
        stats.to_csv(
            f"{DESTINATION_ROOT}/{out_root}/stats_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}_step{step}_{buy_before * step}_{sell_after * step}.txt"
        )"""

    df_stats = df_stats.append(stats, ignore_index=True)
    df_stats.loc[i, "NET_PROFIT [$]"] = (
        df_stats.loc[i, "Equity Final [$]"]
        - deposit
        - df_stats.loc[i, "# Trades"] * comm
    )
    df_stats.loc[i, "STOP_LOSS"] = stop_l
    df_stats.loc[i, "NET_PROFIT_TO_MAX_DRAWDOWN"] = df_stats.loc[
        i, "NET_PROFIT [$]"
    ] / (-df_stats.loc[i, "Max. Drawdown [%]"] * 100000 / 100)



df_plot = df_stats[
    [
        "NET_PROFIT [$]",
        "Sharpe Ratio",
        "Max. Drawdown [%]",
        "Avg. Drawdown [%]",
        "Max. Drawdown Duration",
        "STOP_LOSS",
        "NET_PROFIT_TO_MAX_DRAWDOWN",
    ]
]
fig = px.parallel_coordinates(
    df_plot,
    color="NET_PROFIT [$]",
    labels={
        "NET_PROFIT [$]": "NET_PROFIT [$]",
        "Sharpe Ratio": "SHARPE_RATIO",
        "Max. Drawdown [%]": "MAX.DRAWDOWN [%] ",
        "Avg. Drawdown [%]": "AVG.DRAWDOWN [%]",
        "Max. Drawdown Duration": "MAX.DRAWDOWN DURATION [DAYS]",
        "STOP_LOSS": "STOP_LOSS [$]",
        "NET_PROFIT_TO_MAX_DRAWDOWN": "NET_PROFIT_TO_MAX_DRAWDOWN",
    },
    range_color=[df_plot["NET_PROFIT [$]"].min(), df_plot["NET_PROFIT [$]"].max()],
    color_continuous_scale=px.colors.sequential.Viridis,
    title=f"{name}_{filename[:-4]}",
)

fig.write_html(f"outputs/{out_root}/{name}_{filename[:-4]}.htm")
fig.show()



df_lines = df_stats[["NET_PROFIT_TO_MAX_DRAWDOWN", "STOP_LOSS"]]
'''fig2 = px.bar(df_bars, x='STOP_LOSS', y='NET_PROFIT_TO_MAX_DRAWDOWN', color="NET_PROFIT [$]" )
fig2.update_layout(
    title_text=f"{filename[:-4]} stop loss to (net profit to max drawn)",
    barmode="stack",xaxis=dict(
        title='STOP_LOSS [$]',
    )

)'''
print(df_stats)
print(df_lines)
zero_loss_df = df_lines.loc[df_lines["STOP_LOSS"] == 0]
print(zero_loss_df)
y_cord = zero_loss_df.loc[0, 'NET_PROFIT_TO_MAX_DRAWDOWN']
print(y_cord)
fig2 = px.line(df_lines, x='STOP_LOSS', y='NET_PROFIT_TO_MAX_DRAWDOWN', markers=True, title='Зависимость stop loss и отношения NET_PROFIT_TO_MAX_DRAWDOWN')
fig2.update_layout(title='Зависимость stop loss и отношения NET_PROFIT_TO_MAX_DRAWDOWN',
                   xaxis_title='STOP_LOSS [$]'
                   )
fig2.add_hline(y= y_cord, line_dash="dot",
              annotation_text="Without stop-loss",
              annotation_position="top right", line_color="red" )
fig2.write_html(f"outputs/{out_root}/fig2_{filename[:-4]}.htm")
fig2.show()