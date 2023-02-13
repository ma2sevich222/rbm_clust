
import numpy as np
import pandas as pd
import os
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from backtesting.backtesting import Backtest
from utilits.lazy_strategy import LazyStrategy
from backtesting import _plotting as plt_backtesting
plt_backtesting._MAX_CANDLES = 1_000_000


CASH = 100000
COMMISION = 4.62
STOP_LOSS = -9.3
TIMEFRAME = 'GC_30min'
time_frame = 30
#MODELTYPE = 'best_sharpe'
data_df = pd.read_csv('outputs/testV31_rbm_GC_2019_2022_30min_06_10_2022/0_signals_GC_2019_2022_30min_train_window5280forward_window5_patch33.csv')
signal_df = data_df.copy()
data_df.index = data_df['Datetime']
data_df.index = pd.to_datetime(data_df.index)
bt = Backtest(data_df, strategy=LazyStrategy, cash=CASH, commission_type="absolute", commission=COMMISION,
              features_coeff=10, exclusive_orders=True)
stats = bt.run(stop_loss=STOP_LOSS, take_profit=50.8, clearing=True, time_frame=time_frame)
# bt.plot(relative_equity=False)
trades = stats._trades
# print(stats['Equity Final [$]'], stats['# Trades'])
# trades.to_csv(f'../best_results/Janus_V2_{TIMEFRAME}_clearing_retraining_new_dataset_{MODELTYPE}/stop_loss/trades.csv')

signal_df['Datetime'] = pd.to_datetime(signal_df['Datetime'])
signal_df['Signal'] = 0
for i in range(len(trades)):
    start_idx = signal_df[signal_df['Datetime'] == trades['EntryTime'][i]].index[0] - 1
    fin_idx = signal_df[signal_df['Datetime'] == trades['ExitTime'][i]].index[0] - 1
    signal_df.iloc[start_idx: fin_idx, -1] = trades.loc[i, 'Size']

signal_df.index = signal_df['Datetime']
signal_df.to_csv(' 0_ApateV31_signals_best_stop_loss.csv', index=False)
