import matplotlib.pyplot as plt
from backtesting.backtesting import Backtest
from utilits.stop_loss_lazy_strategy import LazyStrategy
import pandas as pd
import os
import numpy as np
from tqdm import tqdm

signals_dir = 'outputs/seed_test_witched_rbm_GC_2019_2022_30min_12_09_2022/'
best_signals = 'outputs/seed_test_witched_rbm_GC_2019_2022_30min_12_09_2022/4_signals_GC_2019_2022_30min_train_window5280forward_window880_lookback_size48.csv'
prod_signals ='outputs/lean_1_11_bt_switched_rbm_GC_2019_2022_30min_07_09_2022/8_signals_GC_2019_2022_30min_train_window8800forward_window440_patch42.csv'
def format_df(df, df_original):
    df = df.set_index(pd.to_datetime(df_original["Datetime"]))
    df = df.resample('D').last()
    df = df.fillna(method='ffill')
    return df


def do_backtest(test_df, verbose=False):
    test_df_ = test_df.copy()
    test_df_["Datetime"] = pd.to_datetime(test_df_["Datetime"])
    test_df_.set_index('Datetime', inplace=True)
    test_df_.sort_index(inplace=True)
    bt = Backtest(test_df_, strategy=LazyStrategy,
                  cash=100000, commission_type="absolute", commission=4.62,
                  features_coeff=10, exclusive_orders=True)
    stats = bt.run()
    if verbose:
        bt.plot()
        print(stats[:27])
    return stats


def many_graphs(folder_path, best_experiment_path, prod_experiment_path):
    df_list = []
    signals_df_list = []
    for i, filename in tqdm(enumerate(os.listdir(folder_path))):
        # Расчет эквити
        can = pd.read_csv(folder_path + filename, index_col=[0])

        stats = do_backtest(can)
        df_list.append(stats._equity_curve["Equity"] - 1e5)
        signals_df_list.append(can)

    # Вычислем среднее эквити и его std
    dataframe = pd.concat(df_list, axis=1)
    mean_results = dataframe.values.mean(axis=1)
    std = dataframe.values.std(axis=1)
    # Вычисление сигнала голосвания
    signals = pd.concat(signals_df_list, axis=0)
    aggregation_dict = {key: "first" for key in signals.columns}
    aggregation_dict["Signal"] = sum
    signals = signals.groupby(signals.index).agg(aggregation_dict)
    signals.sort_index(inplace=True)
    signals["Signal"] = signals["Signal"].apply(lambda x: np.sign(x) if np.sign(x) != 0 else 1)
    # Расчет эквити
    dataframe_agr = do_backtest(signals, verbose=True)._equity_curve["Equity"] - 1e5

    # Загружаем OHCLV тест файл лучшего эксперимента
    best_df = pd.read_csv(best_experiment_path, sep=',')
    # Расчет эквити
    best = do_backtest(best_df, verbose=True)._equity_curve["Equity"] - 1e5

    # Загружаем OHCLV тест файл лучшего эксперимента
    prod_df = pd.read_csv(prod_experiment_path, sep=',')
    # Расчет эквити
    prod = do_backtest(prod_df, verbose=True)._equity_curve["Equity"] - 1e5

    plt.title(
        f"Средние и стандартные отклонения от работы модели на {i + 1} сидах \n Последний бар:"
        f" μ = {round(mean_results[-1], 2)} | σ = {round(std[-1], 2)} | "
        f"σ/μ = {round(std[-1] / mean_results[-1], 2)}",
        fontsize=10)
    plt.xlabel('Дата', fontsize=7)
    plt.xticks(rotation=45, fontsize=5)
    plt.ylabel('Доходность', fontsize=7)
    plt.errorbar(dataframe.index.values, mean_results, std, ecolor='salmon', label="Average equity")
    plt.plot(dataframe.index.values, dataframe_agr.values, label="Vote equity")
    plt.plot(best.index.values, best.values, label="Best equity")
    plt.plot(prod.index.values, prod.values, label="Prod equity")
    plt.legend()
    plt.savefig(f'ApateV3_{i + 1}seeds.pdf')
    plt.show()

many_graphs(signals_dir, best_signals, prod_signals)