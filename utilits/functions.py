from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
from backtesting import Backtest
from utilits.lazy_strategy import LazyStrategy
import backtesting._plotting as plt_backtesting



def get_train_test(train_df, forward_df, patch):
    scaler = MinMaxScaler()
    train_arr = train_df[['Open','High','Low','Close','Volume']].to_numpy()
    forward_arr = forward_df[['Open','High','Low','Close','Volume']].to_numpy()
    train_samples = [train_arr[i-patch:i] for i in range(len(train_arr)+1) if i - patch >= 0]
    forward_samples = [forward_arr[i-patch:i] for i in range(len(forward_arr)+1) if i - patch >= 0]
    dates_arr = forward_df['Datetime'].values
    #dates_arr = forward_df.index.values
    dates_arr_samp = [dates_arr[i-patch:i] for i in range(len(dates_arr)+1) if i - patch >= 0]
    # Подготавливаем Обучающие данные
    trainX = []
    for arr in train_samples:
        arr_normlzd = scaler.fit_transform(arr)
        trainX.append(arr_normlzd.flatten())

    # Подготавливаем форвардные данные и Сигналы
    signal_dates = [i[-1] for i in dates_arr_samp]
    signal_open = []
    signal_high = []
    signal_low = []
    signal_close = []
    signal_volume = []
    forwardX=[]
    for arr in forward_samples:
        signal_open.append(float(arr[-1, [0]]))
        signal_high.append(float(arr[-1, [1]]))
        signal_low.append(float(arr[-1, [2]]))
        signal_close.append(float(arr[-1, [3]]))
        signal_volume.append(float(arr[-1, [4]]))
        arr_normlzd = scaler.fit_transform(arr)
        forwardX.append(arr_normlzd.flatten())

    Signals = pd.DataFrame(
        {
            "Datetime": signal_dates,
            "Open": signal_open,
            "High": signal_high,
            "Low": signal_low,
            "Close": signal_close,
            "Volume": signal_volume,
        }
    )

    return np.array(trainX), np.array(forwardX), Signals

def std_get_train_test(train_df, forward_df, patch):
    scaler = StandardScaler()
    train_arr = train_df[['Open','High','Low','Close','Volume']].to_numpy()
    forward_arr = forward_df[['Open','High','Low','Close','Volume']].to_numpy()
    train_samples = [train_arr[i-patch:i] for i in range(len(train_arr)+1) if i - patch >= 0]
    forward_samples = [forward_arr[i-patch:i] for i in range(len(forward_arr)+1) if i - patch >= 0]
    dates_arr = forward_df['Datetime'].values


    dates_arr_samp = [dates_arr[i-patch:i] for i in range(len(dates_arr)+1) if i - patch >= 0]
    # Подготавливаем Обучающие данные
    trainX = []
    for arr in train_samples:
        arr_normlzd = scaler.fit_transform(arr)
        trainX.append(arr_normlzd.flatten())

    # Подготавливаем форвардные данные и Сигналы
    signal_dates = [i[-1] for i in dates_arr_samp]
    signal_open = []
    signal_high = []
    signal_low = []
    signal_close = []
    signal_volume = []
    forwardX=[]
    for arr in forward_samples:
        signal_open.append(float(arr[-1, [0]]))
        signal_high.append(float(arr[-1, [1]]))
        signal_low.append(float(arr[-1, [2]]))
        signal_close.append(float(arr[-1, [3]]))
        signal_volume.append(float(arr[-1, [4]]))
        arr_normlzd = scaler.fit_transform(arr)
        forwardX.append(arr_normlzd.flatten())

    Signals = pd.DataFrame(
        {
            "Datetime": signal_dates,
            "Open": signal_open,
            "High": signal_high,
            "Low": signal_low,
            "Close": signal_close,
            "Volume": signal_volume,
        }
    )

    return np.array(trainX), np.array(forwardX), Signals
def get_stat_after_forward(
    result_df,
    lookback_size,
    n_hiddens,
    train_window,
    forward_window,
    source_file_name,
    out_root,
    out_data_root,
    trial_namber,
    get_trade_info=False,
):
    plt_backtesting._MAX_CANDLES = 200_000
    pd.pandas.set_option("display.max_columns", None)
    pd.set_option("expand_frame_repr", False)
    pd.options.display.expand_frame_repr = False
    pd.set_option("display.precision", 2)

    result_df.set_index("Datetime", inplace=True)
    result_df.index = pd.to_datetime(result_df.index)
    result_df = result_df.sort_index()



    """ Параметры тестирования """
    i = 0
    deposit = 100000  # сумма одного контракта GC & CL
    comm = 4.6  # GC - комиссия для золота


    """ Тестирвоание """




    df_stats = pd.DataFrame()


    bt = Backtest(
        result_df,
        strategy=LazyStrategy,
        cash=deposit,
        commission_type="absolute",
        commission=4.62,
        features_coeff=10,
        exclusive_orders=True,
    )
    stats = bt.run()[:27]



    df_stats = df_stats.append(stats, ignore_index=True)
    df_stats["Net Profit [$]"] = (
        df_stats.loc[i, "Equity Final [$]"]
        - deposit
        - df_stats.loc[i, "# Trades"] * comm
    )
    # df_stats.loc[i, "buy_before"] = buy_before * step
    # df_stats.loc[i, "sell_after"] = sell_after * step
    df_stats["train_window"] = train_window
    df_stats["forward_window"] = forward_window
    df_stats["lookback_size"] = lookback_size
    df_stats["n_hidden"] = n_hiddens



    if get_trade_info == True and df_stats["Net Profit [$]"].values > 0:
        bt.plot(
            plot_volume=True,
            relative_equity=False,
            filename=f"{out_root}/{out_data_root}/{trial_namber}_bt_plot_{source_file_name[:-4]}train_window{train_window}forward_window{forward_window}_lookback_size{lookback_size}.html",
        )
        stats.to_csv(
            f"{out_root}/{out_data_root}/{trial_namber}_stats_{source_file_name[:-4]}_train_window{train_window}forward_window{forward_window}_lookback_size{lookback_size}.txt"
        )
        result_df["Signal"] = result_df["Signal"].astype(int)

        result_df.insert(0, "Datetime", result_df.index)
        result_df = result_df.reset_index(drop=True)
        result_df[
            ["Datetime", "Open", "High", "Low", "Close", "Volume", "Signal"]
        ].to_csv(
            f"{out_root}/{out_data_root}/{trial_namber}_signals_{source_file_name[:-4]}_train_window{train_window}forward_window{forward_window}_lookback_size{lookback_size}.csv"
        )

    return df_stats


def dbscan_predict(model, X):

    nr_samples = X.shape[0]

    y_new = np.ones(shape=nr_samples, dtype=int) * -1

    for i in range(nr_samples):
        diff = model.components_ - X[i, :]  # NumPy broadcasting

        dist = np.linalg.norm(diff, axis=1)  # Euclidean distance

        shortest_dist_idx = np.argmin(dist)

        if dist[shortest_dist_idx] < model.eps:
            y_new[i] = model.labels_[model.core_sample_indices_[shortest_dist_idx]]

    return y_new


def train_backtest(train_df, labels, patch, train_backtest_window):
    dates_arr = train_df['Datetime'].values
    #dates_arr = train_df.index.values

    dates_arr_samp = [dates_arr[i - patch:i] for i in range(len(dates_arr) + 1) if i - patch >= 0]
    train_arr = train_df[['Open', 'High', 'Low', 'Close', 'Volume']].to_numpy()
    train_samples = [train_arr[i - patch:i] for i in range(len(train_arr) + 1) if i - patch >= 0]
    signal_dates = [i[-1] for i in dates_arr_samp]
    signal_open = []
    signal_high = []
    signal_low = []
    signal_close = []
    signal_volume = []

    for arr in train_samples:
        signal_open.append(float(arr[-1, [0]]))
        signal_high.append(float(arr[-1, [1]]))
        signal_low.append(float(arr[-1, [2]]))
        signal_close.append(float(arr[-1, [3]]))
        signal_volume.append(float(arr[-1, [4]]))


    result_df = pd.DataFrame(
        {
            "Datetime": signal_dates,
            "Open": signal_open,
            "High": signal_high,
            "Low": signal_low,
            "Close": signal_close,
            "Volume": signal_volume,
        }
    )

    result_df['Signal'] = labels
    result_df.loc[result_df["Signal"] == 0, "Signal"] = -1
    result_df = result_df[-train_backtest_window:]

    plt_backtesting._MAX_CANDLES = 200_000
    pd.pandas.set_option("display.max_columns", None)
    pd.set_option("expand_frame_repr", False)
    pd.options.display.expand_frame_repr = False
    pd.set_option("display.precision", 2)

    result_df.set_index("Datetime", inplace=True)
    result_df.index = pd.to_datetime(result_df.index)
    result_df = result_df.sort_index()


    """ Параметры тестирования """
    i = 0
    deposit = 100000  # сумма одного контракта GC & CL
    comm = 4.6  # GC - комиссия для золота

    """ Тестирвоание """

    df_stats = pd.DataFrame()

    bt = Backtest(
        result_df,
        strategy=LazyStrategy,
        cash=deposit,
        commission_type="absolute",
        commission=4.62,
        features_coeff=10,
        exclusive_orders=True,
    )
    stats = bt.run()[:27]

    df_stats = df_stats.append(stats, ignore_index=True)
    Original_Net_Profit = (
            df_stats.loc[i, "Equity Final [$]"]
            - deposit
            - df_stats.loc[i, "# Trades"] * comm
    )
    result_df.loc[result_df["Signal"] == -1, "Signal"] = -111
    result_df.loc[result_df["Signal"] == 1, "Signal"] = -1
    result_df.loc[result_df["Signal"] == -111, "Signal"] = 1

    #print('switcged',result_df)
    """ Параметры тестирования """
    i = 0
    deposit = 100000  # сумма одного контракта GC & CL
    comm = 4.6  # GC - комиссия для золота

    """ Тестирвоание """

    df_stats = pd.DataFrame()

    bt = Backtest(
        result_df,
        strategy=LazyStrategy,
        cash=deposit,
        commission_type="absolute",
        commission=4.62,
        features_coeff=10,
        exclusive_orders=True,
    )
    stats = bt.run()[:27]

    df_stats = df_stats.append(stats, ignore_index=True)
    Switched_Net_Profit = (
            df_stats.loc[i, "Equity Final [$]"]
            - deposit
            - df_stats.loc[i, "# Trades"] * comm
    )

    if Switched_Net_Profit > Original_Net_Profit:
        switch_signals = 1
    else:
        switch_signals = 0
    return switch_signals


def get_train_test_binary(train_df, forward_df, patch):
    binary_train = train_df.iloc[:, 1:].diff()
    binary_train[binary_train <= 0] = 0
    binary_train[binary_train > 0] = 1
    binary_train = binary_train[1:].to_numpy()
    binary_forward = forward_df.iloc[:, 1:].diff()
    binary_forward[binary_forward <= 0] = 0
    binary_forward[binary_forward > 0] = 1
    binary_forward = binary_forward[1:].to_numpy()
    train_samples = [binary_train[i - patch:i] for i in range(len(binary_train) + 1) if i - patch >= 0]
    forward_samples = [binary_forward[i - patch:i] for i in range(len(binary_forward) + 1) if i - patch >= 0]
    trainX = []
    for arr in train_samples:
        trainX.append(arr.flatten())
    forwardX = []
    for arr in forward_samples:
        forwardX.append(arr.flatten())

    Signals = forward_df[patch:]

    return np.array(trainX), np.array(forwardX), Signals


def binary_train_backtest(train_df, labels, patch, train_backtest_window):
    train_df = train_df[1:]
    dates_arr = train_df['Datetime'].values
    dates_arr_samp = [dates_arr[i - patch:i] for i in range(len(dates_arr) + 1) if i - patch >= 0]
    train_arr = train_df[['Open', 'High', 'Low', 'Close', 'Volume']].to_numpy()
    train_samples = [train_arr[i - patch:i] for i in range(len(train_arr) + 1) if i - patch >= 0]
    signal_dates = [i[-1] for i in dates_arr_samp]
    signal_open = []
    signal_high = []
    signal_low = []
    signal_close = []
    signal_volume = []

    for arr in train_samples:
        signal_open.append(float(arr[-1, [0]]))
        signal_high.append(float(arr[-1, [1]]))
        signal_low.append(float(arr[-1, [2]]))
        signal_close.append(float(arr[-1, [3]]))
        signal_volume.append(float(arr[-1, [4]]))


    result_df = pd.DataFrame(
        {
            "Datetime": signal_dates,
            "Open": signal_open,
            "High": signal_high,
            "Low": signal_low,
            "Close": signal_close,
            "Volume": signal_volume,
        }
    )
    result_df['Signal'] = labels
    result_df.loc[result_df["Signal"] == 0, "Signal"] = -1
    result_df = result_df[-train_backtest_window:]

    plt_backtesting._MAX_CANDLES = 200_000
    pd.pandas.set_option("display.max_columns", None)
    pd.set_option("expand_frame_repr", False)
    pd.options.display.expand_frame_repr = False
    pd.set_option("display.precision", 2)

    result_df.set_index("Datetime", inplace=True)
    result_df.index = pd.to_datetime(result_df.index)
    result_df = result_df.sort_index()


    """ Параметры тестирования """
    i = 0
    deposit = 100000  # сумма одного контракта GC & CL
    comm = 4.6  # GC - комиссия для золота

    """ Тестирвоание """

    df_stats = pd.DataFrame()

    bt = Backtest(
        result_df,
        strategy=LazyStrategy,
        cash=deposit,
        commission_type="absolute",
        commission=4.62,
        features_coeff=10,
        exclusive_orders=True,
    )
    stats = bt.run()[:27]

    df_stats = df_stats.append(stats, ignore_index=True)
    Original_Net_Profit = (
            df_stats.loc[i, "Equity Final [$]"]
            - deposit
            - df_stats.loc[i, "# Trades"] * comm
    )
    result_df.loc[result_df["Signal"] == -1, "Signal"] = -111
    result_df.loc[result_df["Signal"] == 1, "Signal"] = -1
    result_df.loc[result_df["Signal"] == -111, "Signal"] = 1

    #print('switcged',result_df)
    """ Параметры тестирования """
    i = 0
    deposit = 100000  # сумма одного контракта GC & CL
    comm = 4.6  # GC - комиссия для золота

    """ Тестирвоание """

    df_stats = pd.DataFrame()

    bt = Backtest(
        result_df,
        strategy=LazyStrategy,
        cash=deposit,
        commission_type="absolute",
        commission=4.62,
        features_coeff=10,
        exclusive_orders=True,
    )
    stats = bt.run()[:27]

    df_stats = df_stats.append(stats, ignore_index=True)
    Switched_Net_Profit = (
            df_stats.loc[i, "Equity Final [$]"]
            - deposit
            - df_stats.loc[i, "# Trades"] * comm
    )

    if Switched_Net_Profit > Original_Net_Profit:
        switch_signals = 1
    else:
        switch_signals = 0
    return switch_signals



'''def labeled_get_train_test(train_df, forward_df, patch):
    train_df["Signal"] = train_df["Signal"].astype(
        int
    )
    forward_df["Signal"] = forward_df["Signal"].astype(
        int
    )
    train_df.loc[train_df["Signal"] == -1, "Signal"] = 0
    forward_df["Signal"] = 0.5



    scaler = MinMaxScaler()
    train_arr = train_df[['Open','High','Low','Close','Volume']].to_numpy()
    forward_arr = forward_df[['Open','High','Low','Close','Volume']].to_numpy()

    train_samples = [train_arr[i-patch:i] for i in range(len(train_arr)+1) if i - patch >= 0]
    train_sample_labels = np.array(
        [train_df['Signal'].values[i - patch:i] for i in range(len(train_arr) + 1) if i - patch >= 0])
    forward_samples = [forward_arr[i-patch:i] for i in range(len(forward_arr)+1) if i - patch >= 0]
    forward_sample_labels = np.array(
        [forward_df['Signal'].values[i - patch:i] for i in range(len(forward_df) + 1) if i - patch >= 0])

    dates_arr = forward_df['Datetime'].values
    dates_arr_samp = [dates_arr[i-patch:i] for i in range(len(dates_arr)+1) if i - patch >= 0]
    # Подготавливаем Обучающие данные

    trainX = []
    for arr, sig in zip(train_samples, train_sample_labels):
        arr_normlzd = scaler.fit_transform(arr)
        arr_comb = np.insert(arr_normlzd, 5, sig, axis=1)


        trainX.append(arr_comb.flatten())


    # Подготавливаем форвардные данные и Сигналы
    signal_dates = [i[-1] for i in dates_arr_samp]
    signal_open = []
    signal_high = []
    signal_low = []
    signal_close = []
    signal_volume = []
    forwardX=[]

    for arr, sig in zip (forward_samples, forward_sample_labels):
        signal_open.append(float(arr[-1, [0]]))
        signal_high.append(float(arr[-1, [1]]))
        signal_low.append(float(arr[-1, [2]]))
        signal_close.append(float(arr[-1, [3]]))
        signal_volume.append(float(arr[-1, [4]]))
        arr_normlzd = scaler.fit_transform(arr)
        arr_comb = np.insert(arr_normlzd, 5, sig, axis=1)

        forwardX.append(arr_comb.flatten())



    Signals = pd.DataFrame(
        {
            "Datetime": signal_dates,
            "Open": signal_open,
            "High": signal_high,
            "Low": signal_low,
            "Close": signal_close,
            "Volume": signal_volume,
        }
    )

    return np.array(trainX), np.array(forwardX), Signals'''




