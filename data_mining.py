import pandas as pd
import plotly.express as px
import os
import optuna
from datetime import date
import random
from torch.utils.data import  DataLoader
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from utilits.functions import get_train_test, get_stat_after_forward, train_backtest
from utilits.classes_and_models import RBM, RBMDataset
from sklearn.cluster import KMeans

def get_train_test(train_df, forward_df, patch):
    scaler = MinMaxScaler()
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

torch.cuda.set_device(1)
os.environ["PYTHONHASHSEED"] = str(2020)
random.seed(2020)
np.random.seed(2020)
torch.manual_seed(2020)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


today = date.today()
source = "source_root"
out_root = "outputs"
source_file_name = "GC_2020_2022_30min.csv"
start_forward_time = "2021-01-04 00:00:00"
end_test_time = "2021-07-05 00:00:00"
date_xprmnt = today.strftime("%d_%m_%Y")
out_data_root = f"switched_rbm2_{source_file_name[:-4]}_{date_xprmnt}"
os.mkdir(f"{out_root}/{out_data_root}")
intermedia = pd.DataFrame()
intermedia.to_excel(
    f"{out_root}/{out_data_root}/intermedia_{source_file_name[:-4]}.xlsx"
)


start_forward_time = "2021-01-04 00:00:00"
df = pd.read_csv(f"{source}/{source_file_name}")
forward_index = df[df["Datetime"] == start_forward_time].index[0]
end_test_index = df[df["Datetime"] == end_test_time].index[0]
df = df[:end_test_index]

"""""" """""" """""" """""" """"" Параметры для оптимизации   """ """ """ """ """ """ """ """ """ ""

patch = trial.suggest_int("patch", 2, 60, step=2)
HIDDEN_UNITS = trial.suggest_int("hidden_units", 5, 100, step=5)
train_window = trial.suggest_categorical("train_window", [2640, 5280, 10560])
train_backtest_window = trial.suggest_categorical("train_backtest_window", [880, 2640])
forward_window = trial.suggest_categorical(
        "forward_window", [88, 220, 440, 880]
    )

"""""" """""" """""" """""" """"" Параметры сети """ """""" """""" """""" """"""
BATCH_SIZE = 10
VISIBLE_UNITS = 5 * patch
CD_K = 2
EPOCHS = 100





df_for_split = df[(forward_index - train_window) :]
df_for_split = df_for_split.reset_index(drop=True)
n_iters = (len(df_for_split) - int(train_window)) // int(forward_window)

signals = []
for n in range(n_iters):

        train_df = df_for_split[:train_window]

        if n == n_iters - 1:
            forward_df = df_for_split[train_window:]
        else:
            forward_df = df_for_split[
                int(train_window) : sum([int(train_window), int(forward_window)])
            ]
        df_for_split = df_for_split[int(forward_window) :]
        df_for_split = df_for_split.reset_index(drop=True)

        Train_X, Forward_X, Signals = get_train_test(
            train_df, forward_df, patch
        )
        train_dataset = RBMDataset(Train_X)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
        rbm = RBM(VISIBLE_UNITS, HIDDEN_UNITS, CD_K, use_cuda=True)

        ''' Обучаем модель '''
        torch.cuda.empty_cache()
        for epoch in range(EPOCHS):
            epoch_error = 0.0

            for batch in train_dataloader:

                # batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data


                batch = batch.cuda()

                batch_error = rbm.contrastive_divergence(batch)

                epoch_error += batch_error

        feature_set = np.zeros((len(Train_X), HIDDEN_UNITS))

        for i, batch in enumerate(train_dataloader):
            # batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data


            batch = batch.cuda()

            feature_set[i * BATCH_SIZE:i * BATCH_SIZE + len(batch)] = rbm.sample_hidden(batch).cpu().numpy()

        '''pca = PCA(n_components=2)
        compressed_feature_set = pca.fit_transform(feature_set)
        df_pca = pd.DataFrame()
        df_pca['param_1'] = compressed_feature_set[:, [0]].reshape(-1)
        df_pca['param_2'] = compressed_feature_set[:, [1]].reshape(-1)'''
        kmeans = KMeans(n_clusters=2, random_state=0)
        features_labels = kmeans.fit_predict(feature_set)
        '''df_pca["Label"] = features_labels
        fig = px.scatter(df_pca, x="param_1", y="param_2", color="Label")
        fig.show()'''
        switch_signals = train_backtest(train_df, features_labels, patch, train_backtest_window )

        predictions = []
        for forward_array in Forward_X:
            forward_array = torch.tensor(forward_array, dtype=torch.float32).cuda()
            trnsfrmd_forward_array = rbm.sample_hidden(forward_array).cpu().numpy()
            pred = kmeans.predict(trnsfrmd_forward_array.reshape(1, -1).astype(float))

            if switch_signals == 1:

                if int(pred) == 0:
                    predictions.append(1)

                else:
                    predictions.append(-1)

            else:
                if int(pred) == 0:
                    predictions.append(-1)

                else:

                    predictions.append(int(pred))


        Signals["Signal"] = predictions

        signals.append(Signals)

signals_combained = pd.concat(signals, ignore_index=True, sort=False)




df_stata = get_stat_after_forward(
        signals_combained,
        patch,
        HIDDEN_UNITS,
        train_window,
        forward_window,
        source_file_name,
        out_root,
        out_data_root,
        trial.number,
        get_trade_info=True,
    )
net_profit = df_stata["Net Profit [$]"].values[0]
Sharpe_Ratio = df_stata["Sharpe Ratio"].values[0]
trades = df_stata["# Trades"].values[0]