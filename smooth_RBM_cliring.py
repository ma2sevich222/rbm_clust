######################################################
# Copyright © 2021-2099 Ekosphere. All rights reserved
# Author: Evgeny Matusevich
# Contacts: <ma2sevich222@gmail.com>
# File: with_cliring_RBM.py
#######################################################

import os
import random
from datetime import date
import numpy as np
import optuna
import pandas as pd
import plotly.express as px
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from forward.forward import ForwardAnalysis
import pandas as pd
from typing import Union
from os.path import exists
from utilits.classes_and_models import RBM, RBMDataset, Kalman_Smoother
from utilits.functions import sm_std_get_train_test, get_stat_after_forward, train_backtest

if not os.path.isdir("outputs"):
    os.makedirs("outputs")

# torch.cuda.set_device(1)
os.environ["PYTHONHASHSEED"] = str(2020)
random.seed(2020)
np.random.seed(2020)
torch.manual_seed(2020)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


today = date.today()
source = "source_root"
out_root = "outputs"
source_file_name = "GC_2019_2022_30min.csv"
time_f = 30
#start_forward_time = "2021-11-01 22:45:00"  # время начало форварда
# end_forward_time = "2022-05-06 09:30:00"  # конец фоврарда
date_xprmnt = today.strftime("%d_%m_%Y")
out_data_root = f"only_smooth_rbmV3.1_{source_file_name[:-4]}_{date_xprmnt}"
os.mkdir(f"{out_root}/{out_data_root}")
'''intermedia = pd.DataFrame()
intermedia.to_excel(
    f"{out_root}/{out_data_root}/intermedia_{source_file_name[:-4]}.xlsx)"'''


df = pd.read_csv(f"{source}/{source_file_name}")
smoother = Kalman_Smoother(df)
smooth_df = smoother.get_smoothed_df()
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)
"""""" """""" """""" """""" """"" Параметры данных   """ """ """ """ """ """ """ """ """ ""

patch = 33
HIDDEN_UNITS = 55
train_window = 5280
train_backtest_window = 880
forward_window = 5
random_s = 666
"""""" """""" """""" """""" """"" Параметры сети """ """""" """""" """""" """"""
BATCH_SIZE = 10
VISIBLE_UNITS = 5 * patch
CD_K = 2  # количество циклов
EPOCHS = 100

forward = ForwardAnalysis(
    df,
    timeframe=time_f,
    train_window=train_window,
    test_window=forward_window,
    start_test_point="2021-11-01T22:45:00",
)
train_df_list = []
test_df_list = []
signals = []
smoothed_train =[]
smoothed_test = []
dates_dict = {'train_start':[], 'train_end':[]}
for train_w, test_w in forward.run():
    tr_df = df[train_w[0]:train_w[1]]
    tr_df =tr_df.reset_index()
    sm_tr_df = smooth_df[train_w[0]:train_w[1]]
    sm_tr_df = sm_tr_df.reset_index(drop=True)
    tst_df = df[test_w[0] - (patch - 1):test_w[1]]
    tst_df = tst_df.reset_index()
    sm_tt_df = smooth_df[test_w[0] - (patch - 1):test_w[1]]
    sm_tt_df = sm_tt_df.reset_index(drop=True)
    train_df_list.append(tr_df)
    test_df_list.append(tst_df)
    smoothed_train.append(sm_tr_df)
    smoothed_test.append(sm_tt_df)


for train_df, forward_df, sm_train_df, sm_forward_df in zip(train_df_list, test_df_list, smoothed_train, smoothed_test ):
    dates_dict['train_start'].append(str(train_df['Datetime'].values[0]))
    dates_dict['train_end'].append(str(train_df['Datetime'].values[-1]))
    Train_X, Forward_X, Signals = sm_std_get_train_test(forward_df, sm_train_df, sm_forward_df, patch)
    train_dataset = RBMDataset(Train_X)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    torch.manual_seed(random_s)
    rbm = RBM(VISIBLE_UNITS, HIDDEN_UNITS, CD_K, use_cuda=True)

    """ """ " " """ Обучаем модель """ " " """ """

    torch.cuda.empty_cache()
    for epoch in range(EPOCHS):
        epoch_error = 0.0

        for batch in train_dataloader:
            batch = batch.cuda()

            batch_error = rbm.contrastive_divergence(batch)

            epoch_error += batch_error

    feature_set = np.zeros(
        (len(Train_X), HIDDEN_UNITS)
    )  # инициализируем векторы скрытого пространсва

    for i, batch in enumerate(train_dataloader):
        batch = batch.cuda()
        feature_set[i * BATCH_SIZE: i * BATCH_SIZE + len(batch)] = (
            rbm.sample_hidden(batch).cpu().numpy()
        )  # получаем значения скрытого пространсва

    """pca = PCA(n_components=2)
                compressed_feature_set = pca.fit_transform(feature_set) # переводим в 2-ое пространство для отрисовки
                df_pca = pd.DataFrame()
                df_pca['param_1'] = compressed_feature_set[:, [0]].reshape(-1)
                df_pca['param_2'] = compressed_feature_set[:, [1]].reshape(-1)"""
    kmeans = KMeans(n_clusters=2, random_state=0)  # задаем кластеризатор
    features_labels = kmeans.fit_predict(
        feature_set
    )  # обучаем, делаем предикт если хотим отрисовать кластеры
    """df_pca["Label"] = features_labels
        fig = px.scatter(df_pca, x="param_1", y="param_2", color="Label")
        fig.show()"""
    switch_signals = train_backtest(
        train_df, features_labels, patch, train_backtest_window, time_f
    ) # делаем бэктест на трэйне, 0 - не меняем сигналы, 1- меняем

    """ """ " " """ Делаем форвардный анализ """ " " """ """

    predictions = []
    for forward_array in Forward_X:
        forward_array = torch.tensor(forward_array, dtype=torch.float32).cuda()
        trnsfrmd_forward_array = (
            rbm.sample_hidden(forward_array).cpu().numpy()
        )  # получаем скрытые векторы
        pred = kmeans.predict(
            trnsfrmd_forward_array.reshape(1, -1).astype(float)
        )  # предсказываем кластер

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

signals_combained = pd.concat(signals, sort=False)
df_stata = get_stat_after_forward(
    signals_combained,
    patch,
    HIDDEN_UNITS,
    train_window,
    forward_window,
    source_file_name,
    out_root,
    out_data_root,
    1, time_f,
    get_trade_info=True,
)
date_df = pd.DataFrame(dates_dict)
date_df.to_csv(f"{out_root}/{out_data_root}/Apate_train_forward_dates.csv")
