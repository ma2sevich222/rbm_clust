

import pandas as pd
import plotly.express as px
import os
import optuna
from datetime import date
import random
from torch.utils.data import  DataLoader
import numpy as np
import torch
from sklearn.decomposition import PCA
from utilits.functions import get_train_test, get_stat_after_forward
from utilits.classes_and_models import RBM, RBMDataset
from sklearn.cluster import AgglomerativeClustering

#torch.cuda.set_device(1)
os.environ["PYTHONHASHSEED"] = str(2020)
random.seed(2020)
np.random.seed(2020)
torch.manual_seed(2020)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


today = date.today()
source = "source_root"
out_root = "outputs"
source_file_name = "GC_2020_2022_15min.csv"
start_forward_time = "2021-01-04 00:00:00"
date_xprmnt = today.strftime("%d_%m_%Y")
out_data_root = f"aglomer_rbm_{source_file_name[:-4]}_{date_xprmnt}"
os.mkdir(f"{out_root}/{out_data_root}")
intermedia = pd.DataFrame()
intermedia.to_excel(
    f"{out_root}/{out_data_root}/intermedia_{source_file_name[:-4]}.xlsx"
)

n_trials = 300

###################################################################################################

def objective(trial):
    start_forward_time = "2021-01-04 00:00:00"
    df = pd.read_csv(f"{source}/{source_file_name}")
    forward_index = df[df["Datetime"] == start_forward_time].index[0]

    """""" """""" """""" """""" """"" Параметры для оптимизации   """ """ """ """ """ """ """ """ """ ""

    patch = trial.suggest_int("patch", 30, 34, step=2)
    HIDDEN_UNITS = trial.suggest_int("hidden_units", 10, 100, step=5)
    n_neighbors = trial.suggest_int("n_neighbors", 50, 700, step=50)
    train_window = trial.suggest_categorical("train_window", [8800, 10560, 17600])
    forward_window = trial.suggest_categorical(
        "forward_window", [1760, 3520, 5280]
    )

    """""" """""" """""" """""" """"" Параметры сети """ """""" """""" """""" """"""
    BATCH_SIZE = 10
    VISIBLE_UNITS = 5 * patch
    CD_K = 2
    EPOCHS = 30
    #CUDA = torch.cuda.is_available()
    '''CUDA_DEVICE = 0

    if CUDA:
        torch.cuda.set_device(CUDA_DEVICE)'''




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
        predictions = []

        for forward_array in Forward_X:
            Train_X = np.append(Train_X, forward_array.reshape(1, -1), axis=0)
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
            aglomer_cl = AgglomerativeClustering(n_clusters=2).fit(feature_set)
            #features_labels = kmeans.fit_predict(feature_set)
            '''df_pca["Label"] = features_labels
            fig = px.scatter(df_pca, x="param_1", y="param_2", color="Label")
            fig.show()'''
            '''uniq_labeles = np.unique(spec_cl.labels_)

            if len(uniq_labeles) < 2 or len(uniq_labeles) > 3:
                net_profit, Sharpe_Ratio = -222000, 0
                return net_profit, Sharpe_Ratio'''



            pred = aglomer_cl.labels_[-1]


            predictions.append(int(pred))

        Signals["Signal"] = predictions
        signals.append(Signals)

    signals_combained = pd.concat(signals, ignore_index=True, sort=False)
    signals_combained.loc[signals_combained["Signal"] == 0, "Signal"] = -1
    '''signals_combained.loc[signals_combained["Signal"] == 0, "Signal"] = 66
    signals_combained.loc[signals_combained["Signal"] == 1, "Signal"] = -1
    signals_combained.loc[signals_combained["Signal"] == 66, "Signal"] = 1'''


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
    if trades <= 300:
        net_profit = -222000
        Sharpe_Ratio = 0


    trial.set_user_attr("# Trades", trades)
    parameters = trial.params
    parameters.update({"trial": trial.number})
    parameters.update({"values_0": net_profit})
    parameters.update({"values_1": Sharpe_Ratio})
    parameters.update({"# Trades": trades})
    inter = pd.read_excel(
        f"{out_root}/{out_data_root}/intermedia_{source_file_name[:-4]}.xlsx"
    )
    inter = inter.append(parameters, ignore_index=True)
    inter.to_excel(
        f"{out_root}/{out_data_root}/intermedia_{source_file_name[:-4]}.xlsx",
        index=False,
    )

    #torch.save(rbm.state_dict(), f"{out_root}/{out_data_root}/weights.pt")

    return net_profit, Sharpe_Ratio


sampler = optuna.samplers.TPESampler(seed=2020)
study = optuna.create_study(directions=["maximize", "maximize"], sampler=sampler)
study.optimize(objective, n_trials=n_trials)

tune_results = study.trials_dataframe()

tune_results["params_forward_window"] = tune_results["params_forward_window"].astype(
    int
)
tune_results["params_train_window"] = tune_results["params_train_window"].astype(int)
df_plot = tune_results[
    [
        "values_0",
        "values_1",
        "user_attrs_# Trades",
        "params_patch",
        "params_hidden_units",
        "params_train_window",
        "params_forward_window",
    ]
]

fig = px.parallel_coordinates(
    df_plot,
    color="values_0",
    labels={
        "values_0": "Net Profit ($)",
        "values_1": "Sharpe_Ratio",
        "user_attrs_# Trades": "Trades",
        "params_patch": "patch(bars)",
        "params_hidden_units": "n_hidden_units",
        "params_train_window": "train_window (bars)",
        "params_forward_window": "forward_window (bars)",
    },
    range_color=[df_plot["values_0"].min(), df_plot["values_0"].max()],
    color_continuous_scale=px.colors.sequential.Viridis,
    title=f"bayes_parameters_select_{source_file_name[:-4]}_optune_epoch_{n_trials}",
)

fig.write_html(f"{out_root}/{out_data_root}/hyp_par_sel_{source_file_name[:-4]}.htm")
fig.show()
tune_results.to_excel(
    f"{out_root}/{out_data_root}/hyp_par_sel_{source_file_name[:-4]}.xlsx"
)












