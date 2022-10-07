import os
import pandas as pd
import re






path = "outputs/lean_sel_parV31_rbm_GC_2019_2022_30min_04_10_2022"
file_names = os.listdir(path)
all_stata =pd.DataFrame()
for filename in file_names:
  if 'stats' in filename:
      _file = pd.read_csv(f"{path}/{filename}")
      stats_dict = {}
      _key = _file.iloc[:, 0].values
      _value = _file.iloc[:, 1].values
      i = 0
      for k, v in zip(_key, _value):
          stats_dict.update({k: v})
      с_file = pd.DataFrame(stats_dict, index=[0])
      i+=1
      с_file['stat_trial_n'] = int(re.findall(r'\d+', filename[:4])[0])
      с_file['eq/dd'] = abs(с_file['Equity Final [$]'].astype(float) / с_file['Max. Drawdown [%]'].astype(float))
      all_stata = all_stata.append(с_file, ignore_index=True)
  if 'intermedia' in  filename and  filename.endswith('.xlsx'):
      trial_parameters = pd.read_excel(f"{path}/{filename}")


del trial_parameters['# Trades']
trial_parameters= trial_parameters.sort_values(by=['trial'])
all_stata = all_stata.sort_values(by=['stat_trial_n'])
all_stata=all_stata.reset_index(drop=True)
final_df = pd.concat([trial_parameters, all_stata], axis=1)
final_df.to_excel(f"{path}/all_statisticss.xlsx")



