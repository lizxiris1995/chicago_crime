import numpy as np
import pandas as pd
import itertools
import logging
import toml
import warnings
from libs.CrimePredictionApp import CrimePredictionApp

warnings.filterwarnings("ignore")

models = ['rnn']

data_dirs = ["data/ASSAULT.csv", "data/BATTERY.csv", "data/CRIMINAL DAMAGE.csv", "data/NARCOTICS.csv", "data/THEFT.csv"]

batch_sizes = [4, 8, 16, 32, 64]
learning_rates = [0.1, 0.01, 0.001, 0.0001]
momentums = [0.9, 0.95, 0.99]
regs = [0.0005, 0.001, 0.05, 0.1, 1, 10]
betas = [1.005, 0.999, 0.995, 0.99]

# parameters for RNN only
model_types = ["RNN", "LSTM", "GRU"]
num_layers = [1, 2, 3]
dropouts = [0, 0.2, 0.4]

params = [data_dirs, batch_sizes, learning_rates, momentums, regs, betas, model_types, num_layers, dropouts]
param_list = list(itertools.product(*params))


default_config = toml.load('config.toml')
for model in models:
    keys = ['batch_size', 'learning_rate', 'momentum', 'reg', 'beta', 'model_type', 'num_layer', 'dropout']
    config = default_config.copy()
    config["proc"]["run_ffn"] = False
    config["proc"]["run_cnn"] = False
    config["proc"]["run_rnn"] = True

    best_models = []
    for param in param_list:
        config["data_directory"] = param[0]
        config["config"][f"{model}"].update(dict(zip(keys, param[1:])))

        try:
            app = CrimePredictionApp.from_dict(config)
            requests = CrimePredictionApp.build_requests_from_dict(config)
            res = app._main(requests=requests)
            config_df = pd.DataFrame(config["config"][f"{model}"], index=[0])
            config_df['dataset'] = param[0].split("/")[-1].split(".")[0]
            res = pd.concat([res, config_df], axis=1)
            best_models.append(res)
        except Exception as e:
            logging.log("Unhandled Error", exc_info=e)
    best_models = pd.concat(best_models, ignore_index=True)
    best_models.to_csv(f'output/{model}_best_model.csv', index=False)
