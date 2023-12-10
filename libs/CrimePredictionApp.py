import pandas as pd
import numpy as np
import os
import toml

from os import listdir
from os.path import isfile, join
from typing import Dict, List
from .utils import *
from .Request import *
from .DataProvider import *
from .models import fnn, cnn


feature_list = ['is_Weekend', 'Holiday', '5d rolling avg', '30 rolling avg',
                'precipitation', 'Average Temp Norm', 'total population',
                'median income', '% 25 older', '% married', '% highschool graduates',
                '% foreign', '% poverty', '% own house', 'rides', 'housing_price_1b',
                'housing_price_2b', 'housing_price_3b']
label = 'Bin'


class CrimePredictionApp(object):
    def __init__(self,
                 run_ffn: bool,
                 run_cnn: bool,
                 run_rnn: bool,
                 app_dir: str):

        self._run_ffn = run_ffn
        self._run_cnn = run_cnn
        self._run_rnn = run_rnn

        self._output_dir = app_dir + "/output"
        # create_missing_dirs(self._output_dir)

    def _main(self, requests):
        resL = []
        for request in requests:
            if self._run_ffn:
                res_ffn = self._main_ffn(request.input_data, request.ffn_config)
                resL.append(res_ffn)
            if self._run_cnn:
                res_cnn = self._main_cnn(request.input_data, request.cnn_config)
                resL.append(res_cnn)
            if self._run_rnn:
                res_rnn = self._main_rnn(request.input_data, request.rnn_config)
                resL.append(res_rnn)
        res = pd.concat(resL, ignore_index=True)
        return res

    def _main_ffn(self, data, config):

        data.fillna(0)

        num_features = len(feature_list)
        num_bin = int(np.nanmax(data['Bin'].unique()))

        model = fnn.FeedforwardNetwork(num_features, 10, num_bin)
        best_model = train_model(model, data, self._output_dir, **config)

        return best_model

    def _main_cnn(self, data, config):

        data = data.sort_values(by=['Date', 'Ward'])
        data.fillna(0)

        num_features = len(feature_list)
        num_bin = int(np.nanmax(data['Bin'].unique()))
        model = cnn.ConvolutionalNetwork(num_features, num_bin)
        best_model = train_model(model, data, self._output_dir, **config)

        return best_model

    def _main_rnn(self, data, config):

        data = data.sort_values(by=['Date', 'Ward'])
        data.fillna(0)

        num_features = len(feature_list)
        num_bin = int(np.nanmax(data['Bin'].unique()))
        model = rnn.RecurrentNeuralNetwork(num_features, num_bin, model_type=config['model_type'],
                                           num_layers=config['num_layer'], dropout=config['dropout'])
        best_model = train_model(model, data, self._output_dir, **config)

        return best_model

    @classmethod
    def from_toml(cls, file: str):
        with open(file) as toml_file:
            config = toml.load(toml_file.read())

        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, d: Dict):
        proc = d.get("proc")
        run_ffn = proc.get("run_ffn")
        run_cnn = proc.get("run_cnn")
        run_rnn = proc.get("run_rnn")
        app_dir = d.get("app_directory", os.getcwd())

        return cls(run_ffn=run_ffn, run_cnn=run_cnn, run_rnn=run_rnn, app_dir=app_dir)

    @staticmethod
    def build_requests_from_dict(d: dict):
        requests = []

        # LOAD DATA
        app_dir = os.getcwd()
        data_path = app_dir + '/' + d.get("data_directory", "data")
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        else:
            files = [f for f in listdir(data_path) if isfile(join(data_path, f)) if f.endswith(".csv")]
            dfL = []
            for file in files:
                crime_type = file.split('.')[0]
                df = pd.read_csv(data_path + '/' + file)
                df['Crime Type'] = crime_type
                dfL.append(df)
            data = pd.concat(dfL, ignore_index=True)
        data = add_quarter_index(data)

        # Configs
        config = d.get("config")
        ffn_config = config.get("ffn")
        cnn_config = config.get("cnn")
        rnn_config = config.get("rnn")
        request = BaseRequest(data=data,
                              ffn_config=ffn_config,
                              cnn_config=cnn_config,
                              rnn_config=rnn_config)
        requests.append(request)

        return requests

