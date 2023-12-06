import pandas as pd
import numpy as np
import os
import toml
from typing import Dict, List
from .utils import *
from .Request import *
from .DataProvider import *


class CrimePredictionApp(object):
    def __init__(self,
                 run_ffn: bool,
                 run_cnn: bool,
                 run_rnn: bool,
                 output_dir: str):

        self._run_ffn = run_ffn
        self._run_cnn = run_cnn
        self._run_rnn = run_rnn

        self._output_dir = output_dir
        # create_missing_dirs(self._output_dir)

    def _main(self, requests):
        for request in requests:
            if self._run_ffn:
                self._main_ffn(request.input_data, request.ffn_config)
            if self._run_cnn:
                self._main_cnn(request.input_data, request.cnn_config)
            if self._run_rnn:
                self._main_rnn(request.input_data, request.rnn_config)

    def _main_ffn(self, data, config):
        # TODO:
        raise NotImplementedError

    def _main_cnn(self, data, config):
        # TODO:
        raise NotImplementedError

    def _main_rnn(self, data, config):
        # TODO:
        raise NotImplementedError

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
        output_dir = d.get("output_directory")

        return cls(run_ffn=run_ffn, run_cnn=run_cnn, run_rnn=run_rnn, output_dir=output_dir)

    @staticmethod
    def build_requests_from_dict(d: dict):
        requests = []

        # TODO: get data from data provider
        data = np.array([])
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

