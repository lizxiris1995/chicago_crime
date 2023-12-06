import numpy as np
import pandas as pd
from typing import Dict


class BaseRequest(object):
    def __init__(self,
                 data: np.array,
                 ffn_config: Dict,
                 cnn_config: Dict,
                 rnn_config: Dict):

        self._data = data
        self._ffn_config = ffn_config
        self._cnn_config = cnn_config
        self._rnn_config = rnn_config

    @property
    def input_data(self):
        return self._data

    @property
    def ffn_config(self):
        return self._ffn_config

    @property
    def cnn_config(self):
        return self._cnn_config

    @property
    def rnn_config(self):
        return self._rnn_config