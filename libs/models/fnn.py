import torch
import torch.nn as nn
import numpy as np


class FeedforwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        """
        :param input_dim:
        :param hidden_size:
        :param num_classes:
        """

        super(FeedforwardNetwork, self).__init__()

        self._model_name = 'FNN'

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.layer1 = nn.Linear(self.input_dim, self.hidden_size)
        self.layer2 = nn.Linear(self.hidden_size, self.num_classes)
        # self.activation = nn.Sigmoid()
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax()

    @property
    def model_name(self):
        return self._model_name

    def forward(self, x):

        out = torch.reshape(x, (-1, self.input_dim))
        out = self.layer1(out)
        out = self.activation(out)
        out = self.layer2(out)
        # out = self.softmax(out)
        # out = torch.argmax(out, axis=1)

        return out