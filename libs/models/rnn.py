import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_size=32, num_layers=1, dropout=0.2):
        """
        :param input_dim:
        :param num_classes:
        """
        super().__init__()

        self._model_name = 'RNN'
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_wards = 50

        self.rnn = nn.RNN(input_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size*self.num_wards, self.num_wards*self.num_classes)
        self.activation = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        out, _ = self.rnn(x)
        out = out.reshape(batch_size, -1)
        out = self.activation(self.linear(out))
        out = torch.reshape(out, (-1, self.num_classes))
        out = self.softmax(out)
        return out

    @property
    def model_name(self):
        return self._model_name