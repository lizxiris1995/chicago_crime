import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalNetwork(nn.Module):
    def __init__(self, input_dim, num_classes):
        """
        :param input_dim:
        :param num_classes:
        """
        super().__init__()

        self._model_name = 'CNN'
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=20, kernel_size=3, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=25, kernel_size=3, padding=2)
        self.linear1 = nn.Linear(in_features=1000, out_features=50*num_classes)
        self.activation = nn.ELU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.activation(self.conv1(x))
        out = self.maxpool1(out)
        out = self.activation(self.conv2(out))
        out = torch.reshape(out, (batch_size, -1))
        out = self.activation(self.linear1(out))
        out = torch.reshape(out, (-1, self.num_classes))
        out = self.softmax(out)
        return out

    @property
    def model_name(self):
        return self._model_name