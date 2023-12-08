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

        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=num_classes, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.transpose(x, 1, 2)
        x = torch.reshape(x, (-1, self.num_classes))
        x = self.softmax(x)
        return x

    @property
    def model_name(self):
        return self._model_name