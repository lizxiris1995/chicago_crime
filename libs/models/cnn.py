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
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=3, padding=2)
        self.fc1 = nn.Linear(in_features=2000, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = torch.reshape(x, (x.shape[0], -1))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.softmax(x)
        return x

    @property
    def model_name(self):
        return self._model_name