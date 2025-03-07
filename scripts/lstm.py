"""
LSTMModel creation and forwarding
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """LSTM Model in PyTorch."""
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=50, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=50, batch_first=True)
        self.fc1 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = torch.relu(x[:, -1, :])
        return self.sigmoid(self.fc1(x))
