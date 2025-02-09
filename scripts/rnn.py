"""
RNNModel creation and forwarding
"""

import torch
import torch.nn as nn


# RNN Model in PyTorch
class RNNModel(nn.Module):
    def __init__(self, input_size):
        super(RNNModel, self).__init__()
        self.rnn1 = nn.RNN(input_size=input_size, hidden_size=50, batch_first=True)
        self.rnn2 = nn.RNN(input_size=50, hidden_size=50, batch_first=True)
        self.fc1 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = torch.relu(x[:, -1, :])  # Use the last time step output
        return self.sigmoid(self.fc1(x))
