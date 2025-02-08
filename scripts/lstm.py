import torch
import torch.nn as nn





# LSTM Model in PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=50, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=50, batch_first=True)
        self.fc1 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()
