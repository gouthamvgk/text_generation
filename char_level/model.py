import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class predict(nn.Module):


    def __init__(self, hidden_size, voc_size, num_layers = 2, dropout = 0.5):
        super(predict, self).__init__()
        self.hidden_size = hidden_size
        self.voc_size = voc_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.voc_size, self.hidden_size, num_layers=num_layers,dropout = dropout)
        self.pre = nn.Linear(self.hidden_size, self.voc_size)


    def forward(self, inp_word, hn, cn):
        output, (hidden,cell) = self.lstm(inp_word, (hn, cn))
        predict = self.pre(output[0])
        return predict,(hidden, cell)



    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers,batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers,batch_size, self.hidden_size)
        return h0, c0
