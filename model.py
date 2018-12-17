import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class word_gen(nn.Module):
    """
    Module for word level text generation.  It uses LSTM with pre-trained embeddings for
    words in the vocabulary.
    """

    def __init__(self, voc_size, pre_embeddings, hidden_size = 500, num_layers = 2, dropout = 0.5):
        super(word_gen, self).__init__()
        self.voc_size = voc_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_layer = nn.Embedding(self.voc_size, 50)
        self.embedding_layer.weight.data.copy_(torch.from_numpy(pre_embeddings).float().cuda())
        self.lstm = nn.LSTM(50, hidden_size, num_layers = num_layers, dropout = dropout)
        self.predict = nn.Linear(hidden_size, self.voc_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp, hn, cn):
        embeddings = self.embedding_layer(inp)
        output, (hidden, cell) = self.lstm(embeddings, (hn, cn))
        predict = self.predict(output[0])
        return predict, (hidden, cell)


    def hidden_cell_init(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden, cell
