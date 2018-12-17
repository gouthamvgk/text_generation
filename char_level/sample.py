import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def one_hot(lis, le):
    one_hot = torch.zeros(len(lis), le)
    for j in range(len(lis)):
        one_hot[j][lis[j]] = 1
    return one_hot




def eval(model,xdi, idx,  start = 'a', predict_len = 100, temperature = 0.5):


    model.eval()
    predict_string = start
    hidden, cell = model.init_hidden(1)
    hidden = hidden.to(device)
    cell = cell.to(device)


    for i in range(len(start) -1):
        li = [xdi[start[i]]]
        inp = one_hot(li, len(xdi))
        inp = inp.unsqueeze(0)
        inp = inp.to(device)
        _, (hidden, cell) = model(inp, hidden, cell)
    inp = xdi[start[-1]]


    for k in range(predict_len):
        li = [inp]
        inp = one_hot(li, len(xdi))
        inp = inp.unsqueeze(0)
        inp = inp.to(device)
        output, (hidden, cell) = model(inp, hidden, cell)
        output = output.div(temperature).exp()
        inp = torch.multinomial(output, 1).item()
        predict_string += idx[inp]


    print(predict_string)
