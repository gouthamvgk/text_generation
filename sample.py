import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sen_to_index(sentence, mapping):
    
    """
    Converts the given sentence into a list of indices corresponding to that mapping.
    """
    temp = []
    for k in sentence.split(' '):
        temp.append(mapping[k])
    return temp



def eva(model, datagen, warm_up, temperature = 0.2, pre_len = 30):


    """
    This function is used to generate text from the trained model.
    model: trained model for text generation
    datagen: dataset oject created using Dataset_gen class
    warm_up: Text used for warming up sampling process
    temperature: used to induce a randomess in sampling.
    pre_len: desired predicted length string.
    """


    model.train()
    inp_sentence = sen_to_index(warm_up.lower(), datagen.xdi)
    pre_string = warm_up
    hidden, cell = model.hidden_cell_init(1)
    hidden = hidden.to(device)
    cell = cell.to(device)


    for j in range(len(inp_sentence)-1):
        token = torch.LongTensor([[inp_sentence[j]]])
        token = token.to(device)
        _, (hidden,cell) = model(token, hidden, cell)

    inp = inp_sentence[-1]

    for k in range(pre_len):

        token = torch.LongTensor([[inp]])
        token = token.to(device)
        output, (hidden, cell) = model(token, hidden, cell)
        output = output.div(temperature)
        index = torch.max(output, 1)[1].item()
        pre_string += ' ' + datagen.idx[index]
        inp = index
    print(pre_string)
