import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def file_to_tensor(file_path, text_length):
    fi = open(file_path, 'r')

    k =fi.read()

    se = sorted(set(k))

    idx = dict()
    xdi = dict()

    for i,j in enumerate(se):
        idx[i] =j

    for i in idx.keys():
        xdi[idx[i]] = i

    wo = [xdi[ch] for ch in k]

    eg_no = len(wo)//text_length

    ba = torch.zeros(eg_no,text_length)

    i = 0
    for k in range(0,len(wo)-text_length,text_length):
        ba[i] = torch.Tensor(wo[k:k+text_length])
        i = i+ 1

    ba = ba[np.random.permutation(eg_no)]

    ba = ba.long()

    return ba, (xdi, idx)
