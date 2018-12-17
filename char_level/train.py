import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
import numpy as np
from sample import eval
from model import predict
from preprocess import file_to_tensor
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def one_hot(lis, le):
    one_hot = torch.zeros(len(lis), le)
    for j in range(len(lis)):
        one_hot[j][lis[j]] = 1
    return one_hot



def train_and_sample(model, optimizer, criterion , ba, xdi, idx,no_epoch = 1, batch_size = 5, print_every = 100, eval_every = 100, save_every = 500, save_path = None):
    batch_size = batch_size
    eg_no = ba.size()[0]
    text_length = ba.size()[1]

    for ep in range(no_epoch):
        since = time.time()
        total_loss = []
        loss = 0
        iteration = 1

        for i in range(0,eg_no-batch_size,batch_size):
            hidden, cell = model.init_hidden(batch_size)
            hidden = hidden.to(device)
            cell = cell.to(device)
            optimizer.zero_grad()
            model.train()

            for k in range(text_length-1):
                li = []
                li2 = []

                for j in range(batch_size):
                    li2.append(ba[i+j][k].item())
                    li.append(ba[i+j][k+1].item())

                inp = one_hot(li2, len(xdi))
                inp = inp.unsqueeze(0)
                inp = inp.to(device)

                output, (hidden, cell) = model(inp, hidden, cell)
                rite = torch.LongTensor(li)
                rite = rite.to(device)
                loss += criterion(output,rite)

            if ((iteration % print_every) == 0):
                print('Epoch {}'.format(ep+1))
                print('Iteration {}'.format(iteration))
                print('Loss for this iteration: {}'.format(loss.item()))

            total_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            loss = 0

            if((iteration%eval_every) == 0):
                eval(model,xdi, idx, temperature = 0.4, predict_len = 200)
                eval(model,xdi, idx, temperature = 0.5, predict_len = 200)
                eval(model,xdi, idx, temperature = 0.6, predict_len = 200)
                eval(model,xdi, idx, temperature = 0.8, predict_len = 200)
                eval(model,xdi, idx, temperature = 0.4, start = 'at the', predict_len = 200)
                eval(model,xdi, idx, temperature = 0.5, start = 'h', predict_len = 200)
                eval(model,xdi, idx, temperature = 0.6, start = 'he', predict_len = 200)
                eval(model,xdi, idx, temperature = 0.8, start = 'v', predict_len = 200)

            if (save_path and (iteration%save_every) == 0):
                torch.save({'model': model.state_dict(),
                            'optimizer': optimizer.state_dict()}, save_path +'/char_' + str(iteration) + '.pth')
            iteration += 1
        elapsed_time = time.time() - since
        print('Epoch completed in {:.0f}minutes {:.0f}seconds'.format(elapsed_time//60, elapsed_time%60))
        print('Loss for this epoch is {}'.format(sum(total_loss)/len(total_loss)))



if __name__ == "__main__":
    learning_rate = 0.005
    text_length = 200
    hidden_size = 500
    num_layers = 2
    path = ""   #path to the text file
    no_epoch = 1
    batch_size = 4
    print_every = 100
    eval_every = 200
    save_every = 1000
    save_path = "" #path to save the model
    ba, (xdi, idx) = file_to_tensor(path,text_length)
    vocab_size = len(xdi)
    model = predict(hidden_size, vocab_size, num_layers = num_layers)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_and_sample(model, optimizer, criterion, ba, xdi, idx, no_epoch = no_epoch, batch_size = batch_size, print_every = print_every, eval_every = eval_every, save_every = save_every, save_path = save_path)
