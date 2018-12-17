import numpy as np
import bcolz
import pickle
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
import time
from model import word_gen
from dataset_generator import Dataset_gen
from sample import eva

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_and_sample(model, optimizer, criterion , datagen, no_epoch = 1, batch_size = 5, print_every = 100, eval_every = 100, save_every = 5000, save_path = None):



    """
    This function is used to train the model and evaluate it by obtaining samples from it in mentioned interval.
    model: The word_gen model for text generation
    optimizer : The optimizer object corresponding to the model.
    criterion: The loss object.
    datagen : The Dataset object which contains the sentence indexes, mappings, details about vocabulary and
              also the embeddings for the Embedding layer.
    no_epoch: no_epochs the training has to be performed.
    batch_size: Number of sentences in each batch.
    print_every: Tells the interval in which loss has to be printed.
    eval_every: Tells the interval in which model has to be evaluated by sampling text from it.
    save_every: Tells the interval in which model has to be checkpointed.
    save_path" Path for saving the model.
    """


    batch_size = batch_size
    eg_no = len(datagen.data_index) #no. of sentences
    text_length = datagen.seq_len #sequence length

    for ep in range(no_epoch):
        since = time.time()
        data_index = np.asarray(datagen.data_index)
        data_index = data_index[np.random.permutation(eg_no)] #rondomly shuffling the dataset.
        total_loss = []
        loss = 0
        iteration = 1

        for i in range(0,eg_no-batch_size,batch_size):
            hidden, cell = model.hidden_cell_init(batch_size)
            hidden = hidden.to(device)
            cell = cell.to(device)
            optimizer.zero_grad()
            model.train()

            for k in range(text_length):
                li = []
                li2 = []
                for j in range(batch_size):
                    li2.append(data_index[i+j][k])  #caputuring the word index in each time step for all the eg. in a batch.
                    if (k == text_length - 1):
                        li.append(data_index[i+j][k])  #capturing the word index for the last time step alone.

                inp = torch.LongTensor([li2])
                inp = inp.to(device)
                if (k != text_length -1):
                    _, (hidden, cell) = model(inp, hidden, cell)
                else:
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
                warm_up = "harry and hermione were looking through the corridor"
                eva(model, datagen,warm_up , temperature = 0.4, predict_len = 30)
                eva(model,datagen,warm_up, temperature = 0.6, predict_len = 30)
                eva(model,datagen, warm_up , temperature = 0.8, predict_len = 30)


            if (save_path and (iteration%save_every) == 0):
                torch.save({'model': model.state_dict(),
                            'optimizer': optimizer.state_dict()}, save_path +'/word_' + str(iteration) + '.pth')
            iteration += 1
        elapsed_time = time.time() - since
        print('Epoch completed in {:.0f}minutes {:.0f}seconds'.format(elapsed_time//60, elapsed_time%60))
        print('Loss for this epoch is {}'.format(sum(total_loss)/len(total_loss)))




if __name__ == "__main__":
    learning_rate = 0.005
    seq_length = 30
    hidden_size = 500
    num_layers = 2
    path = ""   #path to the text file
    no_epoch = 1
    batch_size = 10
    print_every = 100
    eval_every = 200
    save_every = 1000
    save_path = "" #path to save the model
    vect_path = '' #path to the pretrained embedding container
    word_path = '' #path to the word list of pretrained embedding
    word2ind_path = '' #path to vocabulary dictionary of the embedding

    data_gen = Dataset_gen(path , vect_path, word_path, word2ind_path, seq_len = seq_length) #dataset loader
    data_gen.create_dataset() #This takes some time for larger text files.
    data_gen.remove_words(15,4)
    data_gen.create_mappings_dataset()
    data_gen.create_embeddings()


    model = predict(data_gen.vocab_length, data_gen.embedding, hidden_size = hidden_size, num_layers = num_layers)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_and_sample(model, optimizer, criterion, data_gen, no_epoch = no_epoch, batch_size = batch_size, print_every = print_every, eval_every = eval_every, save_every = save_every, save_path = save_path)
