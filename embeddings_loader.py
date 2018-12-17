import numpy as np
import bcolz
import pickle


"""
This code is used to create numpy array of pre-trained embeddings which is obtained
from the stanford NLP group.(https://nlp.stanford.edu/projects/glove/)
bcolz is a data container package which is used to store the extracted embeddings.
This code is obtained from a medium article by Martin Pellarolo
(https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76)
"""




words = []
idx = 0
word2idx = {}
vectors = bcolz.carray(np.zeros(1), rootdir='6B.50.dat', mode='w')

with open('glove.6B.50d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)

vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir='6B.50.dat', mode='w')  #stores the pre-trained embeddings container.
vectors.flush()
pickle.dump(words, open('6B.50_words.pkl', 'wb')) #serializes the vocab list and stores it
pickle.dump(word2idx, open('6B.50_idx.pkl', 'wb')) #serialises the word to index mapping and stores it
