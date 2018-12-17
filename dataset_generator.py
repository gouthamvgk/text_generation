import numpy as np
import bcolz
import pickle





class Dataset_gen():


    """
    This class is used to construct the dataset in the required format for training the model.
    It also parses the input file and creates mappings and embeddings for processing.
    path: path to the text file that contains the text
    vect_path: path to the embedding container created using bcolz
    word_path: path to the serialised word list corresponding to pre-trained Embeddings
    word2ind_path: path to the word to index mapping corresponding to pre-trained Embeddings
    seq_len: Tells the sequence length of each sentence so that dataset can be created according to it
    step: Tell how much step we have to parse while constructing the dataset from the text.

    """


    def __init__(self, path, vect_path, word_path, word2ind_path, seq_len = 30, step = 10):


        """
        Initialises the object.  Self.content takes the text file and does some pre-processing
        required for parsing.
        """


        self.seq_len = seq_len
        self.path1 = vect_path
        self.path2 = word_path
        self.path3 = word2ind_path
        self.step = step
        self.content = open(path, 'r').read().replace('\n', ' ').replace('   ', ' ').replace('  ', ' ')
        self.idx = {}
        self.xdi = {}
        self.vocab_list = []
        self.vocab_count = {}
        self.vocab_length = 0
        self.sentence_list = []
        self.data_index = []
        self.embedding = []



    def create_vocab(self, sentence_list):


        """
        This function creates the vocab_list that contains all the words in the text
        and the the vocab_count dictionary which contains how many times each word has occured.
        sentence_list: contains list of sentences corresponding to the dataset.
        """

        self.vocab_list = []
        self.vocab_count = {}
        for j in sentence_list:
            for k in j:
                if k not in self.vocab_list:
                    self.vocab_list.append(k)
                    self.vocab_count[k] = 1
                else:
                    self.vocab_count[k] += 1
        self.vocab_list.sort()
        self.vocab_length = len(self.vocab_list)


    def create_dataset(self):

        """
        This function is used to create the sentences from the whole list.
        It parses the whole text and constructs individual sentences of given
        length and takes steps as mentioned.
        """


        lis = self.content.split(' ')
        for j in range(0, len(lis)-self.seq_len, self.step):
            self.sentence_list.append(lis[j:j+self.seq_len])

        self.create_vocab(self.sentence_list)


        print('Totally formed {} sequences'.format(len(self.sentence_list)))
        print('Found {} unique words'.format(self.vocab_length))


    def remove_words(self, length = 15, occurence = 4):


        """
        This function is used to remove sentences from the dataset which contains
        words that is above the given length and appears less times than the mentioned
        occurence.  This function cannot be called after creating the mappings and
        embedding matrix.  Data cleaning should be carried out before this.
        length: length of words which should be considered for removal
        occurence: word frequency necessary for it to be retained.
        """


        temp_sen_len = len(self.sentence_list)
        temp_voc_len = len(self.vocab_list)
        if (len(self.idx) > 1):
            return 'Mappings already created cannot remove words now'
        remove_list = []
        for keys in self.vocab_count.keys():
            if len(keys) >= 15 and self.vocab_count[keys] < 4:
                remove_list.append(keys)
        new_list = []
        for k in self.sentence_list:
            flag = 0
            for each in remove_list:
                if (each in k):
                    flag = 1
                    break
            if (flag == 0):
                new_list.append(k)
        self.sentence_list = new_list
        self.create_vocab(self.sentence_list)
        print('Out of {} sequences {} are retained'.format(temp_sen_len, len(self.sentence_list)))
        print('Out of {} words in vocabulary {} are retained'.format(temp_voc_len, self.vocab_length))



    def create_mappings_dataset(self):


        """
        This function maps the words in the vocabulary of the dataset to some index
        and also creates an inverse mapping of that for processing.
        It also creates the sentence list based on index from that of words
        """


        for i,j in enumerate(self.vocab_list):
            self.idx[i] = j
            self.xdi[j] = i
        print('Mappings created')
        for i in range(len(self.sentence_list)):
            temp = []
            for k in self.sentence_list[i]:
                temp.append(self.xdi[k])
            self.data_index.append(temp)
        print('Dataset formed')


    def create_embeddings(self):


        """
        This function is used to created the embedding matrix for using in the
        embedding layer of the model.  We have obtained pre-trained embeddings based
        on glove and loaded it in memory. A numpy array is created which
        contains a 50-D embedding for each word in vocabulary.  If a word is
        present in vocabulary which has no pre-trained embedding then a
        random one in initialised for it.
        """


        vectors = bcolz.open(self.path1)[:]
        words = pickle.load(open(self.path2, 'rb'))
        word2idx = pickle.load(open(self.path3, 'rb'))

        glove = {w: vectors[word2idx[w]] for w in words}

        self.embedding = np.zeros((self.vocab_length, 50))
        for i,j in self.idx.items():
            try:
                self.embedding[i] = glove[j]
            except KeyError:
                self.embedding[i] = np.random.normal(scale=0.6, size=(50, ))
        print('Embeddings created')



        
