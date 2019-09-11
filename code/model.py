import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Embedding
from keras.models import Model
from keras.callbacks import EarlyStopping
from utils import build_output
from code.sklearn_classifiers import get_glove, load_word2vec


class WSDModel(object):
    def __init__(self,
                 input_vocab,
                 output_vocab,
                 batch=False,
                 use_glove=False,
                 train_embeddings=False,
                 dropout_rates=None):
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.train_embeddings = train_embeddings
        self.vocab_size = len(self.input_vocab)
        self.use_glove = use_glove
        self.embedding_matrix = {}

        assert self.vocab_size == len(self.output_vocab), "Vocab should be the same size"

        self.batch = batch
        self.dropout_rates = dropout_rates
        self.optimizer = 'adam'
        self.epochs = 5
        self.history = None
        self.model = None


        self.params = {
            'dropout_rates': self.dropout_rates,
            'epochs': self.epochs,
            'model': self.model
        }

        def set_up(self, **kwargs):
            build_output()
            build_input()

        def build_embedding_layer(self):
            if self.train_embeddings:
                self.embedding_matrix = np.random.normal(
                    size=(self.vocab_size, self.embedding_dim))
                trainable = True
            else:
                if not self.use_glove:
                    self.embedding_matrix = load_word2vec(self.embedding_path)
                else:
                    self.embedding_matrix = get_glove(vocab=self.vocab)
                trainable = False

            embedding_layer = Embedding(
                self.vocab_size,
                self.embedding_dim,
                weights=[self.embedding_matrix],
                input_length=self.max_seq_len,
                trainable=trainable)

            return embedding_layer

        def build_model(self):
            sequence_input = Input(shape=(self.max_seq_len,), dtype='int32')
            embedding_layer = self.build_embedding_layer()
            embedded_sequences = embedding_layer(sequence_input)
            predictions = self.transform_embedded_sequences(embedded_sequences)
            model = Model(sequence_input, predictions)
            return model
