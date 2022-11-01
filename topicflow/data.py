# ------------------------------------------------------------------------------
# topicflow.data
# --------------
# 
# Datasets and samplers to be used with `topicflow.models`.
# ------------------------------------------------------------------------------


# %% Dependencies
# ------------------------------------------------------------------------------
from typing import Union
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from collections.abc import Sequence
import matplotlib.pyplot as plt


# %% sliceConverter
# ------------------------------------------------------------------------------
## Convertes slices of given document-lengths to an iterator such that they can 
#  be iterated easily.
class sliceConverter(Sequence):
    """Constructs an iteratable sequence to construct slices from document 
    lenghts.
    """
    def __init__(self, N: np.ndarray):
        assert N.ndim == 1
        N_idx = np.cumsum(N)
        N_idx = np.repeat(N_idx, 2)
        self.N_idx   = np.concatenate([[0], N_idx], axis=0)
        self.N_max   = np.mean(N)
        self.N_total = N_idx[-1].numpy()
        self.single_lengths = N

    def __len__(self):
        return int((self.N_idx.shape[0] - 1) / 2)

    def __getitem__(self, k):
        if type(k) == int:
            if k >= 0:
                return slice(self.N_idx[2*k], self.N_idx[2*k + 1], 1)
            if k < 0:
                i = len(self) + k
                return slice(self.N_idx[2*i], self.N_idx[2*i + 1], 1)
        elif type(k) == slice:
            start, stop, step = k.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        else:
            raise TypeError("Index must be integer or slice.")

    def __iter__(self):
        self.idx = 0
        self.maxiter = len(self)
        return self

    def __next__(self):
        if self.idx < self.maxiter:
            ret = self.__getitem__(self.idx)
            self.idx += 1
            return ret
        else:
            raise StopIteration

            
# %% Square Documents Model
# ------------------------------------------------------------------------------
## Inspired by Griffiths & Steyver (2004, 
#  https://doi.org/10.1073/pnas.0307752101). Can be used as a visual tool to
#  check models. Creates documents with a vocabulary of the size of int^2, i. e.
#  they can be visualized as square-images.

class squareLDDocuments(object):
    """Constructs documents generated by a latent dirichlet process which can be
    represented as suqare matrices of counts (vocab size is square).
    """
    def __init__(
        self,
        N_docs: int, 
        sqrt_N_vocab:  int,
        N_words_fixed: int=None,
        N_words_rate:  int=None,
        alpha: float=1.,
        Theta_overwrite: Union[tf.Tensor, np.ndarray]=None
        ) -> object:
        """
        Parameters
        ----------
        N_docs : int
            Number of documents
        sqrt_N_vocab : int
            Size of square root of vocab size. Number of latent topic 
            automatically becomes 2*sqrt_N_vocab
        N_words_fixed : int=None
            Number of words per document fixed.
        N_words_rate : int=None
            Number of words per document as rate of a poisson distribution
        Alphas: float
            Sparsity of Document-Topic matrix (higher is LESS sparse)
        """
        
        ## Overall
        assert (N_words_fixed is None) != (N_words_rate is None), "Exactly one of N_words_fixed and N_words_rate must be passed."
        assert (Theta_overwrite is None) or (Theta_overwrite.shape == (2*sqrt_N_vocab, sqrt_N_vocab**2)), "Incompatible `Theta_overwrite` - shape. Must be `(2*sqrt_N_vocab, sqrt_N_vocab**2)`."
        self.N_topics = int(2*sqrt_N_vocab)
        self.N_vocab = int(sqrt_N_vocab**2)
        self.N_docs = N_docs
        self.alphas = self.N_topics*[alpha]
        if N_words_rate is not None:
            self.uniform_doclengths = False
        if N_words_fixed is not None:
            self.uniform_doclengths = True

        ## Number of words per document
        self.__sample_single_lengths(N_docs, N_words_fixed, N_words_rate)
        
        ## Word grid
        self.__construct_word_gird(sqrt_N_vocab)

        ## Topic-Word Distributions
        if Theta_overwrite is None:
            self.__construct_topic_word_prevalences()
        else: 
            self.Theta = Theta_overwrite

        ## Document-Topic Distribution
        self.__sample_document_topic_prevalences()

        ## Topic Assignments of word c_{dik} of word w_{di}
        self.__sample_word_topic_assignments()

        ## Draw words w_{di}
        self.__sample_words()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    def extract_params(self):
        return self.Theta, self.Pi, self.C_DId, self.C_DIdK, self.W_DId


    def get_doc_lengths(self):
        return self.single_lengths.numpy()


    def __sample_single_lengths(self, 
        N_docs: int,
        N_words_fixed: int=None,
        N_words_rate:  int=None):

        if N_words_rate is not None:
            single_lengths_dist = tfd.Poisson(rate=N_words_rate)
            single_lengths      = tf.cast(single_lengths_dist.sample(N_docs), 
                                          dtype=tf.int32)
            self.single_lengths = single_lengths

        if N_words_fixed is not None:
            self.single_lengths = tf.constant(N_docs*[N_words_fixed],
                                              dtype=tf.int32)


    def __construct_word_gird(self, sqrt_N_vocab: int):
        N_vocab = int(sqrt_N_vocab**2)
        self.V_grid = np.reshape(np.arange(0, N_vocab), 
                                 (sqrt_N_vocab, sqrt_N_vocab))


    def __construct_topic_word_prevalences(self):
        Theta_idx = ([row for row in self.V_grid] + 
                     [col for col in self.V_grid.T])
        Theta = np.zeros((self.N_topics, self.N_vocab))
        denominator = self.V_grid.shape[0]
        for k, idx in enumerate(Theta_idx):
            Theta[k, idx] = 1. / denominator
        self.Theta = tf.constant(Theta, dtype=tf.float32)


    def __sample_document_topic_prevalences(self):
        dist_Pi = tfd.Dirichlet(self.alphas)
        Pi      = dist_Pi.sample(self.N_docs)
        self.Pi = Pi


    def __sample_word_topic_assignments(self):
        dist_C  = tfd.Categorical(probs=self.Pi)

        if not self.uniform_doclengths:
            C_NmaxD = dist_C.sample(tf.reduce_max(self.single_lengths))
            C_DId_list = []
            for d in range(self.N_docs):
                C_DIdd = C_NmaxD[:self.single_lengths[d], d] # Cropping a single doc length
                C_DId_list.append(C_DIdd) # Appending to list
            C_DId = tf.ragged.stack(C_DId_list)

        if self.uniform_doclengths:
            C_IdD = dist_C.sample(self.single_lengths[0])
            C_DId = tf.transpose(C_IdD)

        C_DIdK = tf.one_hot(C_DId, depth=self.N_topics, axis=-1)
        
        self.C_DId = C_DId
        self.C_DIdK =  C_DIdK


    def __sample_words(self):

        if not self.uniform_doclengths:
            C_T = self.C_DId.flat_values
            dist_W_T = tfd.Categorical(probs=tf.gather(self.Theta, C_T))
            W_T = dist_W_T.sample()
            W_DId = tf.RaggedTensor.from_row_lengths(
                W_T, 
                *self.C_DId.nested_row_lengths())

        if self.uniform_doclengths:
            dist_W_DN = tfd.Categorical(probs=tf.gather(self.Theta, self.C_DId))
            W_DId = dist_W_DN.sample() 

        self.W_DId = W_DId