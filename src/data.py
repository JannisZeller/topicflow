# ------------------------------------------------------------------------------
# src_lda.data
# ------------
# 
# This is the source file containing a class which samples couments int the toy 
# model by Griffiths & Steyver (2004, https://doi.org/10.1073/pnas.0307752101).
# It creates documents with a vocabulary of the size of int^2, i. e.
# they can be visualized as square-images. I use a custom procedure here 
# instead of a tfp-joined distribution (tfd.JointDistribution) for 2 reasons
# 1. I want full control and transparency for each step.
# 2. I learned about tfd.JointDistribution when I already finished this script.
# ------------------------------------------------------------------------------


# %% Dependencies
# ------------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions



            
# %% Square Documents Model
# ------------------------------------------------------------------------------

class squareLDDocuments(object):
    """Constructs documents generated by a latent dirichlet process which can be
    represented as suqare matrices of counts (vocab size is square).
    """
    def __init__(
        self,
        N_docs: int, 
        sqrt_N_vocab:  int,
        N_words: int,
        uniform_doclengths: bool=True,
        alpha: float=1.,
        Theta_overwrite: tf.Tensor=None,
        ) -> object:
        """
        Parameters
        ----------
        N_docs : int
            Number of documents
        sqrt_N_vocab : int
            Size of square root of vocab size. Number of latent topic 
            automatically becomes 2*sqrt_N_vocab
        N_words : int
            Number of words per document.
        uniform_doclengths : bool=True
            Bool indicating whether the documents should vary in length. If 
            False the document lengths get sampled from a poisson distribution.
        alpha : float
            Sparsity of Document-Topic matrix (higher is LESS sparse)
        theta_overwrite : tf.Tensor
            Theta to overwrite the construction of theta as stripes of the vocab
            grid (mainly to present the predict-method of sampler.gibbsSampler).
        """
        
        ## Overall data specifications
        self.N_topics = int(2*sqrt_N_vocab)
        self.N_vocab = int(sqrt_N_vocab**2)
        self.N_docs = N_docs
        self.N_words = N_words
        self.alphas = self.N_topics*[alpha]
        self.uniform_doclengths = uniform_doclengths

        ## Sampling
        self.__sample_single_lengths()
        self.__construct_word_gird(sqrt_N_vocab)
        self.__construct_topic_word_prevalences(Theta_overwrite)
        self.__sample_document_topic_prevalences()
        self.__sample_word_topic_assignments()
        self.__sample_words()



    ## "In-Out" Capabilities
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    ## Extract the parameters / data from the object as array-likes
    def extract_params(self):
        return self.Theta, self.Pi, self.C_DId, self.C_DIdK, self.W_DId

    ## Extract the single document lengths as a np.ndarray
    def get_doc_lengths(self):
        return self.single_lengths.numpy()



    ## Parts of the sample
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    ## Construction of the word grid to extract the "topics" which can be 
    #  visualized as "stripes" in this grid.
    def __construct_word_gird(self, sqrt_N_vocab: int):
        N_vocab = int(sqrt_N_vocab**2)
        self.V_grid = np.reshape(np.arange(0, N_vocab), 
                                 (sqrt_N_vocab, sqrt_N_vocab))

    ## Generate single document lengths, depending whether specified with fixed
    #  or random (Poisson-) document lengths.
    def __sample_single_lengths(self):

        if not self.uniform_doclengths:
            single_lengths_dist = tfd.Poisson(rate=self.N_words)
            single_lengths      = tf.cast(
                single_lengths_dist.sample(self.N_docs), dtype=tf.int32)
            self.single_lengths = single_lengths

        if self.uniform_doclengths:
            self.single_lengths = tf.constant(
                self.N_docs*[self.N_words], dtype=tf.int32)

    ## Construction of the topic-word-prevalences (Theta) as "stripes" in the
    #  word grid.
    def __construct_topic_word_prevalences(self, Theta_overwrite):
        if Theta_overwrite is None:
            Theta_idx = ([row for row in self.V_grid] + 
                [col for col in self.V_grid.T])
            Theta = np.zeros((self.N_topics, self.N_vocab))
            denominator = self.V_grid.shape[0]
            for k, idx in enumerate(Theta_idx):
                Theta[k, idx] = 1. / denominator
            self.Theta = tf.constant(Theta, dtype=tf.float32)
        else:
            self.Theta = Theta_overwrite

    ## Sample document-topic prevalences from a Dirichlet. Gets more sparse 
    #  (document is more "focussed" on few topics) with smaller values of alpha.
    def __sample_document_topic_prevalences(self):
        dist_Pi = tfd.Dirichlet(self.alphas)
        Pi      = dist_Pi.sample(self.N_docs)
        self.Pi = Pi

    ## Sample word-topic-assigments (C-matrix) as categoricals
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
        self.C_DIdK = C_DIdK

    ## Sample the individual words.
    def __sample_words(self):
        if not self.uniform_doclengths:
            C_T = self.C_DId.flat_values
            dist_W_T = tfd.Categorical(probs=tf.gather(self.Theta, C_T))
            W_T = dist_W_T.sample()
            W_DId = tf.RaggedTensor.from_row_lengths(W_T, 
                *self.C_DId.nested_row_lengths())
        if self.uniform_doclengths:
            dist_W_DN = tfd.Categorical(probs=tf.gather(self.Theta, self.C_DId))
            W_DId = dist_W_DN.sample() 
        self.W_DId = W_DId