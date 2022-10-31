# ------------------------------------------------------------------------------
# topicflow.models
# ----------------
# 
# LDA Model.
# ------------------------------------------------------------------------------


# %% Dependencies
# ------------------------------------------------------------------------------
from typing import Union
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


# %% LDA Model
# ------------------------------------------------------------------------------
class LDA(object):
    def __init__(
        self, 
        N_topics: int, 
        N_iter: int=200,
        theta_prior: float=1.0,
        pi_prior:    float=1.0,
        fix_vocab: int=None,
        fit_procedure: str="gibbs",
        verbose: int=1):
        """ 
        """

        super(LDA, self).__init__()
        
        self._fit_procedure = fit_procedure
        self.N_iter = N_iter

        self.theta_prior_val = theta_prior
        self.pi_prior_val    = pi_prior

        self._N_topics = N_topics
        self._fix_vocab = fix_vocab

        self.verbose = 1


    def fit(self, W_DId: Union[np.ndarray, tf.Tensor, tf.RaggedTensor]):

        ## Determine if W_DId is ragged, i. e. if we have varying doclengths:
        if any([x==None for x in W_DId.shape]):
            self.ragged = True

        ## Extract Dimensions
        self.data_specs = {}
        self.data_specs["N_topics"] = self._N_topics
        self.data_specs["N_docs"] = W_DId.shape[0]

        if self._fix_vocab:
            self.data_specs["N_vocab"] = self._fix_vocab
        else:
            self.data_specs["N_vocab"] = int(tf.reduce_max(W_DId)) + 1

        if self.ragged:
            self.data_specs["max_doclength"] = int(tf.reduce_max(W_DId.nested_row_lengths()[0]))
        if not self.ragged:
            self.data_specs["max_doclength"] = W_DId.shape[1]

        ## Initializing stuff
        self.__init_C_DIdK_(W_DId)
        self.Theta_history = []

        if self._fit_procedure == "gibbs":

            ## Get in shape
            D = self.data_specs["N_docs"]
            K = self.data_specs["N_topics"]
            V = self.data_specs["N_vocab"]

            ## Setting up priors        
            self.theta_prior = (self.theta_prior_val * 
                tf.ones(shape=(K, V), dtype=tf.float32))
            self.pi_prior    = (self.pi_prior_val
                * tf.ones(shape=(D, K), dtype=tf.float32))

            ## Iterative Sampling  
            if self.verbose == 1:
                iterator = tqdm(range(self.N_iter))
            else: 
                iterator = range(self.N_iter)
            for _ in iterator:
                self._gibbs_sample(W_DId)
                self.Theta_history.append(self.Theta)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


    def _gibbs_sample(self, W_DId):

        ## Calculate N_DKV_ from W_DNmax and C_DNmaxK
        N_DKV = self.__tf_N_tensor(self.C_DIdK, W_DId)
        ## Sample Theta_ and Pi_ from N_DKV_ and priors
        self.Theta = self.__sample_Theta(N_DKV)
        self.Pi = self.__sample_Pi( N_DKV)
        ## Sample C_DNmaxK from Theta, Pi and N_DKV_
        self.C_DIdK = self.__sample_C(self.Theta, self.Pi, W_DId)


    def __init_C_DIdK_(self, W_DId):

        ## Get in shape
        K = self.data_specs["N_topics"]
        D = self.data_specs["N_docs"]
        Nmax = self.data_specs["max_doclength"]

        if self.ragged:
            C_Ntotal = tf.constant(
                np.random.randint(0, K, 
                size=int(tf.reduce_sum(W_DId.nested_row_lengths()[0])))
            )
            C_DId = tf.RaggedTensor.from_row_lengths(
                values=C_Ntotal,
                row_lengths=W_DId.nested_row_lengths()[0])
            C_DIdK = tf.one_hot(C_DId, K, axis=-1)

        if not self.ragged:
            C_Ntotal = tf.constant(
                np.random.randint(0, K, 
                size=D*Nmax)
            )
            C_DId = tf.reshape(C_Ntotal, (D, -1))
            C_DIdK = tf.one_hot(C_DId, K, axis=-1)

        self.C_DIdK = C_DIdK


    @tf.function
    def __tf_N_tensor(self, C_DIdK, W_DId):

        ## Get in shape
        K = self.data_specs["N_topics"]
        V = self.data_specs["N_vocab"]
        
        ## Preparing W-stacking by shifting all entries one "up" s. t. v is counted 
        #  from 1 to V+1 instead from 0 to V. This enables to collapse the "&" in the
        #  set to be collapsed to a matrix product
        Wp1 = W_DId + 1
        W_stacked = tf.stack(K*[Wp1], axis=-1)    

        ## Elementwise product combines logical & in condition.
        #  Choosing int32 as product dtype for efficiency.
        C_DIdK_int = tf.cast(C_DIdK, dtype=tf.int32)
        C_Dot_W = tf.math.multiply(W_stacked, C_DIdK_int)

        ## The v-dimension of N is a one-hot encoding for the vocabulary:
        N_DIdKVp1 = tf.one_hot(C_Dot_W, V+1, dtype=tf.int32)

        ## Reverting the v-shift by dropping the 0 one-hot dimension
        N_DIdKV = N_DIdKVp1[:, :, :, 1:]

        ## Summing along v-dimension
        N_DKV = tf.reduce_sum(N_DIdKV, axis=1)

        ## Turn to float for gibbs sampler
        N_DKV = tf.cast(N_DKV, dtype=tf.float32)

        return N_DKV
    

    ## Vectorized C-Sampling (W_DNmax must be padded!)
    @tf.function
    def __sample_C(self, Theta, Pi, W_DId):

        ## Get in shape
        K = self.data_specs["N_topics"]
        V = self.data_specs["N_vocab"]
        Nmax = self.data_specs["max_doclength"]

        # Padding
        if self.ragged:
            W_DNmax = W_DId.to_tensor(0)
            mask = W_DId.to_tensor(V+1) != V+1
        if not self.ragged:
            W_DNmax = W_DId

        ## Numerator
        Theta_DNmaxK = tf.gather(tf.transpose(Theta), W_DNmax) 
        Pi_block  = tf.stack(Nmax * [Pi], axis=1)
        numerator = tf.math.multiply(Pi_block, Theta_DNmaxK)

        ## Sampling
        C_DNmax_dist = tfd.Categorical(probs=numerator)
        C_DNmax      = C_DNmax_dist.sample()
        if self.ragged:
            C_DId = tf.ragged.boolean_mask(C_DNmax, mask)
            C_DId = C_DId.with_row_splits_dtype(tf.int32)
        if not self.ragged:
            C_DId = C_DNmax

        ## One-Hot-Encoding
        return tf.one_hot(C_DId, K, axis=-1)
        

    @tf.function
    def __sample_Theta(self, N_DKV):
        dist_Theta = tfd.Dirichlet(self.theta_prior + tf.reduce_sum(N_DKV, axis=0))
        return dist_Theta.sample()


    @tf.function
    def __sample_Pi(self, N_DKV):
        dist_Pi = tfd.Dirichlet(self.pi_prior + tf.reduce_sum(N_DKV, axis=-1))
        return dist_Pi.sample()
    


# %% Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("Test")