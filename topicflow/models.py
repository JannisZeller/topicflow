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
        theta_prior: float=1.0,
        pi_prior:    float=1.0,
        fix_vocab_size: int=None,
        fit_procedure: str="gibbs"):
        """ 
        """

        super(LDA, self).__init__()
        
        self._fit_procedure = fit_procedure

        self.theta_prior_val = theta_prior
        self.pi_prior_val    = pi_prior

        self._N_topics = N_topics
        self._fix_vocab_size = fix_vocab_size


    def fit(self, 
        W_DId: Union[np.ndarray, tf.Tensor, tf.RaggedTensor], 
        N_iter: int=200,
        n_batch: int=None,
        n_epochs: int=None,
        n_per_batch: int=5,
        verbose: int=1):
        """
        """

        ## Data inspection
        self.__inspect_data(W_DId)

        ## Determine if batched 
        self.batched = n_batch != None
        assert (n_batch is None) == (n_epochs is None), "Either none of both of n_batch & n_epochs must be specified."
        assert n_batch <= W_DId.shape[0], "Batch size exeeds number of documents."
        if n_batch <= 250:
            print("Warning: A larger batch size (>250) is recommended.")

        ## Counting Vocab (from fixed length or from Word-counts)
        if self._fix_vocab_size:
            if self._fix_vocab_size < int(tf.reduce_max(W_DId)) + 1:
                raise ValueError("Fixed Vocab is too small, there are more unique tokens in the data.")
            self._vocab_size = self._fix_vocab_size
        else:
            self._vocab_size = int(tf.reduce_max(W_DId)) + 1

        ## Determine if Data is ragged, i. e. varying document lengths
        if self.ragged:
            self.data_specs["max_doclength"] = int(tf.reduce_max(W_DId.nested_row_lengths()[0]))
        if not self.ragged:
            self.data_specs["max_doclength"] = W_DId.shape[1]

        ## Initializing Theta-history
        self.Theta_history = []


        ## Gibbs sampling
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        if self._fit_procedure == "gibbs":

            ## Get in shape
            K = self._N_topics
            V = self._vocab_size
            D = self.data_specs["N_docs"]

            ## "Full" Gibbs sampling
            # - - - - - - - - - - - 
            if n_batch is None:

                ## Setting up priors and initials
                C_DIdK = self.__init_C_DIdK_(W_DId)     
                self.theta_prior = (self.theta_prior_val 
                    * tf.ones(shape=(K, V), dtype=tf.float32))
                self.pi_prior    = (self.pi_prior_val
                    * tf.ones(shape=(D, K), dtype=tf.float32))

                ## Setting up Verbosity
                iterator = tqdm(range(N_iter)) # Default
                if verbose == 0:
                    iterator = range(N_iter)

                ## Sampling Loop
                for _ in iterator:
                    self.Theta, self.Pi, C_DIdK = self._gibbs_sample(W_DId, C_DIdK)
                    self.Theta_history.append(self.Theta)


            ## Batched Gibbs sampling 
            #  (only recommended if OOM Error if not batched)
            # - - - - - - - - - - - -
            if n_batch is not None:

                ## When using patched fitting, the ragged-property must be set
                #  to false because the dataset gets padded first. This is taken
                #  into account in the sampling functions.
                self.ragged = False

                ## Get in shape
                D_batch = n_batch
                D_remainder_batch = D % n_batch

                ## Setting up Theta prior (independend of D-dimension)
                self.theta_prior = (self.theta_prior_val 
                    * tf.ones(shape=(K, V), dtype=tf.float32))

                ## Setting up tf.data.Dataset for the batched fitting. Padding
                #  the data!
                data = (tf.data.Dataset.from_tensor_slices(W_DId)
                            .map(lambda x: x) 
                            # Reshuffle is forbidden because the Pi-values are resampled and need to be matched.
                            .shuffle(D, reshuffle_each_iteration=False) 
                            .cache()
                            .padded_batch(
                                batch_size=n_batch, 
                                padded_shapes=self.data_specs["max_doclength"], 
                                padding_values=self._vocab_size+1, 
                                drop_remainder=False) # The remainder is dropp
                            .prefetch(1))
                
                ## Setting up Verbosity
                iterator = tqdm(range(n_epochs)) # Default
                if verbose == 0:
                    iterator = range(n_epochs)

                ## Epochs Loop
                for i_epoch in iterator:
                    if verbose == 2:
                        print("Epoch", end=" ")
                        print(f"{i_epoch+1}\r", end="", flush=True)
                        print("")

                    Pi_list = []

                    ## Batches Loop
                    for i_batch, W_batch in enumerate(data):
                        if verbose == 2:
                            print("Batch", end=" ")
                            print(f"{i_batch+1}\r", end="", flush=True)
                        

                        ## Initializing C either from prior (random) or from 
                        #  Pi of the last epoch and same batch
                        if i_epoch == 0:
                            C_DIdK_batch = self.__init_C_DIdK_(W_batch)
                        else:
                            C_DIdK_batch = self.__sample_C(self.Theta, self.Pi_list[i_batch], W_batch)


                        ## Reshaping Priors to also work for remainder batch
                        if i_batch == 0:
                            self.pi_prior = (self.pi_prior_val
                                * tf.ones(shape=(D_batch, K), dtype=tf.float32))
                        if i_batch == D // n_batch:
                            ## Setting up priors with "corrected dimensions"   
                            self.pi_prior = (self.pi_prior_val
                                * tf.ones(shape=(D_remainder_batch, K), dtype=tf.float32))


                        ## Per Batch Loop
                        for __ in range(n_per_batch):
                            self.Theta, Pi, C_DIdK_batch = self._gibbs_sample(W_batch, C_DIdK_batch)
                        Pi_list.append(Pi)

                    self.Pi_list = Pi_list # tf.stack(Pi_list, axis=-1)
                    self.Theta_history.append(self.Theta)

                self.Pi = tf.concat(self.Pi_list, axis=0)


    def predict(self, X, procedure: str="gibbs", N_iter: int=10):


        ## Simulation Study showed no improvement for N_iter>20 and already
        #  good values for N_iter~10 (sparsity of 0.1)

        self.batched = False # TODO: Balance here

        ## Inspect data
        self.__inspect_data(X)

        ## Get in shape
        K = self._N_topics
        N_pred = self.data_specs["N_docs"]

        if procedure == "gibbs":
            
            self.pi_prior = (self.pi_prior_val
                                * tf.ones(shape=(N_pred, K), dtype=tf.float32))

            C_DIdK = self.__init_C_DIdK_(X)

            for _ in range(N_iter):
                N_DKV   = self.__tf_N_tensor(C_DIdK, X)
                Pi      = self.__sample_Pi(N_DKV)
                C_DIdK  = self.__sample_C(self.Theta, Pi, X)

            return Pi



    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 



    def _gibbs_sample(self, W_DId, C_DIdK):

        ## Calculate N_DKV_ from W_DNmax and C_DNmaxK
        N_DKV = self.__tf_N_tensor(C_DIdK, W_DId)
        ## Sample Theta_ and Pi_ from N_DKV_ and priors
        Theta = self.__sample_Theta(N_DKV)
        Pi = self.__sample_Pi(N_DKV)
        ## Sample C_DNmaxK from Theta, Pi and N_DKV_
        C_DIdK = self.__sample_C(Theta, Pi, W_DId)

        return Theta, Pi, C_DIdK


    def __init_C_DIdK_(self, W_DId):

        ## Get in shape
        K = self._N_topics
        D = W_DId.shape[0]
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

        return C_DIdK


    @tf.function
    def __tf_N_tensor(self, C_DIdK, W_DId):

        ## Get in shape
        K = self._N_topics
        V = self._vocab_size
        
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
    

    @tf.function
    def __sample_C(self, Theta, Pi, W_DId):

        ## Get in shape
        K = self._N_topics
        V = self._vocab_size
        Nmax = self.data_specs["max_doclength"]

        # Padding
        if self.ragged and not self.batched:
            mask = W_DId.to_tensor(V+1) != V+1
            W_DNmax = W_DId.to_tensor(0)
        if not self.ragged and not self.batched:
            W_DNmax = W_DId
        if self.batched: 
            W_DNmax = W_DId
            mask = W_DId != V+1
        # print(W_DNmax.shape)

        ## Numerator
        Theta_DNmaxK = tf.gather(tf.transpose(Theta), W_DNmax) 
        # print(Theta_DNmaxK.shape)
        Pi_block  = tf.stack(Nmax * [Pi], axis=1)
        # print(Pi_block.shape)
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
    

    def __inspect_data(self, W_DId):
        
        self.data_specs = {}
        self.data_specs["N_docs"] = W_DId.shape[0]

        if any([x==None for x in W_DId.shape]):
            self.ragged = True
            self.data_specs["max_doclength"] = int(tf.reduce_max(W_DId.nested_row_lengths()[0]))
        else:
            self.ragged = False
            self.data_specs["max_doclength"] = int(W_DId.shape[1])


# %% Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("Test")