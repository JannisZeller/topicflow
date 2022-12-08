# ------------------------------------------------------------------------------
# src.sampler
# -----------
# 
# This is the source file containing the custom gibbs sampler I implemented. 
# To speed up computation I use tensorflow tensor-operations for the actual
# calculations. Both samplers (collapsed and standard) work for equal as well 
# as unequal sized documents. The standard Gibbs sampler additionally is 
# capable of processing the data in batches, which should be used if OOM Errors 
# are raised when trying to process the whole dataset at once. 
# ------------------------------------------------------------------------------


# %% Dependencies
# ------------------------------------------------------------------------------
from typing import Union
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions




# %% Sampler Class
# ------------------------------------------------------------------------------
class gibbsSampler():
    def __init__(
        self, 
        K_topics:       int, 
        theta_prior:  float=0.5,
        pi_prior:     float=0.5,
        fix_vocab_size: int=None,
        fit_procedure:  str="standard"):        
        ## Setting internal parameters
        self._fit_procedure  = fit_procedure
        self.theta_prior_val = theta_prior
        self.pi_prior_val    = pi_prior
        self._K_topics       = K_topics
        self._fix_vocab_size = fix_vocab_size


    ## Fit and predict procedures
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    def fit(self, 
        W_DId:       Union[tf.Tensor, tf.RaggedTensor], 
        N_iter:      int=200,
        n_batch:     int=None,
        n_epochs:    int=None,
        n_per_batch: int=5,
        verbose:     int=1) -> None:

        ## Data inspection
        self.__inspect_data_and_settings(W_DId, n_batch, n_epochs)
        
        ## Initializing Theta-history
        self.Theta_history = []

        ## Standard Gibbs sampling 
        if self._fit_procedure == "standard":
            ## "Full" Gibbs sampling
            if n_batch is None:
                self._fit_gibbs(W_DId, N_iter, verbose)
            ## Batched Gibbs sampling 
            #  (only recommended if OOM Error if not batched)
            if n_batch is not None:
                self._fit_gibbs_batched(W_DId, n_batch, n_epochs, n_per_batch, verbose)
        
        ## Collapsed Gibbs sampling
        if self._fit_procedure == "collapsed":
            ## "Full" Collapsed Gibbs sampling
            if n_batch is None:
                self._fit_collapsed_gibbs(W_DId, N_iter, verbose)
            ## Batched Collapsed Gibbs sampling is not implemented
            if n_batch is not None:
                raise NotImplementedError("Batched Processing with collapsed Gibbs sampling is not implemented.")


    ## Predict 
    def predict(self, 
        X: tf.Tensor, 
        N_iter: int=10, 
        pi_prior_val: float=0.5) -> tf.Tensor:
        """Predicting Thetas for new data.

        Parameters
        ----------
        X : tf.Tensor:
            New word-Data.
        N_iter : int=10
            Iterations of resampling for the data.     
        pi_prior_val : float=0.5

        Returns
        -------
        Pi : tf.Tensor
            document-topic prevalences for data X.
        """
        ## Only implemented for processing the complete data at once. Batching
        #  can easily be done outside of this method by slicing the data.
        #  An open todo could be to enable processing of data with "new" tokens
        #  in the vocab by padding or similar. But this is outside of the scope
        #  of this repo.
        self.batched = False
        ## Inspect data
        self.__inspect_data_and_settings(X, None, None)
        ## Get in shape
        K = self._K_topics
        N_pred = self.data_specs["N_docs"]
        ## Prdiction is implemented with the standard Gibbs sampling procedure
        self.pi_prior = (pi_prior_val
            * tf.ones(shape=(N_pred, K), dtype=tf.float32))
        C_DIdK = self.__init_C_DIdK(X)
        if N_iter == 0:
            Pi = self.__sample_Pi(tf.zeros((N_pred, K, self._vocab_size)))
        else:
            for _ in range(N_iter):
                N_DKV   = self.__tf_N_tensor(C_DIdK, X)
                Pi      = self.__sample_Pi(N_DKV)
                C_DIdK  = self.__sample_C(self.Theta, Pi, X)
        return Pi

    ## Fit procedure for standard Gibbs complete processing
    def _fit_gibbs(self, W_DId, N_iter, verbose):
        ## Get in shape
        K = self._K_topics
        V = self._vocab_size
        D = self.data_specs["N_docs"]

        ## Setting up priors and initials
        C_DIdK = self.__init_C_DIdK(W_DId)     
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

    ## Fit procedure for standard Gibbs batched processing
    def _fit_gibbs_batched(self, W_DId, n_batch, n_epochs, n_per_batch, verbose):
        ## When using patched fitting, the ragged-property must be set
        #  to false because the dataset gets padded first. This is taken
        #  into account in the sampling functions.
        self.ragged = False

        ## Get in shape
        K = self._K_topics
        V = self._vocab_size
        D = self.data_specs["N_docs"]
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
                        drop_remainder=False) # The remainder is dropped
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
                    C_DIdK_batch = self.__init_C_DIdK(W_batch)
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
                for _ in range(n_per_batch):
                    self.Theta, Pi, C_DIdK_batch = self._gibbs_sample(W_batch, C_DIdK_batch)
                Pi_list.append(Pi)

            self.Pi_list = Pi_list
            self.Theta_history.append(self.Theta)

        self.Pi = tf.concat(self.Pi_list, axis=0)

    ## Fit procedure  for collapsed Gibbs (loops over words anyways).
    def _fit_collapsed_gibbs(self, W_DId, N_iter, verbose):

        ## Get in shape
        K = self._K_topics
        V = self._vocab_size
        D = self.data_specs["N_docs"]

        ## Infer if ragged
        try:
            W_DId = W_DId.numpy().tolist() # Unragged
        except: 
            W_DId = W_DId.to_list() # Ragged
        N_D = [len(doc) for doc in W_DId]


        ## Setting up priors and initials
        C_DNmax, n_KV, n_DK = self.__collapsed_gibbs_init(W_DId, N_D)
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
            C_DNmax = self._collapsed_gibbs_sample(
                C_DNmax, n_KV, n_DK, N_D, W_DId)
        
        self.__collapsed_gibbs_sampling_posteriors(n_KV, n_DK)



    ## Standard Gibbs Sampling
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    ## Gibbs sampling in one shot, calling the single sampling methods
    def _gibbs_sample(self, W_DId, C_DIdK):
        ## Calculate N_DKV_ from W_DNmax and C_DNmaxK
        N_DKV = self.__tf_N_tensor(C_DIdK, W_DId)
        ## Sample Theta_ and Pi_ from N_DKV_ and priors
        Theta = self.__sample_Theta(N_DKV)
        Pi = self.__sample_Pi(N_DKV)
        ## Sample C_DNmaxK from Theta, Pi and N_DKV_
        C_DIdK = self.__sample_C(Theta, Pi, W_DId)
        return Theta, Pi, C_DIdK

    ## Initialize C_DIdK for the standard Gibbs sampler      
    def __init_C_DIdK(self, W_DId):

        ## Get in shape
        K = self._K_topics
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

    ## Calculate the N-tensor from C and W
    @tf.function
    def __tf_N_tensor(self, C_DIdK, W_DId):

        ## Get in shape
        K = self._K_topics
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
    
    ## Sample C
    @tf.function
    def __sample_C(self, Theta, Pi, W_DId):

        ## Get in shape
        K = self._K_topics
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
      
    ## Sample Theta
    @tf.function
    def __sample_Theta(self, N_DKV):
        dist_Theta = tfd.Dirichlet(self.theta_prior + tf.reduce_sum(N_DKV, axis=0))
        return dist_Theta.sample()

    ## Sample Pi
    @tf.function
    def __sample_Pi(self, N_DKV):
        dist_Pi = tfd.Dirichlet(self.pi_prior + tf.reduce_sum(N_DKV, axis=-1))
        return dist_Pi.sample()




    ## Collapsed Gibbs Sampling
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    ## Collapsed Gibbs sampling (not fully vectorized, has to loop over words).
    # @tf.function
    def _collapsed_gibbs_sample(self, C_DNmax, n_KV, n_DK, N_D, W_DId):   # C_DIdK, W_DId):

        ## Get in shape
        D = self.data_specs["N_docs"]

        C_DNmax_new = np.zeros_like(C_DNmax)

        for d in range(D):
            for i in range(N_D[d]):
                w_di = W_DId[d][i]
                
                ## decrement counters
                c_di = int(C_DNmax[d, i])  # previous assignment
                n_KV[c_di, w_di] -= 1
                n_DK[d, c_di] -= 1

                ## assign new topics
                prob = self.__collapsed_topic_prob(n_KV, n_DK, w_di, d)
                cdi_new = np.argmax(np.random.multinomial(1, prob))

                ## increment counter with new assignment
                n_KV[cdi_new, w_di] += 1
                n_DK[d, cdi_new] += 1
                C_DNmax_new[d, i] = cdi_new

        return C_DNmax_new

    ## Collapsed Gibbs sampler Tensor initializer
    def __collapsed_gibbs_init(self, W_DId, N_D):

        ## Get in shape
        K = self._K_topics
        V = self._vocab_size
        D = self.data_specs["N_docs"]
        Nmax = self.data_specs["max_doclength"]

        C_DNmax = np.zeros((D, Nmax))
        n_KV = np.zeros((K, V))
        n_DK = np.zeros((D, K))

        for d in range(D):
            for i in range(N_D[d]):
                # randomly assign topic to word w_{di}
                w_di = W_DId[d][i]
                C_DNmax[d, i] = np.random.randint(K)

                # increment counters
                c_di = int(C_DNmax[d, i])
                n_KV[c_di, w_di] += 1
                n_DK[d, c_di] += 1

        return C_DNmax, n_KV, n_DK

    ## Word probabilities given topic assignments
    def __collapsed_topic_prob(self, n_KV, n_DK, w_di, d):
        ## Get in shape
        K = self._K_topics
        V = self._vocab_size

        prob  = np.empty(K)
        beta  = self.theta_prior_val
        alpha = self.pi_prior_val
        
        for i in range(K):
            # P(w_dn | z_i)
            p1 = (n_KV[i, w_di] + beta) / (n_KV[i, :].sum() + V*beta)
            # P(z_i | d)
            p2 = (n_DK[d, i] + alpha) / (n_DK[d, :].sum() + K*alpha)
            
            prob[i] = p1 * p2
        
        return prob / prob.sum()

    ## Calculating Posteriors
    def __collapsed_gibbs_sampling_posteriors(self, n_KV, n_DK):

        ## Get in shape
        K = self._K_topics
        V = self._vocab_size
        D = self.data_specs["N_docs"]

        Theta = np.empty((K, V))
        Pi    = np.empty((D, K))
        beta  = self.theta_prior_val
        alpha = self.pi_prior_val

        for v in range(V):
            for k in range(K):
                Theta[k, v] = (n_KV[k, v] + beta) / (n_KV[k, :].sum() + V*beta)

        for d in range(D):
            for k in range(K):
                Pi[d, k] = (n_DK[d, k] + alpha) / (n_DK[d, :].sum() + k*alpha)
            
        self.Theta = Theta
        self.Pi    = Pi



    ## Other functionailites
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    ## Inspecting data at init and predict-calls
    def __inspect_data_and_settings(self, W_DId, n_batch, n_epochs):
        self.data_specs = {}
        self.data_specs["N_docs"] = W_DId.shape[0]

        ## Inferring Ragged-ness
        if isinstance(W_DId, tf.RaggedTensor):
            self.ragged = True
            self.data_specs["max_doclength"] = int(tf.reduce_max(W_DId.nested_row_lengths()[0]))
        else:
            self.ragged = False
            self.data_specs["max_doclength"] = int(W_DId.shape[1])
        
        ## Setting document lengths
        if self.ragged:
            self.data_specs["max_doclength"] = int(
                tf.reduce_max(W_DId.nested_row_lengths()[0]))
        if not self.ragged:
            self.data_specs["max_doclength"] = W_DId.shape[1]

        ## Determine if batched 
        self.batched = n_batch != None
        assert (n_batch is None) == (n_epochs is None), "Either none of both of n_batch & n_epochs must be specified."
        if self.batched:
            assert n_batch <= W_DId.shape[0], "Batch size exeeds number of documents."
        if self.batched and (n_batch <= 250):
            print("Warning: A larger batch size (>250) is recommended.")

        ## Counting Vocab (from fixed length or from actual observations)
        if self._fix_vocab_size:
            if self._fix_vocab_size < int(tf.reduce_max(W_DId)) + 1:
                raise ValueError("Fixed Vocab is too small, there are more unique tokens in the data.")
            self._vocab_size = self._fix_vocab_size
        else:
            self._vocab_size = int(tf.reduce_max(W_DId)) + 1

    ## Dunder-method for printing
    def __repr__(self):
        return f"<gibbsSampler-obj for {self._K_topics}-topics Topic Model>"




# %% Main (Do not acutally run this file.)
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("Why do you run me? Whatever I can construct objects like")
    sampler = gibbsSampler(3)
    print(sampler)