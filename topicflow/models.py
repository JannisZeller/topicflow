# ------------------------------------------------------------------------------
# topicflow.models
# ----------------
# 
# LDA Model.
# ------------------------------------------------------------------------------


# %% Dependencies
# ------------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


# %% LDA Model
# ------------------------------------------------------------------------------
class LDA(tf.keras.Model):
    def __init__(self, num_classes=1000):
        super(LDA, self).__init__()
        pass

    def call(self, inputs):
        pass




# %% Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("Test")