import numpy as np
import tensorflow as tf


def setup_deterministic_computation(seed):

    # Batch Normalization layers do not support deterministic computation on GPU
    tf.keras.utils.set_random_seed(seed)
    # tf.config.experimental.enable_op_determinism()
    # tf.random.set_seed(seed)
    # np.random.seed(seed)
