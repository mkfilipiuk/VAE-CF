import tensorflow as tf
from vae.models.layers.densefromsparse import DenseFromSparse

TRAINING = 0
VALIDATION = 1
QUERY = 2


class _VAEGraph(tf.keras.Model):
    def __init__(self, encoder_dims, decoder_dims):
        super(_VAEGraph, self).__init__()
        if encoder_dims[-1] != decoder_dims[0]:
            raise Exception("encoder/decoder dims mismatch")
        self.input_dim = encoder_dims[0]
        self.output_dim = decoder_dims[-1]
        self.encoder = encoder_model(encoder_dims[1:])
        self.decoder = decoder_model(decoder_dims[1:])

    def call(self, inputs: tf.SparseTensor, mode):
        """ Get handlers to VAE output
        :param inputs: batch_size * items_count as sparse tensor.
        :param mode: Either 0,1 or 2 representing type of network
        :return: Tuple of 3 tensors:
            1. decoder output: batch_size * items_count tensor
            2. latent_mean: mean tensor between encoder and decoder. It has size batch_size * size_of_mean_vector
            3. latent_log_var: tesor containing logarithms of variances. It has size batch_size * size_of_var_vector
        """

        latent_all = self.encoder(inputs, training=(mode is TRAINING))
        latent_mean = latent_all[:, 0]
        latent_log_var = latent_all[:, 1]
        latent_std = tf.exp(0.5 * latent_log_var)

        # reparametrization trick
        batch = tf.shape(latent_mean)[0]
        dim = tf.shape(latent_mean)[1]
        epsilon = tf.random_normal(shape=(batch, dim))
        decoder_input = latent_mean + (int(mode is TRAINING)) * latent_std * epsilon

        decoder_output = self.decoder(decoder_input, training=(mode is TRAINING))

        return decoder_output, latent_mean, latent_log_var


def encoder_model(dims):
    assert dims
    last = dims[-1]
    dims[-1] = 2 * last
    layers = tf.keras.layers
    return tf.keras.Sequential(
        [DenseFromSparse(
                dims[0],
                activation=tf.nn.tanh,
                name="encoder_{}".format(dims[0]),
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.truncated_normal_initializer(stddev=0.001),
                kernel_regularizer=tf.contrib.layers.l2_regularizer)
        ] + [
            layers.Dense(
                d,
                activation=tf.nn.tanh,
                name="encoder_{}".format(d),
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.truncated_normal_initializer(stddev=0.001),
                kernel_regularizer=tf.contrib.layers.l2_regularizer)
            for d in dims[1:-1]
        ] + [
            layers.Dense(
                dims[-1],
                name="encoder_{}".format(dims[-1]),
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.truncated_normal_initializer(stddev=0.001),
                kernel_regularizer=tf.contrib.layers.l2_regularizer)
        ] + [layers.Reshape(target_shape=(2, last))])


def decoder_model(dims):
    assert dims
    layers = tf.keras.layers
    return tf.keras.Sequential([
        layers.Dense(
            d,
            activation=tf.nn.tanh,
            name="decoder_{}".format(d),
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.truncated_normal_initializer(stddev=0.001),
            kernel_regularizer=tf.contrib.layers.l2_regularizer) for d in dims[:-1]
        ] + [
    	    layers.Dense(
            dims[-1],
            name="decoder_{}".format(dims[-1]),
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.truncated_normal_initializer(stddev=0.001),
            kernel_regularizer=tf.contrib.layers.l2_regularizer)
        ])
