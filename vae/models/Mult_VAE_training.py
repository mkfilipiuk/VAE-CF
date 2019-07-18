import scipy.sparse as sparse
import tensorflow as tf
import numpy as np
import shutil
import time
import os

from tensorflow.python.saved_model.simple_save import simple_save
from tensorflow.python.saved_model import tag_constants

from sklearn.preprocessing import normalize
from collections import defaultdict

from vae.config import LOG
from vae.models.Mult_VAE_model import _VAEGraph, TRAINING, QUERY, VALIDATION

from vae.utils.top_k import get_top_k_sorted
from vae.utils.round import round_8


class VAE:
    def __init__(self,
                 train_data,
                 encoder_dims,
                 decoder_dims=None,
                 batch_size_train=500,
                 batch_size_validation=2000,
                 lam=3e-2,
                 lr=1e-3,
                 total_anneal_steps=200000,
                 anneal_cap=0.2,
                 device='CPU'):

        if not batch_size_train > 0:
            raise Exception("batch_size_train has to be positive")
        if not batch_size_validation > 0:
            raise Exception("batch_size_validation has to be positive")
        if not lam >= 0:
            raise Exception("lam has to be non-negative")
        if not lr > 0:
            raise Exception("lr has to be positive")
        if encoder_dims is None:
            raise Exception("encoder_dims is mandatory")
        if decoder_dims is None:
            decoder_dims = encoder_dims[::-1]
        for i in encoder_dims + decoder_dims + [batch_size_train, batch_size_validation]:
            if i != round_8(i):
                raise Exception("all dims and batch sizes should be divisible by 8")

        self.metrics_history = None
        self.batch_size_train = batch_size_train
        self.batch_size_validation = batch_size_validation
        self.lam = lam
        self.lr = lr
        self.total_anneal_steps = total_anneal_steps
        self.anneal_cap = anneal_cap
        self.device = device
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims
        self._create_dataset(train_data,
                             batch_size_train,
                             encoder_dims)
        self._setup_model()

    def train(
            self,
            n_epochs: int,
            train_data: sparse.csr_matrix,
            validation_data_input: sparse.csr_matrix,
            validation_data_true: sparse.csr_matrix,
            batch_size_train: int,
            batch_size_validation: int,
            metrics: dict,  # Dict[str, matrix -> matrix -> float]
            validation_step: 10,
    ):
        """
        Train the model
        :param n_epochs: number of epochs
        :param train_data:  train matrix of shape users count x items count
        :param metrics: Dictionary of metric names to metric functions
        :param validation_step: If it's set to n then validation is run once every n epochs
        """

        self.metrics_history = defaultdict(lambda: [])
        self.time_elapsed_training_history = []
        self.time_elapsed_validation_history = []

        self.session.run(self.iter.initializer)
        for epoch in range(1, n_epochs + 1):

            self.log_which_epoch(epoch, n_epochs)
            init_time = time.time()

            for _ in range(self.n_batch_per_train):
                self.session.run(self.optimizer)

            training_duration = time.time() - init_time
            self.time_elapsed_training_history.append(training_duration)
            LOG.info("Train time:\t{}".format(training_duration))

            if epoch % validation_step == 0 or epoch == n_epochs:
                init_time = time.time()
                metrics_scores = self.test(validation_data_input,
                                           validation_data_true,
                                           metrics)

                for name, score in metrics_scores.items():
                    self.metrics_history[name].append(score)

                validation_duration = time.time() - init_time
                self.time_elapsed_validation_history.append(validation_duration)
                LOG.info("Valid time:\t{}".format(validation_duration))
                self.log_metrics(epoch, metrics_scores, n_epochs)

        self.log_training_time()

    def test(
            self,
            test_data_input,
            test_data_true,
            metrics,
    ):
        """
        Test the performance of the model
        :param metrics: Dictionary of metric names to metric functions
        """
        metrics_scores = defaultdict(lambda: [])
        gen= self.batch_iterator_val(test_data_input,
                                     test_data_true)
        for idxs, vals, X_true in gen():
            pred_val = self.session.run(
                self.logits_validation, feed_dict={self.inputs_validation: (idxs, vals)})
            pred_val[idxs[:, 0], idxs[:, 1]] = -np.inf

            pred_top_k_sorted = get_top_k_sorted(pred_val, R=100)

            for name, metric in metrics.items():
                metrics_scores[name].append(metric(X_true, pred_top_k_sorted))

        return {name: np.mean(scores) for name, scores in metrics_scores.items()}

    def benchmark(self,
                  n_epochs,
                  warm_up_epochs,
                  train_data,
                  validation_data_input,
                  validation_data_true,
                  batch_size_train,
                  batch_size_validation,
                  metrics,
                  validation_step):
        """
        Benchmark the training
        :param n_epochs: number of epochs
        :param warm_up_epochs: number of starting epochs that are not taken into account in measurements
        :param validation_step: If it's set to n then validation is run once every n epochs
        """

        metrics_history = defaultdict(lambda: [])
        time_train_history = {}
        time_valid_history = {}

        self.session.run(self.iter.initializer)
        init_time_total = time.time()

        for epoch in range(1, n_epochs + 1):

            self.log_which_epoch(epoch, n_epochs)
            init_time_train = time.time()
            for _ in range(self.n_batch_per_train):
                self.session.run(self.optimizer)
            time_train = time.time() - init_time_train
            time_train_history[epoch] = time_train

            if epoch % validation_step == 0 or epoch == n_epochs:
                init_time_valid = time.time()
                metrics_scores = self.test(validation_data_input,
                                           validation_data_true,
                                           metrics)
                time_valid = time.time() - init_time_valid
                time_valid_history[epoch] = time_valid

                for name, score in metrics_scores.items():
                    metrics_history[name].append(score)
                self.log_metrics(epoch, metrics_scores, n_epochs)

        time_total = time.time() - init_time_total

        time_train_truncated_values = [t for e, t in time_train_history.items() if e > warm_up_epochs]
        time_valid_truncated_values = [t for e, t in time_valid_history.items() if
                                       e > warm_up_epochs // validation_step]

        benchmark_result = {
            "time_total": time_total,
            "time_train_total": np.sum(time_train_history.values()),
            "time_valid_total": np.sum(time_valid_history.values()),
            "time_train": {
                "mean": np.mean(time_train_truncated_values),
                "stddev": np.std(time_train_truncated_values),
            },
            "time_valid": {
                "mean": np.mean(time_valid_truncated_values),
                "stddev": np.std(time_valid_truncated_values),
            },
        }

        users_per_second_rates_train = [train_data.shape[0] / t for t in time_train_truncated_values]
        interactions_per_second_rates_train = [train_data.nnz / t for t in time_train_truncated_values]
        users_per_second_rates_valid = [validation_data_input.shape[0] / t for t in time_train_truncated_values]
        interactions_per_second_rates_valid = [validation_data_input.nnz / t for t in time_train_truncated_values]

        benchmark_result["train_per_second"] = {
            "users": {
                "mean": np.mean(users_per_second_rates_train),
                "stddev": np.std(users_per_second_rates_train),
            },
            "interactions": {
                "mean": np.mean(interactions_per_second_rates_train),
                "stddev": np.std(interactions_per_second_rates_train),
            },
        }
        benchmark_result["valid_per_second"] = {
            "users": {
                "mean": np.mean(users_per_second_rates_valid),
                "stddev": np.std(users_per_second_rates_valid),
            },
            "interactions": {
                "mean": np.mean(interactions_per_second_rates_valid),
                "stddev": np.std(interactions_per_second_rates_valid),
            },
        }
        print(benchmark_result)

    def query(self, input_data: np.ndarray):
        return self.session.run(
            self.logits_query,
            feed_dict={self.inputs_query: (np.stack([np.zeros(len(input_data)),
                                                     input_data], axis=1),
                                           np.ones(len(input_data)))})

    def save(self, directory="export_dir"):
        if os.path.isdir(directory):
            print("DIRECTORY {} IS NOT EMPTY. REMOVING IN 5 SEC...".format(directory))
            time.sleep(5)
            shutil.rmtree(directory)
        simple_save(self.session, directory,
                    inputs={"inputs_validation": self.inputs_validation},
                    outputs={"logits_validation": self.logits_validation})

    def load(self, directory="export_dir"):
        tf.saved_model.loader.load(self.session,
                                   [tag_constants.SERVING],
                                   directory)
    def _setup_model(self):
        with tf.device(self.device):
            self._build_graph()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.session = tf.Session(config=config)
            init = tf.global_variables_initializer()
            self.session.run(init)

    def _build_graph(self):
        self.vae = _VAEGraph(self.encoder_dims, self.decoder_dims)

        self.inputs_validation = tf.sparse.placeholder(
            dtype=tf.float32,
            shape=np.array([self.batch_size_validation, self.vae.input_dim], dtype=np.int32))
        self.inputs_query = tf.sparse.placeholder(
            dtype=tf.float32,
            shape=np.array([1, self.vae.input_dim], dtype=np.int32))

        self.logits_validation, self.loss_validation = self._gen_handlers(mode=VALIDATION)
        self.logits_train, self.loss_train, self.optimizer = self._gen_handlers(mode=TRAINING)
        self.logits_query = self._gen_handlers(mode=QUERY)

    def _create_dataset(self, train_data, batch_size_train, encoder_dims):
        generator, self.n_batch_per_train = self.batch_iterator_train(train_data)
        dataset = tf.data.Dataset \
            .from_generator(generator, output_types=(tf.int64, tf.float32)) \
            .map(lambda i, v: tf.SparseTensor(i, v, (batch_size_train, encoder_dims[0]))) \
            .prefetch(10)
        self.iter = dataset.make_initializable_iterator()
        self.inputs_train = self.iter.get_next()

    def _gen_handlers(self, mode):
        # model input
        if mode is TRAINING:
            inputs = self.inputs_train
        elif mode is VALIDATION:
            inputs = self.inputs_validation
        elif mode is QUERY:
            inputs = self.inputs_query
        else:
            assert False

        if mode is TRAINING:
            batch_size = self.batch_size_train
        elif mode is VALIDATION:
            batch_size = self.batch_size_validation
        elif mode is QUERY:
            batch_size = 1
        else:
            assert False

        # model output
        logits, latent_mean, latent_log_var = self.vae(inputs, mode=mode)
        softmax = tf.nn.log_softmax(logits)

        anneal = tf.math.minimum(
            tf.cast(tf.train.get_or_create_global_step(), tf.float32) /
            self.total_anneal_steps, self.anneal_cap)

        # KL divergence
        KL = tf.reduce_mean(
            tf.reduce_sum(
                (-latent_log_var + tf.exp(latent_log_var) + latent_mean ** 2 - 1)
                / 2,
                axis=1))

        # per-user average negative log-likelihood part of loss
        ll_loss = -tf.reduce_sum(tf.gather_nd(softmax, inputs.indices)) / batch_size

        # regularization part of loss
        reg_loss = 2 * tf.reduce_sum(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        loss = ll_loss + self.lam * reg_loss + anneal * KL

        if mode is VALIDATION:

            return logits, ll_loss
        elif mode is QUERY:
            return logits
        elif mode is TRAINING:
            train_op = self._build_optimizer(loss)
            return logits, ll_loss, train_op
    def _build_optimizer(self, loss):
        return tf.train.AdamOptimizer(self.lr).minimize(
            loss, global_step=tf.train.get_or_create_global_step())

    def batch_iterator_train(self, data_input):
        """
        :return: iterator of consecutive batches and its length
        """
        data_input = normalize(data_input)

        indices = np.arange(data_input.shape[0])
        np.random.shuffle(indices)
        data_input = data_input[list(indices)]

        nsize, _ = data_input.shape
        csize = nsize // self.batch_size_train * self.batch_size_train

        def generator():
            while True:
                for st_idx in range(0, csize, self.batch_size_train):
                    idxs, vals = self.next_batch(data_input,st_idx, self.batch_size_train)

                    nnz = vals.shape[0]
                    vals *= (2 * np.random.randint(2, size=nnz))
                    yield (idxs, vals)


        return generator, int(np.ceil(csize / self.batch_size_train))

    def batch_iterator_val(self, data_input, data_true):
        """
        :return: iterator of consecutive batches and its length
        """

        data_input = normalize(data_input)

        nsize, _ = data_input.shape
        csize = nsize // self.batch_size_validation * self.batch_size_validation

        def generator():
            for st_idx in range(0, csize, self.batch_size_validation):
                idxs, vals = self.next_batch(data_input, st_idx, self.batch_size_validation)
                yield idxs, vals, data_true[st_idx:st_idx + self.batch_size_validation]

        return generator

    def next_batch(self, data_input, st_idx, batch_size):
        batch = data_input[st_idx:st_idx + batch_size].copy()
        batch = batch.tocoo()
        idxs = np.stack([batch.row, batch.col], axis=1)
        vals = batch.data
        return idxs,vals

    def log_metrics(self, epoch, metrics_scores, n_epochs):
        for name, score in metrics_scores.items():
            LOG.info("Mean {}:\t{}".format(name, score))

    def log_which_epoch(self, epoch, n_epochs):
        LOG.info("Epoch: {}".format(epoch))

    def log_training_time(self):
        LOG.info("Total elapsed train time: {}".format(np.sum(self.time_elapsed_training_history)))
        LOG.info("Total elapsed valid time: {}".format(np.sum(self.time_elapsed_validation_history)))
        LOG.info("Epoch average train time: {}".format(np.mean(self.time_elapsed_training_history)))
        LOG.info("Epoch average valid time: {}".format(np.mean(self.time_elapsed_validation_history)))


    def close_session(self):
        if self.session is not None:
            self.session.close()
