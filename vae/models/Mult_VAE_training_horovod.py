import horovod.tensorflow as hvd

import tensorflow as tf
import numpy as np

from sklearn.preprocessing import normalize

from vae.models.Mult_VAE_training import VAE


class VAE_Horovod(VAE):
    def __init__(self, *args, **kwargs):
        hvd.init()
        VAE.__init__(self, *args, **kwargs)

    def _create_dataset(self, train_data, batch_size_train, encoder_dims):
        generator, self.n_batch_per_train = self.batch_iterator(train_data,
                                                                None,
                                                                batch_size_train,
                                                                thread_idx=hvd.rank(),
                                                                thread_num=hvd.size())
        dataset = tf.data.Dataset \
            .from_generator(generator, output_types=(tf.int64, tf.float32)) \
            .map(lambda i, v: tf.SparseTensor(i, v, (batch_size_train, encoder_dims[0]))) \
            .prefetch(10)
        self.iter = dataset.make_initializable_iterator()
        self.inputs_train = self.iter.get_next()

    def _setup_model(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(hvd.local_rank())

        self._build_graph()

        self.session = tf.train.MonitoredTrainingSession(config=config,
                                                         hooks=[hvd.BroadcastGlobalVariablesHook(0)])

    def _build_optimizer(self, loss):
        return hvd.DistributedOptimizer(tf.train.AdamOptimizer(self.lr)).minimize(
            loss, global_step=tf.train.get_or_create_global_step())

    def close_session(self):
        if self.session is not None:
            self.session.close()

    def batch_iterator(self, data_input, data_true=None, batch_size=500, thread_idx=0, thread_num=1):
        # shuffle if training
        training = data_true is None

        data_input = normalize(data_input)

        if training:
            indices = np.arange(data_input.shape[0])
            np.random.shuffle(indices)
            data_in = data_input[list(indices)]
        else:
            data_in = data_input

        # crop
        nsize, isize = data_in.shape
        csize = nsize // batch_size * batch_size

        def generator():
            while True:
                for st_idx in range(thread_idx * batch_size, csize, thread_num * batch_size):
                    batch = data_in[st_idx:st_idx + batch_size].copy()
                    batch = batch.tocoo()
                    idxs = np.stack([batch.row, batch.col], axis=1)
                    vals = batch.data
                    if training:
                        nnz = vals.shape[0]
                        vals *= (2 * np.random.randint(2, size=nnz))
                        yield (idxs, vals)
                    else:
                        yield idxs, vals, data_true[st_idx:st_idx + batch_size]
                if not training:
                    break  # sorry for this

        be = thread_idx * batch_size
        en = csize
        st = thread_num * batch_size
        return generator, int(np.ceil((en - be) / st))
