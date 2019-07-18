import tensorflow as tf
from tensorflow.keras.layers import Dense


class DenseFromSparse(Dense):
    def call(self, inputs):
        if type(inputs) != tf.sparse.SparseTensor:
            raise ValueError("input should be of type " + str(tf.sparse.SparseTensor))
        rank = len(inputs.get_shape().as_list())
        if rank != 2:
            raise NotImplementedError("input should be rank 2")
        else:
            outputs = tf.sparse.sparse_dense_matmul(inputs, self.kernel)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs
