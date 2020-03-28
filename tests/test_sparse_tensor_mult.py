import tensorflow as tf
import numpy as np

test = np.tensordot((delta__ksi_i_mu__k-a/S), (delta__ksi_i_mu__k-a/S), axes=([1], [1]))
test = tf.transpose(test, [0,2,1,3])

mask_tf = tf.convert_to_tensor(mask.toarray(), dtype=tf.bool)
mask_tf = tf.expand_dims(tf.cast(mask_tf, dtype=tf.double), axis=len(mask.shape))
mask_tf = tf.expand_dims(tf.cast(mask_tf, dtype=tf.double), axis=len(mask.shape))

test = mask_tf*test

testSparse = tf.sparse.from_dense(test)


def sparse_dense_matmult_batch(sp_a, b):

    def map_function(x):
        i, dense_slice = x[0], x[1]
        sparse_slice = tf.sparse.reshape(tf.sparse.slice(
            sp_a, [i, 0, 0], [1, sp_a.dense_shape[1], sp_a.dense_shape[2]]),
            [sp_a.dense_shape[1], sp_a.dense_shape[2]])
        mult_slice = tf.sparse.matmul(sparse_slice, dense_slice)
        return mult_slice

    elems = (tf.range(0, sp_a.dense_shape[0], delta=1, dtype=tf.int64), b)
    return tf.map_fn(map_function, elems, dtype=tf.float32, back_prop=True)

test_h = sparse_dense_matmult_batch(testSparse, sig_i_k[:,:S])
