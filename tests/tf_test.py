import tensorflow as tf
import numpy as np
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

J = tf.sparse.SparseTensor(indices=[[0, 0,0,0], [1, 2,1,1]], values=[1, 2], dense_shape=[3, 4, 2,2])

#print(J.values)

sig = np.array([[1,2],
                [3,4],
                [5,6],
                [7,8],
                [9,10]], dtype='int32')

tf.sparse.sparse_dense_matmul(J, sig, adjoint_a=False, adjoint_b=False, name=None)
