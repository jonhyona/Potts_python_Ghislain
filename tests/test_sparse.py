# Not use : non-parallelized
import numpy as np
n = 100
ndims = 4
nnz = 1000
coords = np.random.randint(0, n - 1, size=(ndims, nnz))
data = np.random.random(nnz)

import sparse
x = sparse.COO(coords, data, shape=((n,) * ndims))


y = sparse.tensordot(x, x, axes=((3, 0), (1, 2)))

#y
## <COO: shape=(1000, 1000, 1000, 1000), dtype=float64, nnz=1001588>
#
#z = y.sum(axis=(0, 1, 2))
#z
## <COO: shape=(1000,), dtype=float64, nnz=999>
#
#z.todense()
## array([ 244.0671803 ,  246.38455787,  243.43383158,  256.46068737,
##         261.18598416,  256.36439011,  271.74177584,  238.56059193,
##         ...