cimport cython
from libc.stdlib cimport rand
from libc.math cimport pow
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)

cpdef sgd(int n_components, int n_epochs, int n, int[:] row, int[:] col, double[:] data, double[:,:] low_dim,
            float a, float b, float lr, int negative_sampling_rate, int[:,:] bucket):


    cdef int epoch, pos, start, end, idx, i, j, k
    cdef double d, temp, coef, grad
    cdef double[::1] delta = np.empty(n_components, dtype=np.float64)

        for epoch in range(n_epochs):
            for pos in range(bucket.shape[1]):
                idx = bucket[epoch, pos]
                if idx < 0:
                    continue  
                
                i = row[idx]
                j = col[idx]
                d = 0.0
                for k in range(n_components):
                    delta[k] = low_dim[i,k] - low_dim[j,k]
                    d += delta[k] * delta[k]

                temp = pow(d + 1e-6, b - 1)
                coef = -2.0*a*b*temp/ (1.0 + a * d * temp)

                for k in range(n_components):
                    grad = coef * delta[k]
                    low_dim[i,k] -= lr * grad
                    low_dim[j,k] += lr * grad

                for _ in range(negative_sampling_rate):
                    j = rand() % n
                    d = 0.0
                    delta = [0.0] * n_components
                    for k in range(n_components):
                        delta[k] = low_dim[i,k] - low_dim[j,k]
                        d += delta[k] * delta[k]

                    temp = pow(d + 1e-6, b - 1)
                    coef = 2*a*b*temp/(1.0 + a * d * temp)

                    for k in range(n_components):
                        grad = coef * delta[k]
                        low_dim[i,k] -= lr * grad
                        low_dim[j,k] += lr * grad

    return low_dim