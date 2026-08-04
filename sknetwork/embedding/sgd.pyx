cimport cython
from libc.stdlib cimport rand

@cython.boundscheck(False)
@cython.wraparound(False)

cpdef sgd(int n_components, int n_epochs, int n, int[:] row, int[:] col, float[:] data, float[:,:] low_dim,
            float a, float b, float lr, int negative_sampling_rate,
            double[:] epochs_per_sample, double[:] epoch_of_next_sample):

        
        for epoch in range(n_epochs):
            Pyssize_t idx
            int i, j
            for idx in range(row.shape[0]):

                if epoch_of_next_sample[idx] > epoch:
                    continue
                
                i = row[idx]
                j = col[idx]
                d = 0.0
                delta = [0.0] * n_components
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