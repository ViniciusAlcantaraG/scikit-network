cimport cython

cpdef sgd(int n_epochs, int n, int[:] row, int[:] col, float[:] data, float[:] low_dim,
            float a, float b, float lr, int negative_sampling_rate,
            int[:] epochs_per_sample, int[:] epoch_of_next_sample):

        for epoch in range(n_epochs):
        for idx, (i, j, w) in enumerate(zip(row, col, data)):

            if epoch_of_next_sample[idx] > epoch:
                continue
                
            d = pow(np.linalg.norm(low_dim[i]-low_dim[j]), 2)
            grad = -2*a*b*pow(d, b-1)/(1+a*pow(d, b)) * (low_dim[i] - low_dim[j])
            low_dim[i] = low_dim[i] - lr * grad          
            low_dim[j] = low_dim[j] + lr * grad          
            epoch_of_next_sample[idx] += epochs_per_sample[idx]

            for _ in range(negative_sampling_rate):
                j = rng.randint(0, n)
                d = pow(np.linalg.norm(low_dim[i] - low_dim[j]), 2)
                grad = 2*a*b*pow(d, b-1)/(1+a*pow(d, b)) * (low_dim[i] - low_dim[j])
                low_dim[i] = low_dim[i] - lr * grad          
                low_dim[j] = low_dim[j] + lr * grad

    return low_dim