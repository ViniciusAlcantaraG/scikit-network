from types import SimpleNamespace
from typing import Union

import numpy as np
from scipy import sparse
from scipy.optimize import curve_fit

from sknetwork.embedding.base import BaseEmbedding
from sknetwork.linalg import normalize
from sknetwork.utils.format import get_adjacency
from sknetwork.ranking import PageRank
from sknetwork.embedding import Spectral
from sknetwork.embedding.sgd import sgd

class UGAP(BaseEmbedding):
    r"""Future documentation
    Describe purpose, steps, and parameters
    """
    def __init__(self, n_components: int = 2, n_neighbors: int = 15, min_dist: float = 0.1, spread: float = 1.0,
                 damping_factor: float = 0.85, ppr_n_iter: int = 10, n_epochs: int = 1000,
                 negative_sampling_rate: int = 5, gamma: float = 1.0, random_state: int = 42, lr: float = 0.8,
                 ppr_solver: str = 'piteration', ppr_tol: float = 1e-6):
        
        super(UGAP, self).__init__()

        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.damping_factor = damping_factor 
        self.ppr_n_iter = ppr_n_iter 
        self.ppr_solver = ppr_solver
        self.ppr_tol = ppr_tol
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.min_dist = min_dist
        self.spread = spread
        self.embedding_ = None
        self.negative_sampling_rate = negative_sampling_rate
        self.gamma = gamma
        self.epochs_per_sample = None
        self.lr = lr

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray]) -> 'UGAP':

        adjacency, _ = get_adjacency(input_matrix)

        # PPR matrix
        n = adjacency.shape[0]
        pagerank = PageRank(damping_factor=self.damping_factor, solver=self.ppr_solver,
                            n_iter=self.ppr_n_iter, tol=self.ppr_tol)
        total_scores = []
        for i in range(n):
            weights_local = np.zeros(n)
            weights_local[i] = 1
            scores = pagerank.fit_predict(adjacency, weights=weights_local)
            total_scores.append(scores)
    
        W = np.array(total_scores)
        np.fill_diagonal(W, 0)

        # top-k PPR neighbours
        rows = []
        cols = []
        vals = []

        eps = 1e-9
        self.n_neighbors = min(self.n_neighbors, n - 1)

        for i in range(n):

            idx = np.argpartition(
                W[i],
                -(self.n_neighbors + 1)
            )[-(self.n_neighbors + 1):]

            idx = idx[W[i, idx] > 0]

            if len(idx) == 0:
                continue

            dists = -np.log(
                W[i, idx] + eps
            )

            sigma_i = np.mean(dists) + eps

            memberships = np.exp(
                -dists / sigma_i
            )

            for j, w in zip(
                idx,
                memberships
            ):
                rows.append(i)
                cols.append(j)
                vals.append(w)

        graph = sparse.coo_matrix(
            (vals, (rows, cols)),
            shape=(n, n)
        )

        prod = graph.multiply(graph.T)

        graph = graph + graph.T - prod

        graph = normalize(graph.tocsr(), p=1)

        # low-dimension
        spectral = Spectral(self.n_components)
        low_dim = spectral.fit_transform(adjacency)

        xv = np.linspace(0, self.spread * 3, 500)
        yv = np.zeros(xv.shape)
        yv[xv < self.min_dist] = 1.0
        yv[xv >= self.min_dist] = np.exp(-(xv[xv >= self.min_dist] - self.min_dist)/self.spread)

        def curve(x, a, b):
            return 1.0 / (1.0 + a * x ** (2 * b))

        params, _ = curve_fit(curve, xv, yv)
        a = params[0] 
        b = params[1]

        # edge sampling rate
        graph = graph.tocoo()
        graph.sum_duplicates()
        graph.eliminate_zeros()

        weights = graph.data
        self.epochs_per_sample = np.full(weights.shape[0], -1.0, dtype=np.float64)
        n_samples = self.n_epochs * (weights / weights.max())
        positive = n_samples > 0
        self.epochs_per_sample[positive] = float(self.n_epochs) / np.float64(n_samples[positive])

        # edge scheduler
        self.epoch_of_next_sample = np.copy(self.epochs_per_sample)

        # SGD
        
        self.embedding_ = sgd(self.n_components, self.n_epochs, n, graph.row, graph.col, 
                            low_dim, a, b, self.lr, self.negative_sampling_rate,
                            self.epochs_per_sample, self.epoch_of_next_sample)

        return self
