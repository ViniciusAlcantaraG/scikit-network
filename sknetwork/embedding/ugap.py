from types import SimpleNamespace
from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.embedding.base import BaseEmbedding
from sknetwork.linalg import LanczosEig, Laplacian, Normalizer, normalize
from sknetwork.utils.check import check_format, check_adjacency_vector, check_nonnegative, check_n_components
from sknetwork.utils.format import get_adjacency
from sknetwork.ranking import PageRank
from sknetwork.gnn.optimizer import ADAM
from sklearn.utils.validation import check_random_state

class UGAP(BaseEmbedding):
    r"""Future documentation
    Describe purpose, steps, and parameters
    """
    def __init__(self, n_components: int = 2, n_neighbors: int = 15, damping_factor: float = 0.85, ppr_n_iter: int = 10, n_epochs: int = 1000,
    random_state: int = 42, theta: float = 0.5, lr: float = 1.0, momentum: float = 0.5, final_momentum: float = 0.8,
    momentum_switch_epoch: int = 250, exaggeration: float = 12.0, exaggeration_epochs: int = 250):
        
        super(UGAP, self).__init__()

        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.damping_factor = damping_factor 
        self.ppr_n_iter = ppr_n_iter 
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.theta = theta
        self.embedding_ = None
        self.lr = lr
        self.momentum = momentum
        self.final_momentum = final_momentum
        self.momentum_switch_epoch = momentum_switch_epoch
        self.exaggeration = exaggeration
        self.exaggeration_epochs = exaggeration_epochs

    def fit(self, input_matrix: Union[sparse.csr_matrix, np.ndarray]) -> 'UGAP':

        rng = check_random_state(self.random_state)

        # --------------------------------------------------
        # Step 1 — Compute PPR matrix
        # --------------------------------------------------
        adjacency, _ = get_adjacency(input_matrix)
        n = adjacency.shape[0]
        pagerank = PageRank(damping_factor=self.damping_factor, n_iter=self.ppr_n_iter)
        total_scores = []
        for i in range(n):
            weights_local = np.zeros(n)
            weights_local[i] = 1
            scores = pagerank.fit_predict(adjacency, weights=weights_local)
            total_scores.append(scores)
    
        W = np.array(total_scores)
        np.fill_diagonal(W, 0)

        rows = []
        cols = []
        vals = []

        eps = 1e-9

        for i in range(n):

            idx = np.argpartition(
                W[i],
                -(self.n_neighbors + 1)
            )[-(self.n_neighbors + 1):]

            idx = idx[W[i, idx] > 0]

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

        graph = normalize(
            graph.tocsr(),
            norm="l1",
            axis=1
        )

        #
        # Convert to t-SNE-style probabilities
        #

        P = graph.toarray()

        P = (P + P.T) / 2

        np.fill_diagonal(P, 0)

        P /= P.sum()

        eps = 1e-12

        _lr = self.lr if self.lr is not None else 1.0

        #
        # Small random init
        #
        Y = rng.normal(
            loc=0.0,
            scale=1e-4,
            size=(n, self.n_components)
        )

        embedding_layer = SimpleNamespace(
            weight=Y,
            bias=np.zeros((1, self.n_components)),
            use_bias=False
        )
        optimizer_state = SimpleNamespace(
            layers=[embedding_layer],
            derivative_weight=[np.zeros_like(Y)],
            derivative_bias=[np.zeros((1, self.n_components))]
        )

        optimizer = ADAM(
            learning_rate=_lr
        )

        for epoch in range(self.n_epochs):

            Y = optimizer_state.layers[0].weight

            #
            # Pairwise squared distances
            #
            diff = Y[:, None, :] - Y[None, :, :]
            dist_sq = np.sum(diff ** 2, axis=2)

            #
            # Student-t kernel
            #
            Q_num = 1.0 / (1.0 + dist_sq)

            np.fill_diagonal(Q_num, 0.0)

            #
            # Global normalization
            #
            Q = Q_num / (Q_num.sum() + eps)

            #
            # KL(P||Q)
            #
            loss = np.sum(
                P * (
                    np.log(P + eps)
                    - np.log(Q + eps)
                )
            )

            grad = 4.0 * np.sum(
                ((P - Q) * Q_num)[:, :, None] * diff,
                axis=1
            )

            optimizer_state.derivative_weight[0] = grad
            optimizer_state.derivative_bias[0] = np.zeros((1, self.n_components))

            optimizer.step(optimizer_state)

            Y = optimizer_state.layers[0].weight

            if epoch % 100 == 0:
                print(epoch, float(loss))

        self.embedding_ = Y
        return self 