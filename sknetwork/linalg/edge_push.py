#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Edge-based local push for personalized PageRank."""
import heapq
from typing import Union

import numpy as np
from scipy import sparse

from sknetwork.utils.check import (check_damping_factor, check_format, check_nonnegative, check_square,
                                   check_symmetry, is_proba_array)


def _check_adjacency(adjacency: Union[sparse.csr_matrix, np.ndarray]) -> sparse.csr_matrix:
    """Return a canonical weighted adjacency matrix suitable for EdgePush."""
    adjacency = check_format(adjacency, allow_empty=True).astype(float, copy=True)
    check_square(adjacency)
    if not np.all(np.isfinite(adjacency.data)):
        raise ValueError('Edge weights must be finite.')
    check_nonnegative(adjacency)
    adjacency.sum_duplicates()
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    check_symmetry(adjacency)
    return adjacency


def _get_min_edge(node: int, queues: list, expenses: np.ndarray, thresholds: np.ndarray,
                  edge_weights: np.ndarray):
    """Return the incident edge with minimum current priority."""
    queue = queues[node]
    while queue:
        stored_priority, edge = queue[0]
        current_priority = (expenses[edge] + thresholds[edge]) / edge_weights[edge]
        if stored_priority != current_priority:
            heapq.heappop(queue)
        else:
            return edge, current_priority
    return None


def edge_push_pagerank(adjacency: Union[sparse.csr_matrix, np.ndarray], seeds: np.ndarray,
                       damping_factor: float = 0.85, tol: float = 1e-6) -> np.ndarray:
    r"""Approximate a personalized PageRank vector using EdgePush.

    The input graph must be weighted, undirected and represented by a symmetric
    adjacency matrix. The returned vector is an underestimate of the exact
    Personalized PageRank vector with an :math:`\ell_1` error of at most ``tol``.

    Parameters
    ----------
    adjacency : sparse.csr_matrix, np.ndarray
        Adjacency matrix of the graph.
    seeds : np.ndarray
        Personalization probability vector.
    damping_factor : float
        Probability to continue the random walk.
    tol : float
        Maximum :math:`\ell_1` approximation error. Must be positive.

    Returns
    -------
    scores : np.ndarray
        Approximate Personalized PageRank vector.

    References
    ----------
    Wang, H., Wei, Z., Gan, J., Yuan, Y., Du, X., & Wen, J. R. (2022).
    Edge-based Local Push for Personalized PageRank, PVLDB.
    """
    adjacency = _check_adjacency(adjacency)
    n = adjacency.shape[0]

    number_types = (int, float, np.integer, np.floating)
    if isinstance(damping_factor, (bool, np.bool_)) or not isinstance(damping_factor, number_types) \
            or not np.isfinite(damping_factor):
        raise ValueError('The damping factor must be finite.')
    check_damping_factor(damping_factor)
    if isinstance(tol, (bool, np.bool_)) or not isinstance(tol, number_types) or not np.isfinite(tol) or tol <= 0:
        raise ValueError('The tolerance must be a positive finite number.')

    seeds = np.asarray(seeds, dtype=float)
    if seeds.shape != (n,):
        raise ValueError('The seeds must be a vector with one entry per node.')
    if not np.all(np.isfinite(seeds)) or not is_proba_array(seeds):
        raise ValueError('The seeds must be a finite probability vector.')

    # Each stored entry is one directed orientation of an undirected edge.
    # Consequently, summing over adjacency.data implements the denominator over
    # the bidirectional edge set used in the EdgePush error bound.
    edge_weights = adjacency.data
    if len(edge_weights) == 0:
        return seeds.copy()
    threshold_denominator = np.sqrt(edge_weights).sum()
    thresholds = tol * np.sqrt(edge_weights) / threshold_denominator

    indptr = adjacency.indptr
    indices = adjacency.indices
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    expenses = np.zeros_like(edge_weights)
    income = seeds.copy()

    queues = []
    for node in range(n):
        queue = [(thresholds[edge] / edge_weights[edge], edge)
                 for edge in range(indptr[node], indptr[node + 1])]
        heapq.heapify(queue)
        queues.append(queue)

    node_priority = np.zeros(n)
    non_isolated = degrees > 0
    node_priority[non_isolated] = damping_factor * income[non_isolated] / degrees[non_isolated]

    candidates = []
    active = np.zeros(n, dtype=bool)

    def add_candidate(node: int):
        """Add a node when its minimum-priority edge is eligible."""
        if active[node] or not non_isolated[node]:
            return
        candidate = _get_min_edge(node, queues, expenses, thresholds, edge_weights)
        if candidate is not None and node_priority[node] >= candidate[1]:
            candidates.append(node)
            active[node] = True

    for node in range(n):
        add_candidate(node)

    while candidates:
        node = candidates.pop()
        active[node] = False
        candidate = _get_min_edge(node, queues, expenses, thresholds, edge_weights)
        if candidate is None or node_priority[node] < candidate[1]:
            continue

        edge, _ = candidate
        neighbor = indices[edge]
        edge_weight = edge_weights[edge]
        residual = damping_factor * income[node] * edge_weight / degrees[node] - expenses[edge]

        income[neighbor] += residual
        expenses[edge] += residual
        new_priority = (expenses[edge] + thresholds[edge]) / edge_weight
        heapq.heappush(queues[node], (new_priority, edge))

        if non_isolated[neighbor]:
            node_priority[neighbor] = damping_factor * income[neighbor] / degrees[neighbor]

        add_candidate(node)
        add_candidate(neighbor)

    teleport_probability = 1 - damping_factor
    scores = teleport_probability * income

    # On an undirected graph, isolated nodes form absorbing one-node components.
    scores[~non_isolated] = seeds[~non_isolated]
    return scores


def edge_push(adjacency: Union[sparse.csr_matrix, np.ndarray], root: int, damping_factor: float = 0.85,
              eps: float = 1e-6) -> np.ndarray:
    r"""Approximate single-source Personalized PageRank using EdgePush.

    Parameters
    ----------
    adjacency : sparse.csr_matrix, np.ndarray
        Symmetric adjacency matrix of a weighted undirected graph.
    root : int
        Source node.
    damping_factor : float
        Probability to continue the random walk.
    eps : float
        Maximum :math:`\ell_1` approximation error. Must be positive.

    Returns
    -------
    scores : np.ndarray
        Approximate Personalized PageRank vector.
    """
    adjacency = _check_adjacency(adjacency)
    n = adjacency.shape[0]
    if not isinstance(root, (int, np.integer)):
        raise TypeError('The root must be an integer.')
    if root < 0 or root >= n:
        raise ValueError('The root must be a valid node index.')
    seeds = np.zeros(n)
    seeds[root] = 1
    return edge_push_pagerank(adjacency, seeds, damping_factor, eps)
