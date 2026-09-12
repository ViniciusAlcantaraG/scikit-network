#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for EdgePush."""
import unittest

import numpy as np
from scipy import sparse

from sknetwork.linalg.edge_push import edge_push, edge_push_pagerank
from sknetwork.linalg.ppr_solver import get_pagerank
from sknetwork.utils.check import is_proba_array


class TestEdgePush(unittest.TestCase):

    def setUp(self):
        self.adjacency = sparse.csr_matrix([
            [0, 3, 0, 1, 0],
            [3, 0, 2, 0, 0],
            [0, 2, 0, 4, 1],
            [1, 0, 4, 0, 2],
            [0, 0, 1, 2, 0]
        ], dtype=float)
        self.damping_factor = 0.85

    def test_error_bound(self):
        eps = 1e-2
        seeds = np.zeros(self.adjacency.shape[0])
        seeds[0] = 1
        expected = get_pagerank(self.adjacency, seeds, self.damping_factor, n_iter=100,
                                tol=1e-12, solver='bicgstab')
        scores = edge_push(self.adjacency, root=0, damping_factor=self.damping_factor, eps=eps)
        self.assertLessEqual(np.linalg.norm(scores - expected, ord=1), eps)
        self.assertLessEqual(scores.sum(), 1)

    def test_input_formats(self):
        scores_sparse = edge_push(self.adjacency, root=0, eps=1e-2)
        scores_dense = edge_push(self.adjacency.toarray(), root=0, eps=1e-2)
        self.assertTrue(np.allclose(scores_sparse, scores_dense))

    def test_canonicalization(self):
        adjacency_duplicate = sparse.csr_matrix(
            (np.array([1., 2., 1., 2.]), np.array([1, 1, 0, 0]), np.array([0, 2, 4])), shape=(2, 2))
        adjacency_canonical = sparse.csr_matrix([[0., 3.], [3., 0.]])
        scores_duplicate = edge_push(adjacency_duplicate, root=0, eps=1e-2)
        scores_canonical = edge_push(adjacency_canonical, root=0, eps=1e-2)
        self.assertTrue(np.allclose(scores_duplicate, scores_canonical))

    def test_isolated_root(self):
        adjacency = sparse.csr_matrix([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
        scores = edge_push(adjacency, root=0)
        self.assertTrue(np.array_equal(scores, np.array([1., 0., 0.])))

    def test_pagerank_integration(self):
        seeds = np.ones(self.adjacency.shape[0]) / self.adjacency.shape[0]
        expected = get_pagerank(self.adjacency, seeds, self.damping_factor, n_iter=100,
                                tol=1e-12, solver='bicgstab')
        approximate = edge_push_pagerank(self.adjacency, seeds, self.damping_factor, tol=1e-2)
        self.assertLessEqual(np.linalg.norm(approximate - expected, ord=1), 1e-2)

        scores = get_pagerank(self.adjacency, seeds, self.damping_factor, n_iter=100,
                              tol=1e-2, solver='edge_push')
        self.assertTrue(is_proba_array(scores))

    def test_invalid_graph(self):
        with self.assertRaises(ValueError):
            edge_push(np.array([[0, 1], [0, 0]]), root=0)
        with self.assertRaises(ValueError):
            edge_push(np.array([[0, -1], [-1, 0]]), root=0)
        with self.assertRaises(ValueError):
            edge_push(np.array([[0, np.inf], [np.inf, 0]]), root=0)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            edge_push(self.adjacency, root=-1)
        with self.assertRaises(TypeError):
            edge_push(self.adjacency, root=0.5)
        with self.assertRaises(ValueError):
            edge_push(self.adjacency, root=0, damping_factor=1)
        with self.assertRaises(ValueError):
            edge_push(self.adjacency, root=0, eps=0)
        with self.assertRaises(ValueError):
            edge_push_pagerank(self.adjacency, np.ones(2))
