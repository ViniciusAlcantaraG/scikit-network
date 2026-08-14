#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for UGAP embedding."""

import unittest

import numpy as np

from sknetwork.embedding import Spectral
from sknetwork.embedding.ugap import UGAP
from sknetwork.data import karate_club
from sknetwork.embedding.sgd import sgd
import time
from sknetwork.data import load_netset

class TestUGAP(unittest.TestCase):

    def test_fit(self):
        adjacency = karate_club()
        ugap = UGAP(n_epochs=5, random_state=42)
        ugap.fit(adjacency)
        n = adjacency.shape[0]
        self.assertEqual(ugap.embedding_.shape, (n, ugap.n_components))
        self.assertTrue(np.isfinite(ugap.embedding_).all())

    def test_edge_scheduler(self):
        rng = np.random.default_rng(0)
        n = 8
        n_components, n_epochs = 2, 20
        row = np.array([0, 0, 1, 2, 2, 3], dtype=np.int32)
        col = np.array([1, 2, 3, 3, 0, 1], dtype=np.int32)
        low_dim = rng.normal(size=(n, n_components)).astype(np.float64)
        epochs_per_sample = np.array([1.0, 2.0, 4.0, 5.0, 10.0, 0.5], dtype=np.float64)
        epoch_of_next_sample = epochs_per_sample.copy()
        sgd(n_components, n_epochs, n, row, col, low_dim, 1.0, 1.0, 0.8, 3,
            epochs_per_sample, epoch_of_next_sample)
        self.assertTrue(np.allclose(epoch_of_next_sample, epochs_per_sample * np.round(epoch_of_next_sample / epochs_per_sample)))

    def test_time_small_graph(self):
        graph = load_netset('wikischools')
        adjacency = graph.adjacency
        ugap = UGAP(n_epochs=1000, random_state=42)
        start_time = time.time()
        ugap.fit(adjacency)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time for fitting UGAP on small graph: {elapsed_time:.6f} seconds")

a = TestUGAP()
a.test_time_small_graph()
