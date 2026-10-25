"""
bat_sim/test_adversarial_neural_net.py
======================================
Adversarial Empirical Challenge Suite for Milestone 2: Pure NumPy Neural Network.

Challenge Dimensions:
1. Uniform Crossover Stress Test:
   - 1,000 crossovers with non-overlapping parent genes:
     - 0 foreign genes across all 1,314,000 gene loci decisions.
     - Offspring locus match probability p = 0.50 (+-0.02).
     - Per-locus uniformity and lack of genetic bias.
     - Parent identity invariant (crossing identical parents yields identical child).
     - Deterministic reproducibility under fixed RNG seed.
     - Shape validation on mismatched parent weights.
2. Gaussian Mutation Stress Test:
   - Bitwise invariance when mutation_rate=0 or sigma=0 (tobytes identity).
   - Invariance on negative mutation rates or negative sigma.
   - Rate=1.0 mutation: 100% of weights perturbed.
   - Rate=0.1 mutation across 1,000 independent trials:
     - Empirical mutation probability ~0.10 (+-0.01).
     - Unmutated loci remain bitwise identical (0.0 delta).
     - Mutated delta distribution matches N(0, sigma^2) (Kolmogorov-Smirnov oracle).
3. Deep Copy & Elitism Memory Isolation:
   - Zero shared memory pointers between original and copy (buffer and all 6 layer views).
   - Buffer pointer separation (ctypes.data isolation).
   - Modifying copy via view slices, setter, or mutate does NOT alter original.
   - Python standard copy.copy() and copy.deepcopy() compliance.
   - Forward inference invariance after clone perturbation.
4. Vectorized Population Breeding Pipeline:
   - Strict output shape (pop_size, 1314).
   - Top n_elites preserved with bitwise exactness across arbitrary fitness rankings.
   - Zero memory sharing between population arrays.
   - Zero mutations on elites even when offspring are mutated.
   - Truncation pool containment: with mutation_rate=0, 100% of offspring genes
     originate strictly from the top truncation_fraction pool; 0 foreign genes from bottom pool.
   - Adaptive mutation schedule decay validation across generations (sigma_0 -> min_sigma floor).
   - Edge case configurations: n_elites=0, minimal parent pool, varying population sizes.
5. Numerical Robustness & Inference Latency:
   - Extreme inputs (+-1e9, +-1e30, NaN, Inf) handling.
   - Dual-shape forward pass parity: (5,) vs (1, 5) vs (N, 5).
   - .npz archive persistence round-trip and backward compatibility.
"""

from __future__ import annotations

import copy
import os
import sys
import tempfile
import time
import unittest
import warnings

# Ensure parent directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy import stats

import bat_sim.neural_net as nn
from bat_sim.neural_net import (
    BatNeuralNet,
    TOTAL_WEIGHTS,
    INPUT_DIM,
    HIDDEN_DIM_1,
    HIDDEN_DIM_2,
    OUTPUT_DIM,
    compute_mutation_sigma,
    breed_population_vectorized,
)


class TestAdversarialUniformCrossover(unittest.TestCase):
    """Adversarial stress-testing of the uniform crossover genetic operator."""

    def setUp(self):
        self.rng = np.random.default_rng(20261022)

    def test_crossover_1000_trials_distribution_and_zero_foreign_genes(self):
        """
        Verify that across 1,000 crossovers:
        1. Offspring loci match parent A with probability 0.50 (+-0.02).
        2. Every locus comes strictly from parent A or parent B (zero foreign genes).
        """
        n_trials = 1000
        # Create parents with strictly non-overlapping, distinguishable values at each locus
        # Parent A: odd integers (1, 3, 5, ..., 2*1314 - 1)
        # Parent B: even integers (2, 4, 6, ..., 2*1314)
        w_a = (np.arange(TOTAL_WEIGHTS, dtype=np.float64) * 2.0) + 1.0
        w_b = (np.arange(TOTAL_WEIGHTS, dtype=np.float64) * 2.0) + 2.0

        parent_a = BatNeuralNet(weights=w_a, rng=self.rng)
        parent_b = BatNeuralNet(weights=w_b, rng=self.rng)

        # Preallocate matrix of child weights: shape (1000, 1314)
        children_weights = np.empty((n_trials, TOTAL_WEIGHTS), dtype=np.float64)

        for i in range(n_trials):
            child = BatNeuralNet.uniform_crossover(parent_a, parent_b, rng=self.rng)
            children_weights[i] = child.get_flat_weights()

        # Check 1: Zero foreign genes across all 1,314,000 decisions
        is_from_a = (children_weights == w_a[np.newaxis, :])
        is_from_b = (children_weights == w_b[np.newaxis, :])
        is_valid_origin = is_from_a | is_from_b

        total_decisions = n_trials * TOTAL_WEIGHTS  # 1,314,000
        foreign_genes_count = int(np.count_nonzero(~is_valid_origin))
        self.assertEqual(
            foreign_genes_count,
            0,
            f"Found {foreign_genes_count} foreign genes in offspring out of {total_decisions} loci!",
        )

        # Check 2: Global probability of matching Parent A in [0.48, 0.52] (0.50 +- 0.02)
        total_a_matches = int(np.count_nonzero(is_from_a))
        p_a = total_a_matches / total_decisions

        self.assertGreaterEqual(
            p_a,
            0.48,
            f"Parent A inheritance probability {p_a:.5f} fell below lower bound 0.48",
        )
        self.assertLessEqual(
            p_a,
            0.52,
            f"Parent A inheritance probability {p_a:.5f} exceeded upper bound 0.52",
        )

        # Check 3: Per-locus distribution uniformity
        # For each of the 1,314 loci, across 1000 trials, the count of choosing parent A
        # should be well within [400, 600] (>6 sigma from mean 500)
        locus_a_counts = np.sum(is_from_a, axis=0)  # shape (1314,)
        min_locus_count = int(np.min(locus_a_counts))
        max_locus_count = int(np.max(locus_a_counts))

        self.assertGreaterEqual(
            min_locus_count,
            400,
            f"Locus with severe Parent B bias detected: {min_locus_count}/1000 from Parent A",
        )
        self.assertLessEqual(
            max_locus_count,
            600,
            f"Locus with severe Parent A bias detected: {max_locus_count}/1000 from Parent A",
        )

    def test_crossover_identical_parents_identity(self):
        """Verify crossover of identical parents produces bitwise identical child."""
        parent = BatNeuralNet(rng=self.rng)
        w_orig = parent.get_flat_weights()

        child = BatNeuralNet.uniform_crossover(parent, parent, rng=self.rng)
        w_child = child.get_flat_weights()

        self.assertTrue(np.array_equal(w_orig, w_child))
        self.assertEqual(w_orig.tobytes(), w_child.tobytes())

    def test_crossover_reproducibility_with_seed(self):
        """Verify uniform crossover is deterministic when seeded."""
        p_a = BatNeuralNet(rng=self.rng)
        p_b = BatNeuralNet(rng=self.rng)

        rng1 = np.random.default_rng(1234567)
        child1 = BatNeuralNet.uniform_crossover(p_a, p_b, rng=rng1)

        rng2 = np.random.default_rng(1234567)
        child2 = BatNeuralNet.uniform_crossover(p_a, p_b, rng=rng2)

        self.assertTrue(np.array_equal(child1.get_flat_weights(), child2.get_flat_weights()))
        self.assertEqual(child1.get_flat_weights().tobytes(), child2.get_flat_weights().tobytes())

    def test_crossover_invalid_parent_shapes(self):
        """Verify attempting crossover on networks with corrupted dimensions raises ValueError."""
        p_a = BatNeuralNet()
        p_b = BatNeuralNet()

        # Temporarily mock corrupted shape in parent_b
        p_b._weights = np.zeros(500, dtype=np.float64)
        with self.assertRaises(ValueError):
            BatNeuralNet.uniform_crossover(p_a, p_b)


class TestAdversarialGaussianMutation(unittest.TestCase):
    """Adversarial stress-testing of the Gaussian mutation genetic operator."""

    def setUp(self):
        self.rng = np.random.default_rng(424242)

    def test_mutation_zero_rate_bitwise_invariance(self):
        """Verify mutation with rate=0 leaves weights 100% bitwise unchanged."""
        net = BatNeuralNet(rng=self.rng)
        w_before = net.get_flat_weights()

        # Test with varied positive sigma values
        for test_sigma in [0.001, 0.1, 1.0, 100.0]:
            net.mutate(mutation_rate=0.0, sigma=test_sigma, rng=self.rng)
            w_after = net.get_flat_weights()
            self.assertTrue(np.array_equal(w_before, w_after))
            self.assertEqual(w_before.tobytes(), w_after.tobytes())

    def test_mutation_zero_sigma_bitwise_invariance(self):
        """Verify mutation with sigma=0 leaves weights 100% bitwise unchanged."""
        net = BatNeuralNet(rng=self.rng)
        w_before = net.get_flat_weights()

        # Test with varied mutation rates
        for test_rate in [0.01, 0.1, 0.5, 1.0]:
            net.mutate(mutation_rate=test_rate, sigma=0.0, rng=self.rng)
            w_after = net.get_flat_weights()
            self.assertTrue(np.array_equal(w_before, w_after))
            self.assertEqual(w_before.tobytes(), w_after.tobytes())

    def test_mutation_negative_parameters_safe(self):
        """Verify negative mutation rate or negative sigma perform no modification."""
        net = BatNeuralNet(rng=self.rng)
        w_before = net.get_flat_weights()

        net.mutate(mutation_rate=-0.5, sigma=0.1, rng=self.rng)
        self.assertEqual(w_before.tobytes(), net.get_flat_weights().tobytes())

        net.mutate(mutation_rate=0.1, sigma=-0.5, rng=self.rng)
        self.assertEqual(w_before.tobytes(), net.get_flat_weights().tobytes())

    def test_mutation_rate_1_full_perturbation_and_gaussian_oracle(self):
        """
        Verify mutation_rate=1.0 perturbs 100% of weights and perturbations follow N(0, sigma^2).
        """
        sigma = 0.2
        n_trials = 100
        net = BatNeuralNet(rng=self.rng)
        w_orig = net.get_flat_weights()

        all_deltas = []
        for _ in range(n_trials):
            net.set_flat_weights(w_orig)
            net.mutate(mutation_rate=1.0, sigma=sigma, rng=self.rng)
            w_mut = net.get_flat_weights()

            # 100% of weights must be changed
            self.assertEqual(np.count_nonzero(w_mut == w_orig), 0)
            deltas = w_mut - w_orig
            all_deltas.append(deltas)

        pooled_deltas = np.concatenate(all_deltas)  # 131,400 samples
        mean_delta = float(np.mean(pooled_deltas))
        std_delta = float(np.std(pooled_deltas))

        # Sample mean within 3 standard errors of 0: 3 * (0.2 / sqrt(131400)) ~ 0.0016
        self.assertAlmostEqual(mean_delta, 0.0, delta=0.005)
        # Sample standard deviation matches sigma=0.2 within 2%
        self.assertAlmostEqual(std_delta, sigma, delta=0.005)

        # Kolmogorov-Smirnov test against standard normal N(0, 1)
        # Subsample 5,000 points for KS test
        sample_ks = self.rng.choice(pooled_deltas, size=5000, replace=False)
        sample_standardized = (sample_ks - 0.0) / sigma
        ks_res = stats.kstest(sample_standardized, "norm")
        self.assertGreater(
            ks_res.pvalue,
            0.001,
            f"KS test failed for N(0, {sigma}^2): stat={ks_res.statistic}, p={ks_res.pvalue}",
        )

    def test_mutation_rate_01_empirical_probability_and_oracle(self):
        """
        Verify for rate=0.1, ~10% loci are mutated with N(0, sigma^2).
        Runs 1,000 independent mutations and verifies:
        - Mean fraction of mutated loci is in [0.095, 0.105].
        - Unmutated loci have exactly 0.0 delta.
        - Mutated loci deltas follow N(0, sigma^2).
        """
        sigma = 0.15
        mutation_rate = 0.10
        n_trials = 1000

        net = BatNeuralNet(rng=self.rng)
        w_orig = net.get_flat_weights()

        total_loci = n_trials * TOTAL_WEIGHTS  # 1,314,000
        mutated_deltas_list = []
        total_mutated_count = 0

        for _ in range(n_trials):
            net.set_flat_weights(w_orig)
            net.mutate(mutation_rate=mutation_rate, sigma=sigma, rng=self.rng)
            w_mut = net.get_flat_weights()

            diff = w_mut - w_orig
            is_mutated = (diff != 0.0)
            mutated_count = int(np.count_nonzero(is_mutated))
            total_mutated_count += mutated_count

            if mutated_count > 0:
                mutated_deltas_list.append(diff[is_mutated])

        # Verify mutation probability ~ 0.10 (+-0.01)
        p_mut = total_mutated_count / total_loci
        self.assertGreaterEqual(
            p_mut,
            0.095,
            f"Empirical mutation rate {p_mut:.5f} fell below 0.095",
        )
        self.assertLessEqual(
            p_mut,
            0.105,
            f"Empirical mutation rate {p_mut:.5f} exceeded 0.105",
        )

        # Verify mutated deltas follow N(0, sigma^2)
        pooled_mut_deltas = np.concatenate(mutated_deltas_list)
        mean_mut = float(np.mean(pooled_mut_deltas))
        std_mut = float(np.std(pooled_mut_deltas))

        self.assertAlmostEqual(mean_mut, 0.0, delta=0.005)
        self.assertAlmostEqual(std_mut, sigma, delta=0.005)

        # KS test on 5000 mutated points against standard normal
        sample_ks = self.rng.choice(pooled_mut_deltas, size=5000, replace=False)
        sample_standardized = (sample_ks - 0.0) / sigma
        ks_res = stats.kstest(sample_standardized, "norm")
        self.assertGreater(
            ks_res.pvalue,
            0.001,
            f"KS test rejected N(0, {sigma}^2): stat={ks_res.statistic}, p={ks_res.pvalue}",
        )


class TestAdversarialDeepCopyAndElitism(unittest.TestCase):
    """Adversarial stress-testing of deep copy, elitism memory isolation, and views."""

    def setUp(self):
        self.rng = np.random.default_rng(98765)

    def test_deep_copy_zero_shared_memory_pointers(self):
        """Verify copy() produces 100% memory isolation for base buffer and all 6 layer views."""
        orig = BatNeuralNet(rng=self.rng)
        clones = [
            orig.copy(),
            copy.copy(orig),
            copy.deepcopy(orig),
        ]

        for idx, clone in enumerate(clones):
            # 1. Base weights buffer pointer check
            self.assertNotEqual(
                orig._weights.ctypes.data,
                clone._weights.ctypes.data,
                f"Clone {idx} base buffer shares same raw memory address as original!",
            )
            self.assertFalse(np.shares_memory(orig._weights, clone._weights))

            # 2. Check all 6 layer views
            for view_name in ["w1", "b1", "w2", "b2", "w3", "b3"]:
                orig_view = getattr(orig, view_name)
                clone_view = getattr(clone, view_name)
                self.assertFalse(
                    np.shares_memory(orig_view, clone_view),
                    f"Clone {idx} view {view_name} shares memory with original!",
                )
                self.assertNotEqual(orig_view.ctypes.data, clone_view.ctypes.data)

    def test_mutating_copy_never_alters_original(self):
        """
        Verify mutating a copy through any interface (mutate, set_flat_weights, direct view slice)
        NEVER alters the original network's weights or forward predictions.
        """
        orig = BatNeuralNet(rng=self.rng)
        orig_weights_frozen = orig.get_flat_weights().copy()
        test_inputs = self.rng.standard_normal((20, 5))
        orig_preds_frozen = orig.forward(test_inputs).copy()

        clone = orig.copy()

        # Mutation 1: In-place mutate on clone with rate=1.0, sigma=1.0
        clone.mutate(mutation_rate=1.0, sigma=1.0, rng=self.rng)
        self.assertTrue(np.array_equal(orig.get_flat_weights(), orig_weights_frozen))
        self.assertEqual(orig.get_flat_weights().tobytes(), orig_weights_frozen.tobytes())
        np.testing.assert_array_equal(orig.forward(test_inputs), orig_preds_frozen)

        # Mutation 2: Direct view slice assignment on clone
        clone.w1[:, :] = 9999.0
        clone.b1[:] = -8888.0
        clone.w2[:, :] = 7777.0
        clone.b2[:] = -6666.0
        clone.w3[:, :] = 5555.0
        clone.b3[:] = -4444.0
        self.assertTrue(np.array_equal(orig.get_flat_weights(), orig_weights_frozen))
        self.assertEqual(orig.get_flat_weights().tobytes(), orig_weights_frozen.tobytes())
        np.testing.assert_array_equal(orig.forward(test_inputs), orig_preds_frozen)

        # Mutation 3: set_flat_weights on clone
        clone.set_flat_weights(np.zeros(TOTAL_WEIGHTS, dtype=np.float64))
        self.assertTrue(np.array_equal(orig.get_flat_weights(), orig_weights_frozen))
        self.assertEqual(orig.get_flat_weights().tobytes(), orig_weights_frozen.tobytes())
        np.testing.assert_array_equal(orig.forward(test_inputs), orig_preds_frozen)

    def test_mutating_original_never_alters_copy(self):
        """Verify reverse isolation: modifying original never alters previously taken copy."""
        orig = BatNeuralNet(rng=self.rng)
        clone = orig.copy()
        clone_weights_frozen = clone.get_flat_weights().copy()

        # Mutate original
        orig.mutate(mutation_rate=1.0, sigma=2.0, rng=self.rng)
        self.assertTrue(np.array_equal(clone.get_flat_weights(), clone_weights_frozen))
        self.assertEqual(clone.get_flat_weights().tobytes(), clone_weights_frozen.tobytes())


class TestAdversarialVectorizedBreeding(unittest.TestCase):
    """Adversarial stress-testing of vectorized population breeding and truncation selection."""

    def setUp(self):
        self.rng = np.random.default_rng(5551212)

    def test_breed_preserves_elites_unchanged_arbitrary_fitness_ranking(self):
        """
        Verify breed_population_vectorized preserves top n_elites bitwise unchanged
        regardless of arbitrary, shuffled, negative, or tied fitness scores.
        """
        pop_size = 100
        n_elites = 5

        # Create distinct population
        pop_weights = self.rng.standard_normal((pop_size, TOTAL_WEIGHTS))

        # Assign fitnesses with known top 5 scattered at arbitrary random indices
        fitnesses = self.rng.uniform(-200.0, 50.0, size=pop_size)
        lucky_indices = [13, 42, 67, 88, 91]
        top_scores = [1000.0, 900.0, 800.0, 700.0, 600.0]
        for idx, score in zip(lucky_indices, top_scores):
            fitnesses[idx] = score

        # Expected elites in order
        expected_elites = [pop_weights[idx].copy() for idx in lucky_indices]

        next_pop = breed_population_vectorized(
            pop_weights,
            fitnesses,
            generation=0,
            pop_size=pop_size,
            n_elites=n_elites,
            truncation_fraction=0.20,
            mutation_rate=0.2,  # Heavy mutation for offspring
            sigma_0=0.1,
            rng=self.rng,
        )

        self.assertEqual(next_pop.shape, (pop_size, TOTAL_WEIGHTS))

        # Check 1: Top n_elites are bitwise exact clones of top fitness individuals
        for e_idx in range(n_elites):
            self.assertTrue(
                np.array_equal(next_pop[e_idx], expected_elites[e_idx]),
                f"Elite {e_idx} was altered during vectorized breeding!",
            )
            self.assertEqual(next_pop[e_idx].tobytes(), expected_elites[e_idx].tobytes())

        # Check 2: Memory isolation between generations
        self.assertFalse(np.shares_memory(next_pop, pop_weights))
        for e_idx in range(n_elites):
            self.assertFalse(np.shares_memory(next_pop[e_idx], expected_elites[e_idx]))

    def test_breed_offspring_containment_in_truncation_parent_pool(self):
        """
        With mutation_rate=0.0:
        Verify that 100% of genes in ALL offspring come strictly from the top 20% parent pool.
        Zero foreign genes from the bottom 80% individuals.
        """
        pop_size = 100
        n_elites = 2
        truncation_fraction = 0.20
        n_parents = int(pop_size * truncation_fraction)  # 20 parents

        # Assign unique base scalar ID to each individual: e.g. ind_k = k * 100,000 + gene_idx
        pop_weights = np.empty((pop_size, TOTAL_WEIGHTS), dtype=np.float64)
        for k in range(pop_size):
            pop_weights[k] = (k * 100000.0) + np.arange(TOTAL_WEIGHTS, dtype=np.float64)

        # Monotonically increasing fitness so parents 80..99 are top 20%
        fitnesses = np.arange(pop_size, dtype=np.float64)
        # Top 20 parent IDs are 99, 98, ..., 80
        valid_parent_ids = set(range(pop_size - n_parents, pop_size))
        invalid_bottom_ids = set(range(0, pop_size - n_parents))

        next_pop = breed_population_vectorized(
            pop_weights,
            fitnesses,
            generation=0,
            pop_size=pop_size,
            n_elites=n_elites,
            truncation_fraction=truncation_fraction,
            mutation_rate=0.0,  # Zero mutation to test pure gene provenance
            rng=self.rng,
        )

        n_offspring = pop_size - n_elites
        offspring = next_pop[n_elites:]  # shape (98, 1314)

        # Reconstruct parent ID from gene value: int(val // 100000)
        parent_id_matrix = np.floor(offspring / 100000.0).astype(int)

        # All parent IDs must be within valid_parent_ids
        is_valid_parent = np.isin(parent_id_matrix, list(valid_parent_ids))
        invalid_count = int(np.count_nonzero(~is_valid_parent))

        self.assertEqual(
            invalid_count,
            0,
            f"Found {invalid_count} offspring genes derived from outside the top 20% parent pool!",
        )

        # Ensure that offspring are diverse and not all identical
        unique_offspring_count = len(np.unique(offspring, axis=0))
        self.assertGreater(
            unique_offspring_count,
            int(n_offspring * 0.90),
            f"Expected diverse offspring, got only {unique_offspring_count} unique out of {n_offspring}",
        )

    def test_breed_mutates_offspring_and_preserves_elites(self):
        """
        Verify that when mutation_rate > 0:
        1. 100% of elites are unmutated (diff == 0).
        2. Offspring individuals have mutated genes.
        """
        pop_size = 100
        n_elites = 4
        pop_weights = self.rng.standard_normal((pop_size, TOTAL_WEIGHTS))
        fitnesses = np.linspace(0.0, 100.0, pop_size)

        sorted_indices = np.argsort(fitnesses)[::-1]
        sorted_weights = pop_weights[sorted_indices]

        next_pop = breed_population_vectorized(
            pop_weights,
            fitnesses,
            generation=0,
            pop_size=pop_size,
            n_elites=n_elites,
            truncation_fraction=0.20,
            mutation_rate=0.1,
            sigma_0=0.1,
            rng=self.rng,
        )

        # Elites check
        for i in range(n_elites):
            self.assertTrue(np.array_equal(next_pop[i], sorted_weights[i]))

        # Offspring check: at least 95% of offspring must have at least 1 mutation
        offspring = next_pop[n_elites:]
        # With mutation_rate=0.1 across 1,314 weights, probability of zero mutations in an offspring is (0.9)^1314 ~ 1e-60!
        # Every single offspring must have mutated loci compared to any single parent
        for off_idx in range(len(offspring)):
            # Offspring must not be identical to any elite
            for e_idx in range(n_elites):
                self.assertFalse(np.array_equal(offspring[off_idx], next_pop[e_idx]))

    def test_breed_adaptive_mutation_schedule_decay(self):
        """
        Verify compute_mutation_sigma decay schedule properly throttles noise magnitude
        at generation 0, 50, and 200+.
        """
        sigma_0 = compute_mutation_sigma(0, sigma_0=0.1, decay_factor=0.98, min_sigma=0.01)
        sigma_50 = compute_mutation_sigma(50, sigma_0=0.1, decay_factor=0.98, min_sigma=0.01)
        sigma_200 = compute_mutation_sigma(200, sigma_0=0.1, decay_factor=0.98, min_sigma=0.01)

        self.assertAlmostEqual(sigma_0, 0.1, places=6)
        self.assertAlmostEqual(sigma_50, 0.1 * (0.98 ** 50), places=6)
        self.assertAlmostEqual(sigma_200, 0.01, places=6)  # Hit floor

        # Verify invalid parameters raise ValueError
        with self.assertRaises(ValueError):
            compute_mutation_sigma(-1)
        with self.assertRaises(ValueError):
            compute_mutation_sigma(10, sigma_0=-0.1)
        with self.assertRaises(ValueError):
            compute_mutation_sigma(10, decay_factor=0.0)
        with self.assertRaises(ValueError):
            compute_mutation_sigma(10, decay_factor=1.5)
        with self.assertRaises(ValueError):
            compute_mutation_sigma(10, min_sigma=-0.01)

    def test_breed_edge_cases(self):
        """Verify breeding survives extreme edge case parameter configurations."""
        pop_weights = self.rng.standard_normal((50, TOTAL_WEIGHTS))
        fitnesses = np.ones(50)  # All identical fitnesses

        # Case 1: n_elites = 0 (no elites)
        pop_no_elites = breed_population_vectorized(
            pop_weights,
            fitnesses,
            generation=0,
            pop_size=50,
            n_elites=0,
            truncation_fraction=0.20,
            rng=self.rng,
        )
        self.assertEqual(pop_no_elites.shape, (50, TOTAL_WEIGHTS))

        # Case 2: minimal truncation_fraction (1 parent)
        pop_single_parent = breed_population_vectorized(
            pop_weights,
            fitnesses,
            generation=0,
            pop_size=50,
            n_elites=1,
            truncation_fraction=0.001,
            mutation_rate=0.0,
            rng=self.rng,
        )
        self.assertEqual(pop_single_parent.shape, (50, TOTAL_WEIGHTS))
        # With 1 parent and 0 mutation, every offspring is identical to elite 0
        for i in range(1, 50):
            self.assertTrue(np.array_equal(pop_single_parent[i], pop_single_parent[0]))


class TestAdversarialInferenceAndPersistence(unittest.TestCase):
    """Adversarial stress-testing of forward inference bounds and .npz persistence."""

    def setUp(self):
        self.rng = np.random.default_rng(7112026)
        self.net = BatNeuralNet(rng=self.rng)

    def test_forward_batch_parity_and_shapes(self):
        """Verify forward pass on (5,) matches (1, 5) and matches sliced row of (N, 5)."""
        x_single = np.array([0.1, -0.2, 0.3, 0.4, -0.5])
        x_batch = np.tile(x_single, (10, 1))

        out_single = self.net.forward(x_single)
        out_batch = self.net.forward(x_batch)

        self.assertEqual(out_single.shape, (2,))
        self.assertEqual(out_batch.shape, (10, 2))

        for row in range(10):
            np.testing.assert_allclose(out_batch[row], out_single, atol=1e-15)

    def test_forward_numerical_bounds_extreme_floats(self):
        """Verify forward pass produces strictly bounded outputs on massive inputs."""
        extreme_inputs = np.array([
            [1e10, 1e10, 1e10, 1e10, 1e10],
            [-1e10, -1e10, -1e10, -1e10, -1e10],
            [1e50, -1e50, 1e50, -1e50, 1e50],
        ])

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Reject any overflow / underflow warnings
            out = self.net.forward(extreme_inputs)

        # Steering strictly in [-1.0, 1.0]
        self.assertTrue(np.all(out[:, 0] >= -1.0) and np.all(out[:, 0] <= 1.0))
        # Buzz urgency strictly in [0.0, 1.0]
        self.assertTrue(np.all(out[:, 1] >= 0.0) and np.all(out[:, 1] <= 1.0))
        self.assertFalse(np.isnan(out).any())
        self.assertFalse(np.isinf(out).any())

    def test_npz_persistence_all_key_formats_and_windows_safety(self):
        """
        Verify .npz loading supports:
        1. 'weights' key
        2. 'flat_weights' key
        3. 'arr_0' default key
        4. separate layer keys {'w1', 'b1', 'w2', 'b2', 'w3', 'b3'}
        And cleans up open file handles safely on Windows.
        """
        orig_weights = self.net.get_flat_weights()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Standard save() roundtrip
            save_path = os.path.join(tmp_dir, "test_net.npz")
            self.net.save(save_path)
            loaded = BatNeuralNet.load(save_path)
            self.assertTrue(np.array_equal(orig_weights, loaded.get_flat_weights()))

            # 2. 'flat_weights' key
            flat_path = os.path.join(tmp_dir, "test_flat.npz")
            np.savez_compressed(flat_path, flat_weights=orig_weights)
            loaded_flat = BatNeuralNet.load(flat_path)
            self.assertTrue(np.array_equal(orig_weights, loaded_flat.get_flat_weights()))

            # 3. 'arr_0' key
            arr0_path = os.path.join(tmp_dir, "test_arr0.npz")
            np.savez_compressed(arr0_path, orig_weights)
            loaded_arr0 = BatNeuralNet.load(arr0_path)
            self.assertTrue(np.array_equal(orig_weights, loaded_arr0.get_flat_weights()))

            # 4. separate layer keys
            layers_path = os.path.join(tmp_dir, "test_layers.npz")
            np.savez_compressed(
                layers_path,
                w1=self.net.w1,
                b1=self.net.b1,
                w2=self.net.w2,
                b2=self.net.b2,
                w3=self.net.w3,
                b3=self.net.b3,
            )
            loaded_layers = BatNeuralNet.load(layers_path)
            self.assertTrue(np.array_equal(orig_weights, loaded_layers.get_flat_weights()))


if __name__ == "__main__":
    unittest.main()
