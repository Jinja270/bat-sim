"""
bat_sim/test_adversarial_m2.py
==============================
Adversarial empirical challenge suite for Milestone 2:
Pure NumPy Feedforward Neural Network (bat_sim/neural_net.py).

Verifies:
1. Output bounds across 10,000 random inputs spanning [-1000, 1000] and extreme inputs (+-1e30).
2. Dual-shape inference across shapes (5,), (1, 5), (100, 5), (1000, 5) and error handling.
3. Lossless roundtrip serialization across 1,000 random networks (bitwise and byte equality).
4. Save/load stress: rapid write/read/delete of 100 .npz files without Windows file lock leaks.
5. Genetic operator invariants (crossover gene source, mutation in-place sync, elitism preservation).
"""

from __future__ import annotations

import gc
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from bat_sim.neural_net import (
    BatNeuralNet,
    INPUT_DIM,
    HIDDEN_DIM_1,
    HIDDEN_DIM_2,
    OUTPUT_DIM,
    TOTAL_WEIGHTS,
    W1_SHAPE,
    B1_SHAPE,
    W2_SHAPE,
    B2_SHAPE,
    W3_SHAPE,
    B3_SHAPE,
    compute_mutation_sigma,
    breed_population_vectorized,
)


class TestAdversarialOutputBounds(unittest.TestCase):
    """Task 1: Adversarial output bounds testing across 10,000 random inputs spanning [-1000, 1000]."""

    def test_output_bounds_10k_inputs(self):
        rng = np.random.default_rng(42)
        net = BatNeuralNet(rng=rng)

        # Generate 10,000 random inputs spanning [-1000, 1000]
        inputs = rng.uniform(-1000.0, 1000.0, size=(10000, 5))
        outputs = net.forward(inputs)

        # Assert shape
        self.assertEqual(outputs.shape, (10000, 2))

        # Assert finite
        self.assertTrue(np.all(np.isfinite(outputs)), "Non-finite values encountered in output!")

        # Output 0 (steering rate): strictly in [-1.0, 1.0]
        steering = outputs[:, 0]
        self.assertTrue(
            np.all(steering >= -1.0) and np.all(steering <= 1.0),
            f"Steering violated [-1, 1] bounds! Min={steering.min()}, Max={steering.max()}",
        )

        # Output 1 (buzz urgency): strictly in [0.0, 1.0]
        urgency = outputs[:, 1]
        self.assertTrue(
            np.all(urgency >= 0.0) and np.all(urgency <= 1.0),
            f"Urgency violated [0, 1] bounds! Min={urgency.min()}, Max={urgency.max()}",
        )

    def test_output_bounds_across_diverse_networks(self):
        """Test output bounds across 100 different networks each receiving 100 inputs spanning [-1000, 1000]."""
        rng = np.random.default_rng(123)
        for net_idx in range(100):
            net = BatNeuralNet(rng=rng)
            inputs = rng.uniform(-1000.0, 1000.0, size=(100, 5))
            outputs = net.forward(inputs)

            self.assertTrue(np.all(np.isfinite(outputs)))
            self.assertTrue(np.all(outputs[:, 0] >= -1.0) and np.all(outputs[:, 0] <= 1.0))
            self.assertTrue(np.all(outputs[:, 1] >= 0.0) and np.all(outputs[:, 1] <= 1.0))

    def test_output_bounds_extreme_adversarial_values(self):
        """Test edge cases with extreme input values: +/-1e15, +/-1e30, 0.0, subnormals."""
        rng = np.random.default_rng(999)
        net = BatNeuralNet(rng=rng)

        extreme_scalars = [
            -1e30, -1e15, -1e10, -1000.0, -1.0, -1e-6, -1e-30,
            0.0,
            1e-30, 1e-6, 1.0, 1000.0, 1e10, 1e15, 1e30,
        ]
        # Grid of extreme inputs
        test_inputs = np.array([[s] * 5 for s in extreme_scalars], dtype=np.float64)
        outputs = net.forward(test_inputs)

        self.assertTrue(np.all(np.isfinite(outputs)), "Extreme inputs produced NaN/Inf!")
        self.assertTrue(np.all(outputs[:, 0] >= -1.0) and np.all(outputs[:, 0] <= 1.0))
        self.assertTrue(np.all(outputs[:, 1] >= 0.0) and np.all(outputs[:, 1] <= 1.0))

    def test_output_bounds_with_extreme_weights(self):
        """Test numerical stability when weights are extraordinarily large (+-100.0)."""
        rng = np.random.default_rng(777)
        huge_weights = rng.uniform(-100.0, 100.0, size=TOTAL_WEIGHTS)
        net = BatNeuralNet(weights=huge_weights, rng=rng)

        inputs = rng.uniform(-500.0, 500.0, size=(1000, 5))
        outputs = net.forward(inputs)

        self.assertTrue(np.all(np.isfinite(outputs)), "Huge weights produced NaN or Inf in outputs!")
        self.assertTrue(np.all(outputs[:, 0] >= -1.0) and np.all(outputs[:, 0] <= 1.0))
        self.assertTrue(np.all(outputs[:, 1] >= 0.0) and np.all(outputs[:, 1] <= 1.0))


class TestAdversarialDualShapeInference(unittest.TestCase):
    """Task 2: Dual-shape inference testing across (5,), (1, 5), (100, 5), (1000, 5)."""

    def setUp(self):
        self.rng = np.random.default_rng(31415)
        self.net = BatNeuralNet(rng=self.rng)

    def test_exact_shapes(self):
        # 1. Shape (5,) -> (2,)
        in_1d = self.rng.standard_normal(size=(5,))
        out_1d = self.net.forward(in_1d)
        self.assertEqual(out_1d.shape, (2,))
        self.assertEqual(out_1d.ndim, 1)

        # 2. Shape (1, 5) -> (1, 2)
        in_2d_single = in_1d[None, :]
        out_2d_single = self.net.forward(in_2d_single)
        self.assertEqual(out_2d_single.shape, (1, 2))
        self.assertEqual(out_2d_single.ndim, 2)

        # Numerical parity between (5,) and (1, 5)
        np.testing.assert_array_equal(out_1d, out_2d_single[0])

        # 3. Shape (100, 5) -> (100, 2)
        in_100 = self.rng.standard_normal(size=(100, 5))
        out_100 = self.net.forward(in_100)
        self.assertEqual(out_100.shape, (100, 2))
        self.assertEqual(out_100.ndim, 2)

        # 4. Shape (1000, 5) -> (1000, 2)
        in_1000 = self.rng.standard_normal(size=(1000, 5))
        out_1000 = self.net.forward(in_1000)
        self.assertEqual(out_1000.shape, (1000, 2))
        self.assertEqual(out_1000.ndim, 2)

    def test_callable_syntax_matches_forward(self):
        """__call__ must produce identical output to forward()."""
        x = self.rng.standard_normal(size=(25, 5))
        np.testing.assert_array_equal(self.net(x), self.net.forward(x))

    def test_row_by_row_parity(self):
        """Batch inference on (100, 5) must match 100 individual forward calls on (5,) within float64 precision."""
        batch_in = self.rng.standard_normal(size=(100, 5))
        batch_out = self.net.forward(batch_in)

        for i in range(100):
            single_out = self.net.forward(batch_in[i])
            np.testing.assert_allclose(batch_out[i], single_out, rtol=1e-14, atol=1e-14)

    def test_input_dimension_mismatch_raises(self):
        """Invalid shapes must raise ValueError with descriptive message."""
        invalid_shapes = [
            (4,),
            (6,),
            (1,),
            (0,),
            (10, 4),
            (10, 6),
            (1, 1),
            (2, 5, 2),  # 3D
            (5, 5, 5),  # 3D
        ]
        for shape in invalid_shapes:
            x = np.zeros(shape, dtype=np.float64)
            with self.assertRaises(ValueError, msg=f"Failed to raise ValueError for input shape {shape}"):
                self.net.forward(x)

    def test_list_input_handling(self):
        """Lists and tuples should be converted to array and evaluated properly."""
        x_list = [1.0, 2.0, 3.0, 4.0, 5.0]
        out = self.net.forward(x_list)
        self.assertEqual(out.shape, (2,))

        x_batch_list = [[1.0, 2.0, 3.0, 4.0, 5.0], [0.5, 0.5, 0.5, 0.5, 0.5]]
        out_batch = self.net.forward(x_batch_list)
        self.assertEqual(out_batch.shape, (2, 2))


class TestAdversarialRoundtripSerialization(unittest.TestCase):
    """Task 3: Stress test roundtrip serialization across 1,000 random networks."""

    def test_1000_networks_roundtrip_bitwise_and_bytes(self):
        rng = np.random.default_rng(20261022)
        test_inputs = rng.standard_normal(size=(10, 5))

        for i in range(1000):
            # Generate random flat weights
            random_weights = rng.standard_normal(size=TOTAL_WEIGHTS)
            net1 = BatNeuralNet(weights=random_weights, rng=rng)

            # Extract weights
            exported_weights = net1.get_flat_weights()

            # Verify shape and type
            self.assertEqual(exported_weights.shape, (1314,))
            self.assertEqual(exported_weights.dtype, np.float64)

            # Reconstruct net2
            net2 = BatNeuralNet(weights=exported_weights, rng=rng)

            # 1. Bitwise array equality
            self.assertTrue(
                np.array_equal(exported_weights, net2.get_flat_weights()),
                f"Iteration {i}: get_flat_weights array_equal failed",
            )
            self.assertTrue(
                np.array_equal(net1.w1, net2.w1) and
                np.array_equal(net1.b1, net2.b1) and
                np.array_equal(net1.w2, net2.w2) and
                np.array_equal(net1.b2, net2.b2) and
                np.array_equal(net1.w3, net2.w3) and
                np.array_equal(net1.b3, net2.b3),
                f"Iteration {i}: Internal layer views failed array_equal",
            )

            # 2. Byte-for-byte exact equality
            bytes1 = exported_weights.tobytes()
            bytes2 = net2.get_flat_weights().tobytes()
            self.assertEqual(bytes1, bytes2, f"Iteration {i}: .tobytes() mismatch")

            # 3. Functional equality on forward pass
            out1 = net1.forward(test_inputs)
            out2 = net2.forward(test_inputs)
            np.testing.assert_array_equal(out1, out2)

            # 4. __eq__ operator test
            self.assertEqual(net1, net2)

    def test_set_flat_weights_in_place_view_update(self):
        """Calling set_flat_weights must mutate the internal views in-place without breaking memory buffer."""
        rng = np.random.default_rng(101)
        net = BatNeuralNet(rng=rng)

        w1_original = net.w1.copy()
        new_weights = rng.standard_normal(size=TOTAL_WEIGHTS)
        net.set_flat_weights(new_weights)

        # w1 view must now match the first W1_SIZE elements of new_weights
        expected_w1 = new_weights[:160].reshape(W1_SHAPE)
        np.testing.assert_array_equal(net.w1, expected_w1)
        self.assertFalse(np.array_equal(net.w1, w1_original))

        # Modifying the view directly modifies the flat buffer
        net.w1[0, 0] = 999.999
        self.assertEqual(net._weights[0], 999.999)
        self.assertEqual(net.get_flat_weights()[0], 999.999)

    def test_defensive_copy_isolation(self):
        """get_flat_weights() and .flat_weights property must return independent copies."""
        net = BatNeuralNet()
        exported = net.get_flat_weights()
        exported[0] = -123456.789
        self.assertNotEqual(net._weights[0], -123456.789)

        prop_exported = net.flat_weights
        prop_exported[0] = -987654.321
        self.assertNotEqual(net._weights[0], -987654.321)


class TestAdversarialSaveLoadStress(unittest.TestCase):
    """Task 4: Save/load stress: write/read 100 .npz files rapidly, verify Windows safety."""

    def test_rapid_save_load_100_npz(self):
        rng = np.random.default_rng(54321)
        test_inputs = rng.standard_normal(size=(5, 5))

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_paths = []
            nets = []

            # 1. Rapidly save 100 distinct models
            for i in range(100):
                net = BatNeuralNet(rng=rng)
                nets.append(net)
                fp = os.path.join(tmp_dir, f"model_{i:03d}.npz")
                file_paths.append(fp)
                net.save(fp)
                self.assertTrue(os.path.exists(fp), f"File {fp} was not created!")

            # 2. Rapidly load each and verify bitwise and byte equality
            for i in range(100):
                fp = file_paths[i]
                loaded = BatNeuralNet.load(fp)
                orig = nets[i]

                # Assert bitwise weight equality
                self.assertTrue(np.array_equal(loaded.get_flat_weights(), orig.get_flat_weights()))
                self.assertEqual(
                    loaded.get_flat_weights().tobytes(),
                    orig.get_flat_weights().tobytes(),
                )

                # Assert inference forward parity
                np.testing.assert_array_equal(loaded.forward(test_inputs), orig.forward(test_inputs))

                # Assert loaded instance equality
                self.assertEqual(loaded, orig)

            # 3. Test immediate deletion without Windows PermissionError
            # On Windows, open file descriptors trigger PermissionError during os.remove
            for fp in file_paths:
                try:
                    os.remove(fp)
                except PermissionError as e:
                    self.fail(f"Leaked file handle detected! Windows PermissionError deleting {fp}: {e}")

            # Verify directory is clean
            remaining = [f for f in os.listdir(tmp_dir) if f.endswith(".npz")]
            self.assertEqual(len(remaining), 0, f"Remaining files after deletion: {remaining}")

    def test_save_load_without_extension(self):
        """BatNeuralNet.load should resolve .npz extension automatically if omitted."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = os.path.join(tmp_dir, "my_model")
            orig = BatNeuralNet()
            orig.save(base_path)

            loaded = BatNeuralNet.load(base_path)
            self.assertEqual(orig, loaded)


class TestAdversarialGeneticOperators(unittest.TestCase):
    """Task 5: Empirical challenge of crossover, mutation, and vectorized breeding."""

    def test_crossover_gene_provenance_and_distribution(self):
        """Every gene in child must come from either parent A or parent B."""
        rng = np.random.default_rng(888)
        parent_a = BatNeuralNet(rng=rng)
        parent_b = BatNeuralNet(rng=rng)

        wa = parent_a.get_flat_weights()
        wb = parent_b.get_flat_weights()

        parent_a_counts = []
        for _ in range(100):
            child = BatNeuralNet.uniform_crossover(parent_a, parent_b, rng=rng)
            wc = child.get_flat_weights()

            from_a = (wc == wa)
            from_b = (wc == wb)
            self.assertTrue(
                np.all(np.logical_or(from_a, from_b)),
                "Child has genes that did not originate from either parent!",
            )
            parent_a_counts.append(np.mean(from_a))

        # Mean fraction from parent A should be approximately 0.50 (e.g. 0.45 to 0.55)
        overall_mean = float(np.mean(parent_a_counts))
        self.assertAlmostEqual(overall_mean, 0.50, delta=0.05)

    def test_crossover_immutability(self):
        """Crossover must not alter parents."""
        parent_a = BatNeuralNet()
        parent_b = BatNeuralNet()

        wa_before = parent_a.get_flat_weights()
        wb_before = parent_b.get_flat_weights()

        child = BatNeuralNet.uniform_crossover(parent_a, parent_b)
        child.mutate(mutation_rate=1.0, sigma=5.0)

        np.testing.assert_array_equal(parent_a.get_flat_weights(), wa_before)
        np.testing.assert_array_equal(parent_b.get_flat_weights(), wb_before)

    def test_mutation_zero_rate(self):
        """mutation_rate=0.0 must leave weights identical."""
        net = BatNeuralNet()
        w_before = net.get_flat_weights()
        net.mutate(mutation_rate=0.0, sigma=1.0)
        np.testing.assert_array_equal(net.get_flat_weights(), w_before)

    def test_mutation_rate_1_perturbs_all(self):
        """mutation_rate=1.0 must perturb all 1,314 weights."""
        rng = np.random.default_rng(42)
        net = BatNeuralNet(rng=rng)
        w_before = net.get_flat_weights()
        net.mutate(mutation_rate=1.0, sigma=0.5, rng=rng)
        w_after = net.get_flat_weights()

        self.assertFalse(np.any(w_before == w_after))

    def test_vectorized_breeding_elitism(self):
        """Vectorized breeding must preserve top n_elites bitwise losslessly."""
        rng = np.random.default_rng(2026)
        pop_size = 100
        pop_weights = rng.standard_normal(size=(pop_size, TOTAL_WEIGHTS))
        fitnesses = rng.uniform(-100.0, 100.0, size=pop_size)

        best_idx_1 = int(np.argmax(fitnesses))
        elite_1 = pop_weights[best_idx_1].copy()

        new_pop = breed_population_vectorized(
            pop_weights,
            fitnesses,
            generation=0,
            pop_size=pop_size,
            n_elites=2,
            rng=rng,
        )

        self.assertEqual(new_pop.shape, (pop_size, TOTAL_WEIGHTS))
        # Top elite must be identical
        np.testing.assert_array_equal(new_pop[0], elite_1)

    def test_mutation_sigma_schedule_validation(self):
        """Test compute_mutation_sigma boundaries and parameter validation."""
        # Normal decay behavior
        self.assertAlmostEqual(compute_mutation_sigma(0, sigma_0=0.1), 0.1)
        self.assertAlmostEqual(compute_mutation_sigma(50, sigma_0=0.1, decay_factor=0.98), 0.1 * (0.98 ** 50))
        # Min sigma floor
        self.assertEqual(compute_mutation_sigma(1000, min_sigma=0.01), 0.01)

        # Validation errors
        with self.assertRaises(ValueError):
            compute_mutation_sigma(-1)
        with self.assertRaises(ValueError):
            compute_mutation_sigma(0, sigma_0=0.0)
        with self.assertRaises(ValueError):
            compute_mutation_sigma(0, decay_factor=0.0)
        with self.assertRaises(ValueError):
            compute_mutation_sigma(0, decay_factor=1.5)
        with self.assertRaises(ValueError):
            compute_mutation_sigma(0, min_sigma=-0.01)

    def test_deep_copy_memory_isolation(self):
        """Copying a network must yield completely independent memory buffers."""
        import copy
        net = BatNeuralNet()
        c1 = net.copy()
        c2 = copy.copy(net)
        c3 = copy.deepcopy(net)

        self.assertEqual(net, c1)
        self.assertEqual(net, c2)
        self.assertEqual(net, c3)

        # Mutate c1, ensure net is unaffected
        c1.w1[0, 0] = 777.777
        self.assertNotEqual(net.w1[0, 0], 777.777)
        self.assertNotEqual(c2.w1[0, 0], 777.777)
        self.assertNotEqual(c3.w1[0, 0], 777.777)

    def test_save_load_error_handling(self):
        """Test FileNotFoundError on missing files and KeyError on invalid archives."""
        with self.assertRaises(FileNotFoundError):
            BatNeuralNet.load("non_existent_weights_archive_xyz123.npz")

        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_archive = os.path.join(tmp_dir, "bad.npz")
            np.savez_compressed(bad_archive, random_key=np.zeros(10))
            with self.assertRaises(KeyError):
                BatNeuralNet.load(bad_archive)

    def test_save_auto_creates_directories(self):
        """Saving to a nested non-existent directory must create intermediate directories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            nested_fp = os.path.join(tmp_dir, "sub1", "sub2", "model.npz")
            net = BatNeuralNet()
            net.save(nested_fp)
            self.assertTrue(os.path.exists(nested_fp))
            loaded = BatNeuralNet.load(nested_fp)
            self.assertEqual(net, loaded)

    def test_empty_batch_inference(self):
        """Inference on empty batch (0, 5) must return (0, 2) cleanly."""
        net = BatNeuralNet()
        out = net.forward(np.empty((0, 5)))
        self.assertEqual(out.shape, (0, 2))
        self.assertEqual(out.dtype, np.float64)


if __name__ == "__main__":
    unittest.main()
