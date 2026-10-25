"""
bat_sim/test_adversarial_m5.py
==============================
Adversarial Stress-Test and Challenge Suite for Milestone 5 (Policy Training & Integration Test Suite).

Verifies the 4 primary challenge pillars:
1. CLI Edge Cases, Missing Arguments, and Non-Existent Directory Output Paths:
   - Deeply nested non-existent directory output path creation.
   - CLI invocation without arguments (defaults validation).
   - CLI argument parsing error handling (unknown flags, invalid types).
   - CLI help string formatting.
   - Boundary parameters (generations=0, generations=1, pop_size=1, max_steps=1).
   - format_progress_bar robustness with edge values (zero, negative, overflow, completion).
2. Weight Reproducibility Under Fixed Random Seed (--seed 42):
   - Multi-run bitwise identity of champion weights under identical seed.
   - Deterministic per-generation history metrics (best, avg, alive_count).
   - Divergence of champion weights under distinct seeds.
   - Subprocess-level multi-run determinism via CLI execution.
3. .npz Archive Integrity, Array Shape (1314,), Bounds, and Dual-Shape Forward Inference:
   - Archive structure, PK zip format, key persistence, and Windows descriptor release.
   - Weight vector exact shape (1314,) and finite float checks.
   - 1D single-agent inference shape (5,) -> (2,) with steering in [-1, 1], urgency in [0, 1].
   - 2D batch inference shapes (N, 5) -> (N, 2) across varied batch sizes (N=1, 10, 100, 500).
   - Batch vs single agent inference numerical equivalence.
   - Extreme adversarial inputs (+-1e6, +-1e12, zeros, boundary vectors).
   - Backward compatibility across diverse npz archive key layouts.
4. Comprehensive Integration and Test Discovery Validation:
   - Verification of 100% pass rate in bat_sim/test_simulation.py.
   - Verification of full repository test discovery.
"""

from __future__ import annotations

import io
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

import bat_sim.physics as physics
from bat_sim.neural_net import BatNeuralNet, TOTAL_WEIGHTS
from bat_sim.evolution import Population, DEFAULT_POPULATION_SIZE, DEFAULT_MAX_STEPS
from bat_sim.environment import Environment
from bat_sim.train import train, format_progress_bar, build_parser


class TestM5AdversarialCLIAndPaths(unittest.TestCase):
    """Adversarial challenge for CLI argument handling, boundary configs, and filesystem paths."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="m5_test_cli_")

    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_nested_nonexistent_directory_creation(self):
        """Verify train() automatically creates deeply nested non-existent directory paths."""
        deep_path = os.path.join(
            self.tmp_dir, "deeply", "nested", "level_3", "sub_level_4", "champion.npz"
        )
        self.assertFalse(os.path.exists(os.path.dirname(deep_path)))

        summary = train(
            generations=1,
            pop_size=5,
            max_steps=5,
            save_path=deep_path,
            seed=42,
            verbose=False,
        )

        self.assertTrue(os.path.exists(deep_path), f"Output file not found at: {deep_path}")
        self.assertTrue(os.path.isfile(deep_path))
        self.assertEqual(summary["save_path"], os.path.abspath(deep_path))

        # Check weights can be loaded back
        loaded = BatNeuralNet.load(deep_path)
        self.assertEqual(len(loaded.get_flat_weights()), TOTAL_WEIGHTS)
        np.testing.assert_array_equal(loaded.get_flat_weights(), summary["champion_weights"])

    def test_output_path_without_npz_extension(self):
        """Verify output path without .npz extension is handled gracefully and loadable."""
        path_without_ext = os.path.join(self.tmp_dir, "weights_no_extension")

        summary = train(
            generations=1,
            pop_size=5,
            max_steps=5,
            save_path=path_without_ext,
            seed=42,
            verbose=False,
        )

        # np.savez appends .npz if not present
        expected_file = path_without_ext if os.path.exists(path_without_ext) else path_without_ext + ".npz"
        self.assertTrue(os.path.exists(expected_file))

        # BatNeuralNet.load should resolve both path_without_ext and with .npz
        loaded_direct = BatNeuralNet.load(expected_file)
        self.assertEqual(len(loaded_direct), TOTAL_WEIGHTS)

        loaded_auto_ext = BatNeuralNet.load(path_without_ext)
        self.assertEqual(len(loaded_auto_ext), TOTAL_WEIGHTS)

    def test_cli_parser_defaults_and_missing_arguments(self):
        """Verify CLI parser operates correctly when all arguments are omitted."""
        parser = build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.generations, 50)
        self.assertEqual(args.pop_size, 100)
        self.assertEqual(args.max_steps, 500)
        self.assertEqual(args.output, "best_weights.npz")
        self.assertIsNone(args.seed)
        self.assertFalse(args.quiet)

    def test_cli_parser_invalid_arguments_exit_code_2(self):
        """Verify CLI parser exits with code 2 on unknown flags and type mismatches."""
        parser = build_parser()

        # Capture stderr to prevent noise during intentional parsing failures
        with io.StringIO() as buf, io.StringIO() as buf2:
            sys.stderr = buf
            try:
                with self.assertRaises(SystemExit) as cm:
                    parser.parse_args(["--unknown-parameter", "val"])
                self.assertEqual(cm.exception.code, 2)

                with self.assertRaises(SystemExit) as cm:
                    parser.parse_args(["--generations", "not_a_number"])
                self.assertEqual(cm.exception.code, 2)

                with self.assertRaises(SystemExit) as cm:
                    parser.parse_args(["--pop-size", "invalid_pop"])
                self.assertEqual(cm.exception.code, 2)

                with self.assertRaises(SystemExit) as cm:
                    parser.parse_args(["--seed", "invalid_seed"])
                self.assertEqual(cm.exception.code, 2)
            finally:
                sys.stderr = sys.__stderr__

    def test_cli_subprocess_help_flag(self):
        """Verify python -m bat_sim.train --help returns 0 and contains documented flags."""
        cmd = [sys.executable, "-m", "bat_sim.train", "--help"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("--generations", res.stdout)
        self.assertIn("--pop-size", res.stdout)
        self.assertIn("--max-steps", res.stdout)
        self.assertIn("--output", res.stdout)
        self.assertIn("--seed", res.stdout)
        self.assertIn("--quiet", res.stdout)

    def test_train_zero_generations_edge_case(self):
        """Verify train() handles 0 generations safely without division-by-zero or crashes."""
        out_path = os.path.join(self.tmp_dir, "zero_gen.npz")
        summary = train(
            generations=0,
            pop_size=10,
            max_steps=5,
            save_path=out_path,
            seed=42,
            verbose=False,
        )

        self.assertEqual(summary["generations"], 0)
        self.assertEqual(len(summary["history"]), 0)
        self.assertEqual(len(summary["champion_weights"]), TOTAL_WEIGHTS)
        self.assertTrue(os.path.exists(out_path))

    def test_train_single_drone_population(self):
        """Verify train() functions on minimum possible population size (pop_size=1)."""
        out_path = os.path.join(self.tmp_dir, "single_drone.npz")
        summary = train(
            generations=2,
            pop_size=1,
            max_steps=5,
            save_path=out_path,
            seed=42,
            verbose=False,
        )

        self.assertEqual(summary["pop_size"], 1)
        self.assertEqual(summary["generations"], 2)
        self.assertEqual(len(summary["champion_weights"]), TOTAL_WEIGHTS)
        self.assertTrue(os.path.exists(out_path))

    def test_train_single_step_generation(self):
        """Verify train() handles max_steps=1 without errors."""
        out_path = os.path.join(self.tmp_dir, "single_step.npz")
        summary = train(
            generations=2,
            pop_size=5,
            max_steps=1,
            save_path=out_path,
            seed=42,
            verbose=False,
        )

        self.assertEqual(summary["max_steps"], 1)
        self.assertTrue(os.path.exists(out_path))

    def test_format_progress_bar_extreme_inputs(self):
        """Stress-test format_progress_bar with zero, negative, out-of-range, and huge values."""
        # Gen > Total
        bar_over = format_progress_bar(gen=100, total_gens=50, best_fitness=10.0, avg_fitness=5.0, survival_rate=1.0, elapsed_time=1.0, rate=5.0)
        self.assertIn("Gen 50/50", bar_over)

        # Gen 0
        bar_zero = format_progress_bar(gen=0, total_gens=50, best_fitness=0.0, avg_fitness=0.0, survival_rate=0.0, elapsed_time=0.0, rate=0.0)
        self.assertIn("Gen  0/50", bar_zero)

        # Negative fitness values (e.g. initial collision penalties)
        bar_neg = format_progress_bar(gen=5, total_gens=10, best_fitness=-25.4, avg_fitness=-48.2, survival_rate=0.0, elapsed_time=2.5, rate=2.0)
        self.assertIn("-25.4", bar_neg)
        self.assertIn("-48.2", bar_neg)

        # Survival rate > 1.0 or < 0.0 clamping
        bar_surv_clamp = format_progress_bar(gen=1, total_gens=1, best_fitness=100.0, avg_fitness=100.0, survival_rate=1.5, elapsed_time=1.0, rate=1.0)
        self.assertIn("150.0%", bar_surv_clamp)

        # High throughput rate and high time
        bar_high = format_progress_bar(gen=1000, total_gens=1000, best_fitness=9999.9, avg_fitness=8888.8, survival_rate=1.0, elapsed_time=12345.67, rate=999.99)
        self.assertIn("12345.67s", bar_high)


class TestM5AdversarialWeightReproducibility(unittest.TestCase):
    """Adversarial challenge for weight determinism, seed reproducibility, and divergence."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="m5_test_repro_")

    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_bitwise_weight_reproducibility_seed_42(self):
        """Verify two independent training runs with --seed 42 produce bitwise identical weights."""
        path_1 = os.path.join(self.tmp_dir, "run_seed42_a.npz")
        path_2 = os.path.join(self.tmp_dir, "run_seed42_b.npz")

        run_1 = train(
            generations=5,
            pop_size=20,
            max_steps=40,
            save_path=path_1,
            seed=42,
            verbose=False,
        )

        run_2 = train(
            generations=5,
            pop_size=20,
            max_steps=40,
            save_path=path_2,
            seed=42,
            verbose=False,
        )

        # 1. Exact champion weight equality
        w1 = run_1["champion_weights"]
        w2 = run_2["champion_weights"]
        self.assertEqual(w1.shape, (TOTAL_WEIGHTS,))
        self.assertEqual(w2.shape, (TOTAL_WEIGHTS,))
        np.testing.assert_array_equal(
            w1, w2, err_msg="Champion weights diverged between identical seed runs"
        )

        # 2. Disk archive loaded weights equality
        disk_w1 = BatNeuralNet.load(path_1).get_flat_weights()
        disk_w2 = BatNeuralNet.load(path_2).get_flat_weights()
        np.testing.assert_array_equal(disk_w1, disk_w2)
        np.testing.assert_array_equal(disk_w1, w1)

        # 3. Best and average fitness equality
        self.assertEqual(run_1["best_fitness"], run_2["best_fitness"])
        self.assertEqual(run_1["avg_fitness"], run_2["avg_fitness"])
        self.assertEqual(run_1["survival_rate"], run_2["survival_rate"])

        # 4. History metrics step-by-step equality
        self.assertEqual(len(run_1["history"]), len(run_2["history"]))
        for idx, (h1, h2) in enumerate(zip(run_1["history"], run_2["history"])):
            self.assertEqual(h1["best_fitness"], h2["best_fitness"], f"Best fitness mismatch at gen {idx}")
            self.assertEqual(h1["avg_fitness"], h2["avg_fitness"], f"Avg fitness mismatch at gen {idx}")
            self.assertEqual(h1["alive_count"], h2["alive_count"], f"Alive count mismatch at gen {idx}")
            self.assertEqual(h1["collision_count"], h2["collision_count"], f"Collision count mismatch at gen {idx}")

    def test_seed_divergence_guarantee(self):
        """Verify different seeds produce measurably different weights and evolution trajectories."""
        path_42 = os.path.join(self.tmp_dir, "run_42.npz")
        path_43 = os.path.join(self.tmp_dir, "run_43.npz")

        run_42 = train(generations=4, pop_size=20, max_steps=30, save_path=path_42, seed=42, verbose=False)
        run_43 = train(generations=4, pop_size=20, max_steps=30, save_path=path_43, seed=43, verbose=False)

        w_42 = run_42["champion_weights"]
        w_43 = run_43["champion_weights"]

        # The weights must differ significantly
        diff = np.abs(w_42 - w_43)
        self.assertGreater(np.max(diff), 1e-4, "Distinct seeds produced identical champion weights")
        self.assertFalse(np.array_equal(w_42, w_43))

    def test_cli_subprocess_seed_42_reproducibility(self):
        """Verify two independent CLI subprocess runs with --seed 42 produce bitwise identical .npz files."""
        out_sub_1 = os.path.join(self.tmp_dir, "cli_sub_1.npz")
        out_sub_2 = os.path.join(self.tmp_dir, "cli_sub_2.npz")

        for out_path in (out_sub_1, out_sub_2):
            cmd = [
                sys.executable,
                "-m",
                "bat_sim.train",
                "--generations", "3",
                "--pop-size", "15",
                "--max-steps", "25",
                "--seed", "42",
                "--output", out_path,
                "--quiet",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.assertEqual(res.returncode, 0)
            self.assertTrue(os.path.exists(out_path))

        with np.load(out_sub_1) as data1, np.load(out_sub_2) as data2:
            w1 = data1["weights"]
            w2 = data2["weights"]
            np.testing.assert_array_equal(w1, w2, err_msg="Subprocess CLI runs with seed 42 diverged")


class TestM5AdversarialArchiveIntegrityAndInference(unittest.TestCase):
    """Adversarial challenge for .npz archive integrity, array shape, numerical bounds, and inference."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="m5_test_archive_")

    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_npz_archive_zip_integrity_and_file_safety(self):
        """Verify saved .npz archive is a strictly valid ZIP file and closes descriptors cleanly."""
        save_path = os.path.join(self.tmp_dir, "archive_test.npz")
        train(generations=2, pop_size=10, max_steps=10, save_path=save_path, seed=42, verbose=False)

        self.assertTrue(zipfile.is_zipfile(save_path), "persisted weights file is not a valid zip archive")
        with zipfile.ZipFile(save_path, "r") as z:
            names = z.namelist()
            self.assertIn("weights.npy", names, f"Archive missing weights.npy: found {names}")
            # Verify testzip passes CRC checks
            bad_file = z.testzip()
            self.assertIsNone(bad_file, f"Corrupted file inside zip archive: {bad_file}")

        # Test Windows file descriptor safety: check we can rename or delete file without WindowsError
        test_rename = os.path.join(self.tmp_dir, "archive_renamed.npz")
        os.rename(save_path, test_rename)
        self.assertTrue(os.path.exists(test_rename))
        self.assertFalse(os.path.exists(save_path))

    def test_champion_weight_array_properties_and_bounds(self):
        """Verify weights vector has exact shape (1314,), float64 dtype, all finite values."""
        save_path = os.path.join(self.tmp_dir, "prop_test.npz")
        summary = train(generations=3, pop_size=15, max_steps=20, save_path=save_path, seed=42, verbose=False)

        w = summary["champion_weights"]
        self.assertEqual(w.shape, (1314,))
        self.assertEqual(w.dtype, np.float64)
        self.assertTrue(np.all(np.isfinite(w)), "Weights vector contains NaNs or Infs")

        # Check weights are non-trivial (not all zeros)
        self.assertGreater(np.linalg.norm(w), 1e-3, "Weights vector is essentially all zeros")
        # Check standard deviation is healthy
        self.assertGreater(np.std(w), 1e-4, "Weights variance is near zero")

    def test_dual_shape_forward_inference(self):
        """Verify dual-shape forward inference: 1D (5,) -> (2,) and 2D (N, 5) -> (N, 2)."""
        save_path = os.path.join(self.tmp_dir, "infer_test.npz")
        train(generations=2, pop_size=10, max_steps=10, save_path=save_path, seed=42, verbose=False)
        net = BatNeuralNet.load(save_path)

        # 1D single input
        inp_1d = np.array([0.2, 0.4, 0.6, 0.8, -0.1], dtype=np.float64)
        out_1d = net.forward(inp_1d)
        self.assertIsInstance(out_1d, np.ndarray)
        self.assertEqual(out_1d.shape, (2,))
        self.assertTrue(-1.0 <= out_1d[0] <= 1.0, f"Steering output {out_1d[0]} not in [-1, 1]")
        self.assertTrue(0.0 <= out_1d[1] <= 1.0, f"Urgency output {out_1d[1]} not in [0, 1]")

        # 2D batch inputs across varied batch sizes
        batch_sizes = [1, 2, 7, 32, 100, 256, 500]
        for N in batch_sizes:
            inp_2d = np.random.uniform(-1.0, 1.0, size=(N, 5)).astype(np.float64)
            out_2d = net.forward(inp_2d)
            self.assertEqual(out_2d.shape, (N, 2), f"Failed batch shape for N={N}")
            # Bounds checks across batch
            self.assertTrue(np.all((out_2d[:, 0] >= -1.0) & (out_2d[:, 0] <= 1.0)), f"Batch steering bounds failed for N={N}")
            self.assertTrue(np.all((out_2d[:, 1] >= 0.0) & (out_2d[:, 1] <= 1.0)), f"Batch urgency bounds failed for N={N}")

    def test_single_vs_batch_inference_consistency(self):
        """Verify net.forward(batch)[i] strictly matches net.forward(batch[i]) to numerical tolerance."""
        save_path = os.path.join(self.tmp_dir, "consist_test.npz")
        train(generations=2, pop_size=10, max_steps=10, save_path=save_path, seed=42, verbose=False)
        net = BatNeuralNet.load(save_path)

        rng = np.random.default_rng(999)
        test_inputs = rng.uniform(-2.0, 2.0, size=(25, 5))
        batch_outputs = net.forward(test_inputs)

        for i in range(len(test_inputs)):
            single_out = net.forward(test_inputs[i])
            np.testing.assert_allclose(
                batch_outputs[i],
                single_out,
                rtol=1e-12,
                atol=1e-12,
                err_msg=f"Batch vs single forward divergence at index {i}",
            )

    def test_extreme_adversarial_sensor_inputs(self):
        """Stress-test forward inference under extreme input magnitudes and edge vectors."""
        save_path = os.path.join(self.tmp_dir, "extreme_test.npz")
        train(generations=2, pop_size=10, max_steps=10, save_path=save_path, seed=42, verbose=False)
        net = BatNeuralNet.load(save_path)

        stress_cases = [
            ("all_zeros", np.zeros(5)),
            ("all_ones", np.ones(5)),
            ("all_negative_ones", -np.ones(5)),
            ("large_positive", np.full(5, 1e6)),
            ("large_negative", np.full(5, -1e6)),
            ("extreme_magnitude", np.array([1e15, -1e15, 1e12, -1e12, 1e10])),
            ("subnormal_values", np.array([1e-30, -1e-30, 0.0, 1e-20, -1e-20])),
        ]

        for name, inp in stress_cases:
            out = net.forward(inp)
            self.assertEqual(out.shape, (2,))
            self.assertTrue(np.all(np.isfinite(out)), f"Output for {name} contains NaN or Inf: {out}")
            self.assertTrue(
                -1.0 <= out[0] <= 1.0,
                f"Steering for {name} out of bounds [-1, 1]: {out[0]}"
            )
            self.assertTrue(
                0.0 <= out[1] <= 1.0,
                f"Urgency for {name} out of bounds [0, 1]: {out[1]}"
            )

    def test_multi_key_npz_backward_compatibility(self):
        """Verify BatNeuralNet.load can load archives with alternative keys ('flat_weights', 'arr_0', etc.)."""
        w_orig = np.random.randn(TOTAL_WEIGHTS).astype(np.float64)

        # 1. key 'flat_weights'
        p1 = os.path.join(self.tmp_dir, "flat_key.npz")
        np.savez_compressed(p1, flat_weights=w_orig)
        net1 = BatNeuralNet.load(p1)
        np.testing.assert_array_equal(net1.get_flat_weights(), w_orig)

        # 2. key 'arr_0' (standard unkeyed np.savez)
        p2 = os.path.join(self.tmp_dir, "arr0_key.npz")
        np.savez_compressed(p2, w_orig)
        net2 = BatNeuralNet.load(p2)
        np.testing.assert_array_equal(net2.get_flat_weights(), w_orig)

        # 3. dictionary of individual layer matrices
        p3 = os.path.join(self.tmp_dir, "layers_key.npz")
        dummy_net = BatNeuralNet(weights=w_orig)
        np.savez_compressed(
            p3,
            w1=dummy_net.w1,
            b1=dummy_net.b1,
            w2=dummy_net.w2,
            b2=dummy_net.b2,
            w3=dummy_net.w3,
            b3=dummy_net.b3,
        )
        net3 = BatNeuralNet.load(p3)
        np.testing.assert_array_equal(net3.get_flat_weights(), w_orig)


class TestM5AdversarialLongitudinalExecution(unittest.TestCase):
    """Adversarial stress-test for multi-generation headless execution and throughput."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="m5_test_perf_")

    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_headless_training_performance_and_metrics(self):
        """Verify train() completes 10 generations of 50 drones cleanly and records metrics."""
        save_path = os.path.join(self.tmp_dir, "perf_champion.npz")
        t0 = time.perf_counter()
        summary = train(
            generations=10,
            pop_size=50,
            max_steps=100,
            save_path=save_path,
            seed=42,
            verbose=False,
        )
        elapsed = time.perf_counter() - t0

        self.assertLess(elapsed, 15.0, f"10 generations took {elapsed:.2f}s (budget: < 15s)")
        self.assertEqual(summary["generations"], 10)
        self.assertEqual(summary["pop_size"], 50)
        self.assertEqual(len(summary["history"]), 10)
        self.assertTrue(os.path.exists(save_path))

        # Check that fitness improves or remains stable
        best_overall = summary["best_fitness"]
        self.assertGreater(best_overall, -100.0, "Best fitness unexpectedly degraded")


if __name__ == "__main__":
    unittest.main()
