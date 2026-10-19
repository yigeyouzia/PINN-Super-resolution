"""Module boundaries, migration checks and failure-path regression tests."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np
import torch

from src.pinn_sr.checkpoint import (
    LEGACY_V1_SOURCE, build_from_checkpoint, read_checkpoint,
    restore_checkpoint, save_checkpoint, validate_resume,
)
from src.pinn_sr.data import PhysicsSampler, load_inputs
from src.pinn_sr.experiment import ROOT, source_hashes, validate_output_root
from src.pinn_sr.pinn import ConfigurablePINN
from tests.fixtures import SyntheticCase


class RefactorTests(SyntheticCase, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def checkpoint(self):
        self.assertEqual(self.run_cli(self.args()), 0)
        return next(self.output.glob("*/checkpoints/step_00000002.pt"))

    def legacy_state(self):
        state = read_checkpoint(self.checkpoint())
        state["config"].pop("method_contract")
        state["config"]["source_sha256"] = dict(LEGACY_V1_SOURCE)
        return state

    def test_model_and_inference_restore_do_not_construct_optimizer(self):
        path = self.checkpoint()
        state = read_checkpoint(path)
        _, _, _, b, _ = load_inputs(self.data, self.phy, "float32")
        with mock.patch("torch.optim.Adam", side_effect=AssertionError("optimizer in inference")):
            model = ConfigurablePINN(self.config(), b["ub"], b["lb"],
                                     b["ub_velocity"], b["lb_velocity"], torch.device("cpu"))
            self.assertFalse(hasattr(model, "adam"))
            restored, _ = restore_checkpoint(path)
            self.assertFalse(restored.net.training)
            # Inference also supports a state without any optimizer or sampler.
            for key in ("optimizer_state", "sampler_state", "torch_rng", "cuda_rng"):
                state.pop(key)
            build_from_checkpoint(state, torch.device("cpu"))

    def test_imports_do_not_load_author_or_reset_rng(self):
        code = """
import random, sys, numpy as np, torch
random.seed(987); np.random.seed(987); torch.manual_seed(987)
a, b, c = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
import src.pinn_sr.cli
assert random.getstate() == a
after = np.random.get_state()
assert after[0] == b[0] and np.array_equal(after[1], b[1]) and after[2:] == b[2:]
assert torch.equal(torch.get_rng_state(), c)
assert not any(name.startswith('_pinn_sr_author') for name in sys.modules)
assert 'network' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_legacy_inference_preserves_predictions_and_state_layout(self):
        path = self.checkpoint()
        original, state = restore_checkpoint(path)
        legacy = copy.deepcopy(state)
        legacy["config"].pop("method_contract")
        legacy["config"]["source_sha256"] = dict(LEGACY_V1_SOURCE)
        new_path = self.root / "legacy.pt"
        save_checkpoint(new_path, legacy)
        restored, _ = restore_checkpoint(new_path)
        xyz, *_ = load_inputs(self.data, self.phy, "float32")
        x = torch.as_tensor(xyz)
        torch.testing.assert_close(original.net(x), restored.net(x), rtol=0, atol=0)
        self.assertEqual(set(original.net.state_dict()), set(restored.net.state_dict()))
        self.assertIn("alpha", legacy["model_state"])
        self.assertIn("ub", legacy["model_state"])
        self.assertIn("net.0.layer.weight", legacy["model_state"])

    def test_legacy_resume_requires_opt_in_and_matches_continuation(self):
        self.assertEqual(self.run_cli(self.args(4)), 0)
        full = next(self.output.iterdir())
        state = self.legacy_state()
        legacy_path = self.root / "legacy.pt"
        save_checkpoint(legacy_path, state)
        content = legacy_path.read_bytes()
        args = self.args(4) + ["--resume", str(legacy_path)]
        count = len(list(self.output.iterdir()))
        self.assertEqual(self.run_cli(args), 1)
        self.assertEqual(len(list(self.output.iterdir())), count)
        self.assertEqual(self.run_cli(args + ["--allow-legacy-resume"]), 0)
        migrated = [p for p in self.output.iterdir()
                    if json.loads((p / "config.json").read_text()).get(
                        "resume_compatibility", {}).get("mode") == "audited-legacy-v1"]
        self.assertEqual(len(migrated), 1)
        expected = read_checkpoint(full / "checkpoints/step_00000004.pt")
        actual = read_checkpoint(migrated[0] / "checkpoints/step_00000004.pt")
        for name, tensor in expected["model_state"].items():
            torch.testing.assert_close(actual["model_state"][name], tensor, rtol=0, atol=0)
        self.assertEqual(legacy_path.read_bytes(), content)

    def test_unknown_sources_method_and_changed_data_cannot_bypass_resume(self):
        state = read_checkpoint(self.checkpoint())
        current = copy.deepcopy(state["config"])
        current["steps"] = 4
        validate_resume(current, state)
        for kind in ("source", "method", "data"):
            bad = copy.deepcopy(state)
            if kind == "source":
                bad["config"]["source_sha256"]["train.py"] = "unknown"
            elif kind == "method":
                bad["config"]["method_contract"] = "unverified-method"
            else:
                bad["config"]["data_sha256"]["physics"] = "changed"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_resume(current, bad, allow_legacy=True)
        # A legacy-looking file with unrecognized source is also refused.
        bad = self.legacy_state()
        bad["config"]["source_sha256"]["train.py"] = "unknown"
        with self.assertRaises(ValueError):
            validate_resume(current, bad, allow_legacy=True)

    def test_strict_model_restore_rejects_missing_weights(self):
        state = read_checkpoint(self.checkpoint())
        state["model_state"].pop("net.0.layer.weight")
        with self.assertRaises(RuntimeError):
            build_from_checkpoint(state, torch.device("cpu"))

    def test_invalid_checkpoint_is_rejected(self):
        for i, payload in enumerate(({}, {"format": "unknown"},
                                     {"format": "pinn-sr-state-v1"})):
            path = self.root / f"invalid_{i}.pt"
            save_checkpoint(path, payload)
            with self.assertRaises(ValueError):
                read_checkpoint(path)

    def test_absolute_cli_from_other_working_directory_preserves_root_paths(self):
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "train.py"), "--check-only",
             "--device", "cpu", "--output-root", str(self.output)],
            cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        config = json.loads(result.stdout)
        self.assertEqual(Path(config["data_dir"]), ROOT / "data/5")
        self.assertFalse(self.output.exists())

    def test_sources_are_hashed_and_src_is_protected(self):
        hashes = source_hashes()
        self.assertIn("src/pinn_sr/pinn.py", hashes)
        self.assertIn("train.py", hashes)
        self.assertNotIn("pinn/pinn.py", hashes)
        for path in ("src", "src/pinn_sr/runs"):
            with self.assertRaises(ValueError):
                validate_output_root((ROOT / path).resolve(), self.data, self.phy)

    def test_random_sampler_state_restores_exact_next_batch(self):
        a = PhysicsSampler(7, 3, "random", 1234)
        a.next_indices()
        state = copy.deepcopy(a.state())
        b = PhysicsSampler(7, 3, "random", 987)
        b.restore(state)
        for _ in range(4):
            np.testing.assert_array_equal(a.next_indices(), b.next_indices())

    def test_failed_training_records_summary_and_closes_log_handlers(self):
        import logging
        with mock.patch.object(ConfigurablePINN, "loss_terms",
                               side_effect=FloatingPointError("synthetic failure")):
            self.assertEqual(self.run_cli(self.args()), 1)
        run = next(self.output.iterdir())
        summary = json.loads((run / "summary.json").read_text())
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["completed_step"], 0)
        self.assertEqual(logging.getLogger(f"pinn_sr.{run.name}").handlers, [])

    def test_metrics_are_batched_and_preserve_training_mode(self):
        from src.pinn_sr.metrics import evaluate_fit
        model, _ = restore_checkpoint(self.checkpoint())
        xyz, velocity, *_ = load_inputs(self.data, self.phy, "float32")
        model.net.train()
        a = evaluate_fit(model, xyz, velocity, torch.device("cpu"), torch.float32, 1)
        self.assertTrue(model.net.training)
        b = evaluate_fit(model, xyz, velocity, torch.device("cpu"), torch.float32, len(xyz))
        for key in ("mae_uvw", "rmse_uvw", "relative_l2_uvw"):
            np.testing.assert_allclose(a[key], b[key], rtol=1e-6, atol=1e-7)
        self.assertEqual(a["evaluation"], "training_observation_fit_not_independent_test")

    def test_pure_metric_reduction(self):
        from src.pinn_sr.metrics import metrics_from_sums
        result = metrics_from_sums([2, 8, 18], [2, 4, 6], [8, 32, 72], 2)
        self.assertEqual(result["mae_uvw"], [1, 2, 3])
        self.assertEqual(result["rmse_uvw"], [1, 2, 3])
        self.assertEqual(result["relative_l2_uvw"], [.5, .5, .5])
        with self.assertRaises(ValueError):
            metrics_from_sums([0, 0, 0], [0, 0, 0], [0, 0, 0], 0)

    def test_keyboard_interrupt_records_interrupted_status(self):
        with mock.patch.object(ConfigurablePINN, "loss_terms", side_effect=KeyboardInterrupt()):
            self.assertEqual(self.run_cli(self.args()), 130)
        run = next(self.output.iterdir())
        self.assertEqual(json.loads((run / "summary.json").read_text())["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
