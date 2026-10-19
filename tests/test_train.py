"""CPU regression tests; temporary synthetic CSVs, never author data/weights."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from train import main, restore_checkpoint
from src.pinn_sr.config import make_parser
from src.pinn_sr.data import load_inputs, PhysicsSampler
from src.pinn_sr.experiment import ROOT, validate_output_root
from src.pinn_sr.pinn import ConfigurablePINN
from src.pinn_sr.pinn_train import make_optimizer
from src.pinn_sr.checkpoint import read_checkpoint, save_checkpoint
from src.pinn_sr.runtime import select_device
from tests.author_reference import load_author_classes
from tests.fixtures import SyntheticCase


class RunnerTests(SyntheticCase, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_author_classes()
        torch.set_num_threads(1)

    def test_default_forward_loss_and_gradients_match_author(self):
        legacy, legacy_dnn = load_author_classes()
        config = self.config()
        _, _, _, bounds, _ = load_inputs(self.data, self.phy, "float32")
        # Reference constructor auto-selects CUDA; force CPU in memory for this CPU-only test.
        old_a = legacy.__init__.__globals__["device"]
        old_b = legacy_dnn.__init__.__globals__["device"]
        try:
            legacy.__init__.__globals__["device"] = torch.device("cpu")
            legacy_dnn.__init__.__globals__["device"] = torch.device("cpu")
            torch.manual_seed(1234)
            reference = legacy([3, 8, 8, 4], np.array(bounds["ub"]), np.array(bounds["lb"]),
                               np.array(bounds["ub_velocity"]), np.array(bounds["lb_velocity"]))
        finally:
            legacy.__init__.__globals__["device"] = old_a
            legacy_dnn.__init__.__globals__["device"] = old_b
        torch.manual_seed(1234)
        actual = ConfigurablePINN(config, bounds["ub"], bounds["lb"],
                    bounds["ub_velocity"], bounds["lb_velocity"], torch.device("cpu"))
        xyz, vel, phy, *_ = load_inputs(self.data, self.phy, "float32")
        tensors = [torch.as_tensor(a) for a in (xyz, vel, phy)]
        torch.testing.assert_close(actual.net(tensors[0]), reference.net(tensors[0]),
                                   rtol=0, atol=0)
        expected_loss = reference.loss_func(*tensors)
        actual_loss = actual.loss_terms(*tensors)
        for a, b in zip(actual_loss, expected_loss):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        actual_loss[0].backward()
        expected_loss[0].backward()
        for (_, a), (_, b) in zip(actual.net.named_parameters(), reference.net.named_parameters()):
            if a.grad is None or b.grad is None:
                self.assertIsNone(a.grad)
                self.assertIsNone(b.grad)
            else:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_device_cpu_and_learning_rate_weights_are_effective(self):
        config = self.config()
        config.update(lr=0.002, weight_data=2.0, weight_physics=3.0)
        xyz, vel, phy, bounds, _ = load_inputs(self.data, self.phy, "float32")
        with mock.patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(select_device("cpu").type, "cpu")
            model = ConfigurablePINN(config, bounds["ub"], bounds["lb"],
                    bounds["ub_velocity"], bounds["lb_velocity"], torch.device("cpu"))
        self.assertEqual(model.net.ub.device.type, "cpu")
        self.assertEqual(make_optimizer(model, config).param_groups[0]["lr"], 0.002)
        total, data, physics = model.loss_terms(*(torch.as_tensor(x) for x in (xyz, vel, phy)))
        torch.testing.assert_close(total, 2 * data + 3 * physics)

    def test_explicit_cuda_unavailable_is_rejected(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(ValueError):
                select_device("cuda")

    def test_validation_does_not_create_run(self):
        self.assertEqual(self.run_cli(self.args() + ["--check-only"]), 0)
        self.assertFalse(self.output.exists())

    def test_protected_output_and_symlink_rejected(self):
        for output in (ROOT / "data", ROOT / "pinn",
                       ROOT / "result_model_pinn_100" / "runs"):
            with self.assertRaises(ValueError):
                validate_output_root(output.resolve(), self.data, self.phy)
        link = self.root / "alias"
        link.symlink_to(ROOT / "result_model_pinn_100", target_is_directory=True)
        with self.assertRaises(ValueError):
            validate_output_root((link / "runs").resolve(), self.data, self.phy)

    def test_zero_and_constant_coordinates_rejected(self):
        values = np.zeros((1, 5))
        np.savetxt(self.data / "velocity_x.csv", values, delimiter=",")
        self.assertEqual(self.run_cli(self.args()), 1)
        self.assertFalse(self.output.exists())
        np.savetxt(self.data / "velocity_x.csv", np.ones((1, 5)), delimiter=",")
        xyz = np.loadtxt(self.data / "coordinate.csv", delimiter=",")
        xyz[:, 0] = 2.5
        np.savetxt(self.data / "coordinate.csv", xyz, delimiter=",")
        self.assertEqual(self.run_cli(self.args()), 1)

    def test_sampler_covers_tail_without_empty_batch(self):
        sampler = PhysicsSampler(6, 3, "sequential", 1234)
        self.assertEqual([sampler.next_indices().tolist() for _ in range(3)],
                         [[0, 1, 2], [3, 4, 5], [0, 1, 2]])
        sampler = PhysicsSampler(7, 3, "sequential", 1234)
        self.assertEqual([len(sampler.next_indices()) for _ in range(4)], [3, 3, 1, 3])

    def test_same_arguments_new_runs_preserve_existing_files(self):
        self.assertEqual(self.run_cli(self.args()), 0)
        first = next(self.output.iterdir())
        before = {p.relative_to(first): p.read_bytes() for p in first.rglob("*") if p.is_file()}
        self.assertEqual(self.run_cli(self.args()), 0)
        self.assertEqual(len(list(self.output.iterdir())), 2)
        for path, content in before.items():
            self.assertEqual((first / path).read_bytes(), content)
        summary = json.loads((first / "summary.json").read_text())
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["checkpoint_roundtrip"], "passed")
        model, state = restore_checkpoint(first / "checkpoints/step_00000002.pt")
        self.assertEqual(state["step"], 2)
        self.assertEqual(model.net.lb.device.type, "cpu")
        checkpoint = first / "checkpoints/step_00000002.pt"
        with self.assertRaises(FileExistsError):
            save_checkpoint(checkpoint, state)

    def test_resume_matches_uninterrupted_cpu_training_and_uses_new_run(self):
        self.assertEqual(self.run_cli(self.args(4)), 0)
        full = next(self.output.iterdir())
        self.assertEqual(self.run_cli(self.args(2)), 0)
        partial = next(p for p in self.output.iterdir() if p != full)
        checkpoint = partial / "checkpoints/step_00000002.pt"
        checkpoint_bytes = checkpoint.read_bytes()
        self.assertEqual(self.run_cli(self.args(4) + ["--resume", str(checkpoint)]), 0)
        resumed = next(p for p in self.output.iterdir() if p not in (full, partial))
        a = read_checkpoint(full / "checkpoints/step_00000004.pt")
        b = read_checkpoint(resumed / "checkpoints/step_00000004.pt")
        for name in a["model_state"]:
            torch.testing.assert_close(a["model_state"][name], b["model_state"][name],
                                       rtol=0, atol=0)
        self.assertEqual(checkpoint.read_bytes(), checkpoint_bytes)
        self.assertEqual(self.run_cli(self.args(5) + ["--resume", str(checkpoint),
                                                    "--lr", "0.002"]), 1)

    def test_seed_changes_initialization(self):
        config = self.config()
        _, _, _, bounds, _ = load_inputs(self.data, self.phy, "float32")
        weights = []
        for seed in (1234, 4321):
            torch.manual_seed(seed)
            model = ConfigurablePINN(config, bounds["ub"], bounds["lb"],
                     bounds["ub_velocity"], bounds["lb_velocity"], torch.device("cpu"))
            weights.append(model.net.net[0].layer.weight.detach().clone())
        self.assertFalse(torch.equal(*weights))


if __name__ == "__main__":
    unittest.main()
