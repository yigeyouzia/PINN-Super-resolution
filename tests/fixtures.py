"""Reusable synthetic CSV fixture; all generated experiment data stays in /tmp."""
import contextlib
import io
from pathlib import Path
import tempfile
import numpy as np
from train import main
from src.pinn_sr.config import make_parser


class SyntheticCase:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pinn-sr-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.output = self.root / "runs"
        xyz = np.array([[0, 0, 0], [1, 0.2, 0.1], [0.3, 1, 0.4],
                        [0.1, 0.5, 1], [0.6, 0.7, 0.8]], dtype=np.float32)
        velocity = xyz + np.array([1, 2, 3], dtype=np.float32)
        np.savetxt(self.data / "coordinate.csv", xyz, delimiter=",")
        for i, axis in enumerate("xyz"):
            np.savetxt(self.data / f"velocity_{axis}.csv",
                       velocity[:, i][None, :], delimiter=",")
        self.phy = self.data / "phy.csv"
        np.savetxt(self.phy, np.random.default_rng(42).random((7, 3)), delimiter=",")

    def args(self, steps=2):
        return ["--data-dir", str(self.data), "--phy-file", str(self.phy),
                "--output-root", str(self.output), "--device", "cpu",
                "--hidden-layers", "2", "--hidden-width", "8", "--steps", str(steps),
                "--phy-batch-size", "3", "--save-every", "1", "--num-threads", "1"]

    def run_cli(self, args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(args)

    def config(self):
        return vars(make_parser().parse_args(self.args()))
