"""Validated observation CSVs and resumable collocation sampling."""
from pathlib import Path
import numpy as np
from .experiment import sha256


def load_inputs(data_dir: Path, phy_file: Path, dtype_name: str):
    files = [data_dir / "coordinate.csv"] + [
        data_dir / f"velocity_{axis}.csv" for axis in ("x", "y", "z")]
    for path in files + [phy_file]:
        if not path.is_file():
            raise ValueError(f"Missing input file: {path}")
    dtype = np.float32 if dtype_name == "float32" else np.float64
    xyz = np.loadtxt(files[0], delimiter=",", ndmin=2, dtype=dtype)
    if xyz.shape[1] != 3 or len(xyz) == 0:
        raise ValueError("coordinate.csv must be a nonempty N x 3 numeric array")
    components = []
    for path in files[1:]:
        column = np.loadtxt(path, delimiter=",", ndmin=2, dtype=dtype)
        if 1 not in column.shape or column.size != len(xyz):
            raise ValueError(f"{path.name}: expected 1 x N or N x 1, N={len(xyz)}")
        components.append(column.reshape(-1))
    velocity = np.stack(components, axis=1)
    phy = np.loadtxt(phy_file, delimiter=",", ndmin=2, dtype=dtype)
    if phy.shape[1] != 3 or len(phy) == 0:
        raise ValueError("phy.csv must be a nonempty M x 3 numeric array")
    if not all(np.isfinite(array).all() for array in (xyz, velocity, phy)):
        raise ValueError("Inputs contain NaN/Inf or overflow in the selected dtype")
    if np.any(velocity == 0):
        raise ValueError("Zero velocity denominator in author relative-error loss; "
                         "no epsilon is silently added")
    ub, lb = xyz.max(axis=0), xyz.min(axis=0)
    if not np.all(np.isfinite(ub - lb) & ((ub - lb) > 0)):
        raise ValueError("Zero/invalid coordinate range; constant-plane dimensions "
                         "need a separately defined physical normalization domain")
    metadata = {
        "observation_count": len(xyz), "collocation_count": len(phy),
        "min_abs_velocity": np.abs(velocity).min(axis=0).tolist(),
        "ub": ub.tolist(), "lb": lb.tolist(),
        "ub_velocity": velocity.max(axis=0).tolist(),
        "lb_velocity": velocity.min(axis=0).tolist(),
    }
    hashes = {str(path): sha256(path) for path in files + [phy_file]}
    return xyz, velocity, phy, metadata, hashes

class PhysicsSampler:
    def __init__(self, size, batch, mode, seed):
        self.size = size
        self.batch = min(batch, size)
        self.mode = mode
        self.offset = 0
        self.rng = np.random.default_rng(seed)

    def next_indices(self):
        if self.mode == "random":
            return self.rng.choice(self.size, self.batch, replace=False)
        end = min(self.offset + self.batch, self.size)
        indices = np.arange(self.offset, end)
        self.offset = end if end < self.size else 0
        return indices

    def state(self):
        return {"offset": self.offset, "rng": self.rng.bit_generator.state}

    def restore(self, state):
        self.offset = state["offset"]
        self.rng.bit_generator.state = state["rng"]
