#!/usr/bin/env python3
"""Configurable, non-overwriting runner for the author's steady 3D PINN."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import functools
import hashlib
import importlib.util
import json
import logging
import math
import platform
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import uuid

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
FORMAT = "pinn-sr-state-v1"
COEFFICIENTS = ("alpha", "lambda_1x", "lambda_1y", "lambda_1z",
                "lambda_2x", "lambda_2y", "lambda_2z")
SIGNATURE_FIELDS = (
    "model", "hidden_layers", "hidden_width", "dtype", "optimizer", "lr",
    "weight_data", "weight_physics", "seed", "phy_batch_size", "phy_sampling",
    "data_sha256", "source_sha256",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return parsed


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", default="data/5", help="observation CSV directory")
    parser.add_argument("--phy-file", default="data/phy.csv", help="collocation CSV")
    parser.add_argument("--model", choices=["mlp"], default="mlp")
    parser.add_argument("--hidden-layers", type=positive_int, default=10)
    parser.add_argument("--hidden-width", type=positive_int, default=100)
    parser.add_argument("--steps", type=positive_int, default=20,
                        help="total target Adam updates, including resumed steps")
    parser.add_argument("--optimizer", choices=["adam"], default="adam")
    parser.add_argument("--lr", type=positive_float, default=1e-3)
    parser.add_argument("--phy-batch-size", type=positive_int, default=128)
    parser.add_argument("--phy-sampling", choices=["sequential", "random"],
                        default="sequential")
    parser.add_argument("--weight-data", type=nonnegative_float, default=5.0)
    parser.add_argument("--weight-physics", type=nonnegative_float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="auto selects CUDA if available, otherwise CPU; no MPS")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--num-threads", type=positive_int, default=None,
                        help="optional PyTorch CPU thread limit")
    parser.add_argument("--log-every", type=positive_int, default=1)
    parser.add_argument("--save-every", type=positive_int, default=10)
    parser.add_argument("--eval-batch-size", type=positive_int, default=1024)
    parser.add_argument("--output-root", default="runs", help="parent for a NEW run")
    parser.add_argument("--run-tag", default="", help="optional readable label")
    parser.add_argument("--resume", default=None,
                        help="new-format state checkpoint; always creates a NEW run")
    parser.add_argument("--check-only", action="store_true",
                        help="validate paths/data/device without training or writing")
    return parser


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; refusing CPU fallback")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


@functools.lru_cache(maxsize=1)
def load_author_classes():
    """Reuse loss_func/predict exactly, without changing author files or device globals."""
    previous_network = sys.modules.get("network")
    previous_paths = list(sys.path)
    previous_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        network_spec = importlib.util.spec_from_file_location(
            "_pinn_sr_author_network", ROOT / "pinn/network.py")
        network = importlib.util.module_from_spec(network_spec)
        network_spec.loader.exec_module(network)
        sys.modules["network"] = network
        pinn_spec = importlib.util.spec_from_file_location(
            "_pinn_sr_author_pinn", ROOT / "pinn/pinn.py")
        pinn = importlib.util.module_from_spec(pinn_spec)
        pinn_spec.loader.exec_module(pinn)
        return pinn.PINN, network.DNN
    finally:
        if previous_network is None:
            sys.modules.pop("network", None)
        else:
            sys.modules["network"] = previous_network
        sys.path[:] = previous_paths
        sys.dont_write_bytecode = previous_bytecode


class Layer(nn.Module):
    def __init__(self, n_in: int, n_out: int, activation):
        super().__init__()
        self.layer = nn.Linear(n_in, n_out)
        self.activation = activation

    def forward(self, value):
        value = self.layer(value)
        return self.activation(value) if self.activation is not None else value


class PortableDNN(nn.Module):
    """Same Tanh/Xavier architecture, but normalization tensors are portable buffers."""
    def __init__(self, layers, ub, lb, dtype):
        super().__init__()
        self.net = nn.ModuleList(
            Layer(layers[i], layers[i + 1],
                  nn.Tanh() if i < len(layers) - 2 else None)
            for i in range(len(layers) - 1))
        self.register_buffer("ub", torch.as_tensor(ub, dtype=dtype))
        self.register_buffer("lb", torch.as_tensor(lb, dtype=dtype))
        self.to(dtype=dtype)
        self.apply(self.initialize)

    @staticmethod
    def initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, value):
        value = (value - self.lb) / (self.ub - self.lb)
        for layer in self.net:
            value = layer(value)
        return value


class ConfigurablePINN:
    def __init__(self, config, ub, lb, ub_velocity, lb_velocity, device):
        author, _ = load_author_classes()
        dtype = getattr(torch, config["dtype"])
        layers = [3] + [config["hidden_width"]] * config["hidden_layers"] + [4]
        self.net = PortableDNN(layers, ub, lb, dtype).to(device)
        self.ub_stem = np.asarray(ub_velocity)
        self.lb_stem = np.asarray(lb_velocity)
        for name in COEFFICIENTS:
            parameter = nn.Parameter(torch.zeros(1, device=device, dtype=dtype))
            setattr(self, name, parameter)
            self.net.register_parameter(name, parameter)
        self.predict = author.predict.__get__(self)
        self._author_loss = author.loss_func.__get__(self)
        self.adam = torch.optim.Adam(self.net.parameters(), lr=config["lr"])
        self.weight_data = config["weight_data"]
        self.weight_physics = config["weight_physics"]

    def loss_terms(self, xyz, velocity, phy):
        _, data, physics = self._author_loss(xyz, velocity, phy)
        return self.weight_data * data + self.weight_physics * physics, data, physics


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


def validate_output_root(output: Path, data_dir: Path, phy_file: Path):
    protected = [ROOT / name for name in ("data", "pinn", "docs", ".git", "tests")]
    protected += list(ROOT.glob("result_model_pinn*"))
    protected += [data_dir, phy_file]
    for directory in protected:
        directory = directory.resolve()
        if output == directory or output.is_relative_to(directory):
            raise ValueError(f"Protected output location: {output} (under {directory})")
    if output.exists() and not output.is_dir():
        raise ValueError(f"Output parent is not a directory: {output}")


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")[:30] or "data"


def create_run_dir(output: Path, config: dict) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    readable = (
        f'{config["model"]}_L{config["hidden_layers"]}_W{config["hidden_width"]}'
        f'_case{slug(Path(config["data_dir"]).name)}_lr{config["lr"]:g}'
        f'_phy{config["phy_batch_size"]}_wd{config["weight_data"]:g}'
        f'wp{config["weight_physics"]:g}_seed{config["seed"]}'
    )
    if config["run_tag"]:
        readable += "_" + slug(config["run_tag"])
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = output / f"{readable}__{stamp}__{digest}-{uuid.uuid4().hex[:8]}"
    run.mkdir(exist_ok=False)  # exclusive; never reuse an existing experiment
    (run / "checkpoints").mkdir()
    return run


def write_json(path: Path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def save_checkpoint(path: Path, payload):
    with path.open("xb") as stream:
        torch.save(payload, stream)


def read_checkpoint(path: Path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("format") != FORMAT:
        raise ValueError("Only train.py state checkpoints are accepted, not author pickles")
    return state


def build_from_checkpoint(state, device):
    config = state["config"]
    bounds = state["bounds"]
    model = ConfigurablePINN(config, bounds["ub"], bounds["lb"],
                             bounds["ub_velocity"], bounds["lb_velocity"], device)
    model.net.load_state_dict(state["model_state"])
    model.adam.load_state_dict(state["optimizer_state"])
    return model


def restore_checkpoint(path: Path, device: str = "cpu"):
    """Inference/verification helper; no unsafe whole-object loading."""
    state = read_checkpoint(Path(path))
    return build_from_checkpoint(state, select_device(device)), state


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


def require_finite(tensors, label):
    tensors = list(tensors)
    if tensors and not bool(torch.stack([torch.isfinite(t).all() for t in tensors]).all()):
        raise FloatingPointError(f"Nonfinite {label}; stopping without changing the loss")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_fit(model, xyz, velocity, device, dtype, batch):
    squared = np.zeros(3, dtype=np.float64)
    absolute = np.zeros(3, dtype=np.float64)
    model.net.eval()
    with torch.no_grad():
        for start in range(0, len(xyz), batch):
            inputs = torch.as_tensor(xyz[start:start + batch], device=device, dtype=dtype)
            output = model.net(inputs)
            require_finite([output], "evaluation predictions")
            error = output[:, :3].cpu().numpy().astype(np.float64) - velocity[start:start + batch]
            squared += np.sum(error ** 2, axis=0)
            absolute += np.sum(np.abs(error), axis=0)
    denominator = np.sum(velocity.astype(np.float64) ** 2, axis=0)
    return {"evaluation": "training_observation_fit_not_independent_test",
            "points": len(xyz), "mae_uvw": (absolute / len(xyz)).tolist(),
            "rmse_uvw": np.sqrt(squared / len(xyz)).tolist(),
            "relative_l2_uvw": np.sqrt(squared / denominator).tolist()}


def environment(config, device):
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--short"], text=True,
            stderr=subprocess.DEVNULL).splitlines()
    except (OSError, subprocess.CalledProcessError):
        commit, status = None, []
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version, "platform": platform.platform(),
            "torch": str(torch.__version__), "numpy": str(np.__version__),
            "cuda_runtime": torch.version.cuda, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch_cpu_threads": torch.get_num_threads(), "git_commit": commit,
            "git_status": status, "command": sys.argv,
            "source_sha256": config["source_sha256"]}


def training(config, arrays, bounds, device, resume):
    # Author imports reset seed at import, so load them BEFORE setting the requested seed.
    load_author_classes()
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    if config["num_threads"]:
        torch.set_num_threads(config["num_threads"])
    dtype = getattr(torch, config["dtype"])
    xyz, velocity, phy = arrays
    model = ConfigurablePINN(config, bounds["ub"], bounds["lb"],
                             bounds["ub_velocity"], bounds["lb_velocity"], device)
    sampler = PhysicsSampler(len(phy), config["phy_batch_size"],
                             config["phy_sampling"], config["seed"])
    completed = 0
    if resume:
        model.net.load_state_dict(resume["model_state"])
        model.adam.load_state_dict(resume["optimizer_state"])
        sampler.restore(resume["sampler_state"])
        torch.set_rng_state(resume["torch_rng"])
        if device.type == "cuda" and resume["cuda_rng"]:
            if len(resume["cuda_rng"]) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(resume["cuda_rng"])
        completed = resume["step"]
    x_tensor = torch.as_tensor(xyz, device=device, dtype=dtype)
    velocity_tensor = torch.as_tensor(velocity, device=device, dtype=dtype)
    run = create_run_dir(Path(config["output_root"]), config)
    logger = logging.getLogger(f"pinn_sr.{run.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    console = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(run / "train.log", mode="x", encoding="utf-8")
    logger.addHandler(console)
    logger.addHandler(file_handler)
    logger.info("RUN_DIR=%s", run)
    write_json(run / "config.json", config)
    write_json(run / "environment.json", environment(config, device))
    summary = {"status": "running", "run_dir": str(run), "completed_step": completed}
    metrics = (run / "metrics.csv").open("x", newline="", encoding="utf-8")
    writer = csv.writer(metrics)
    writer.writerow(["step", "loss_total", "loss_data", "loss_physics",
                     "observation_count", "phy_batch_actual", "elapsed_seconds", "lr"])
    last_checkpoint = None
    elapsed = 0.0
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        started = time.perf_counter()
        for step in range(completed + 1, config["steps"] + 1):
            model.net.train()
            indices = sampler.next_indices()
            physics_tensor = torch.as_tensor(phy[indices], device=device, dtype=dtype)
            model.adam.zero_grad(set_to_none=True)
            total, data, physics = model.loss_terms(x_tensor, velocity_tensor, physics_tensor)
            require_finite((total, data, physics), "loss")
            total.backward()
            require_finite((p.grad for p in model.net.parameters() if p.grad is not None),
                           "gradients")
            model.adam.step()
            require_finite(model.net.parameters(), "parameters")
            completed = step
            values = [float(t.detach()) for t in (total, data, physics)]
            elapsed = time.perf_counter() - started
            writer.writerow([step, *values, len(xyz), len(indices), elapsed, config["lr"]])
            if step == 1 or step % config["log_every"] == 0 or step == config["steps"]:
                metrics.flush()
                logger.info("step=%d total=%.6g data=%.6g phy=%.6g actual_phy=%d time=%.3fs",
                            step, *values, len(indices), elapsed)
            if step % config["save_every"] == 0 or step == config["steps"]:
                last_checkpoint = run / "checkpoints" / f"step_{step:08d}.pt"
                save_checkpoint(last_checkpoint, {
                    "format": FORMAT, "model_state": model.net.state_dict(),
                    "optimizer_state": model.adam.state_dict(), "config": config,
                    "bounds": bounds, "step": step, "sampler_state": sampler.state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                })
                logger.info("checkpoint=%s", last_checkpoint)
        synchronize(device)
        elapsed = time.perf_counter() - started
        summary.update({
            "training_wall_seconds_this_run": elapsed,
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda" else None,
            "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2
                if device.type == "cuda" else None,
            "last_checkpoint": str(last_checkpoint),
        })
        # Verify the newly saved state, not any author whole-object checkpoint.
        restored, _ = restore_checkpoint(last_checkpoint, str(device))
        with torch.no_grad():
            expected = model.net(x_tensor[:32])
            actual = restored.net(x_tensor[:32])
            require_finite((expected, actual), "checkpoint verification predictions")
            torch.testing.assert_close(actual, expected)
        summary["checkpoint_roundtrip"] = "passed"
        summary["train_fit"] = evaluate_fit(model, xyz, velocity, device, dtype,
                                             config["eval_batch_size"])
        summary["status"] = "completed"
        logger.info("checkpoint round-trip PASS; fit RMSE uvw=%s",
                    summary["train_fit"]["rmse_uvw"])
    except BaseException as error:
        summary["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        summary["last_checkpoint"] = str(last_checkpoint) if last_checkpoint else None
        logger.exception("Run stopped")
        raise
    finally:
        summary["completed_step"] = completed
        metrics.close()
        write_json(run / "summary.json", summary)
        for handler in (console, file_handler):
            logger.removeHandler(handler)
            handler.close()
    return run


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        if not 0 <= args.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        if args.weight_data == 0 and args.weight_physics == 0:
            raise ValueError("At least one loss weight must be positive")
        device = select_device(args.device)
        data_dir, phy_file = resolve_path(args.data_dir), resolve_path(args.phy_file)
        output = resolve_path(args.output_root)
        validate_output_root(output, data_dir, phy_file)
        xyz, velocity, phy, bounds, hashes = load_inputs(data_dir, phy_file, args.dtype)
        config = vars(args).copy()
        config.update({"data_dir": str(data_dir), "phy_file": str(phy_file),
                       "output_root": str(output), "resolved_device": str(device),
                       "data_info": bounds, "data_sha256": hashes,
                       "source_sha256": {name: sha256(ROOT / name)
                           for name in ("train.py", "pinn/pinn.py", "pinn/network.py")},
                       "effective_phy_batch_size": min(args.phy_batch_size, len(phy))})
        # Compare content hashes, not machine-specific input paths, when resuming.
        config["data_sha256"] = {
            "coordinate": hashes[str(data_dir / "coordinate.csv")],
            **{f"velocity_{axis}": hashes[str(data_dir / f"velocity_{axis}.csv")]
               for axis in ("x", "y", "z")},
            "physics": hashes[str(phy_file)],
        }
        resume = None
        if args.resume:
            resume_path = resolve_path(args.resume)
            config["resume"] = str(resume_path)
            resume = read_checkpoint(resume_path)
            mismatches = [name for name in SIGNATURE_FIELDS
                          if config[name] != resume["config"][name]]
            if mismatches:
                raise ValueError("Resume configuration/data/source mismatch: "
                                 + ", ".join(mismatches)
                                 + "; repeat the original training arguments")
            if args.steps <= resume["step"]:
                raise ValueError("--steps must exceed the checkpoint's completed step")
            if resume["config"]["resolved_device"] != str(device):
                print("NOTE: cross-device resume; bitwise-identical continuation is not promised",
                      file=sys.stderr)
        if args.check_only:
            print(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        if args.phy_batch_size > len(phy):
            print(f"NOTE: phy batch capped at {len(phy)}; actual size is recorded",
                  file=sys.stderr)
        if min(bounds["min_abs_velocity"]) < 1e-8:
            print("WARNING: near-zero velocity in unmodified relative loss", file=sys.stderr)
        training(config, (xyz, velocity, phy), bounds, device, resume)
        return 0
    except (ValueError, OSError, FloatingPointError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted; previous checkpoints were preserved", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
