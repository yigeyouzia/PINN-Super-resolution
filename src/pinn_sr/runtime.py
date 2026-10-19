"""Explicit execution environment; imports do not reset random seeds."""
from datetime import datetime, timezone
import platform
import random
import subprocess
import sys
import numpy as np
import torch
from .experiment import ROOT


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; refusing CPU fallback")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)

def require_finite(tensors, label):
    tensors = list(tensors)
    if tensors and not bool(torch.stack([torch.isfinite(t).all() for t in tensors]).all()):
        raise FloatingPointError(f"Nonfinite {label}; stopping without changing the loss")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

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

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
