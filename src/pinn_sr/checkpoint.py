"""Safe state-only v1 loading, inference recovery and audited resume migration."""
from pathlib import Path
import pickle

import torch

from .config import METHOD_CONTRACT, SIGNATURE_FIELDS
from .pinn import ConfigurablePINN
from .runtime import select_device

FORMAT = "pinn-sr-state-v1"

# Exact source profile audited before the 2026-10-19 extraction, not a wildcard.
LEGACY_V1_SOURCE = {
    "train.py": "a517de36e903d71c1a1e772530c4a7b5e468273323cc6fcfdd436042247101ff",
    "pinn/pinn.py": "ddd6bbb22feefd96603ec30f4b2bb18753d27da0e2464273913f721e5e2b5a62",
    "pinn/network.py": "353b91c99aa31baecfb010bd6d5cc1974fed2d2bf59a6b2cab8ba7eb70f36f23"
}


def save_checkpoint(path: Path, payload):
    with path.open("xb") as stream:
        torch.save(payload, stream)


def read_checkpoint(path: Path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError) as error:
        raise ValueError("Invalid state checkpoint; author whole-object pickles are not accepted") from error
    if not isinstance(state, dict) or state.get("format") != FORMAT:
        raise ValueError("Only pinn-sr-state-v1 checkpoints are accepted, not author pickles")
    required = ("config", "bounds", "model_state", "step")
    if any(name not in state for name in required):
        raise ValueError("Incomplete checkpoint: expected config/bounds/model_state/step")
    if not isinstance(state["config"], dict) or not isinstance(state["bounds"], dict):
        raise ValueError("Invalid checkpoint config/bounds")
    return state


def build_from_checkpoint(state, device):
    """Restore model only; no optimizer allocation, even for training checkpoints."""
    config, bounds = state["config"], state["bounds"]
    model = ConfigurablePINN(config, bounds["ub"], bounds["lb"],
                             bounds["ub_velocity"], bounds["lb_velocity"], device)
    model.net.load_state_dict(state["model_state"], strict=True)
    model.net.eval()
    return model


def restore_checkpoint(path: Path, device: str = "cpu"):
    state = read_checkpoint(Path(path))
    return build_from_checkpoint(state, select_device(device)), state


def validate_resume(config, state, allow_legacy=False):
    """Separate an audited method migration from ordinary strict-source resume."""
    previous = state["config"]
    mismatches = [key for key in SIGNATURE_FIELDS
                  if key not in previous or config[key] != previous[key]]
    if mismatches:
        raise ValueError("Resume configuration/data mismatch: " + ", ".join(mismatches)
                         + "; repeat the original training arguments")
    if not isinstance(state["step"], int) or state["step"] < 0:
        raise ValueError("Invalid completed step")
    for key in ("optimizer_state", "sampler_state", "torch_rng", "cuda_rng"):
        if key not in state:
            raise ValueError("Resume requires checkpoint field: " + key)
    if config["steps"] <= state["step"]:
        raise ValueError("--steps must exceed the checkpoint's completed step")

    same_source = config["source_sha256"] == previous.get("source_sha256")
    if same_source and previous.get("method_contract") == METHOD_CONTRACT:
        return {"mode": "same-source", "parent_step": state["step"]}
    audited_legacy = (previous.get("method_contract") is None
                      and previous.get("source_sha256") == LEGACY_V1_SOURCE)
    if audited_legacy:
        if not allow_legacy:
            raise ValueError("Audited pre-refactor v1 checkpoint: add --allow-legacy-resume "
                             "for explicit migration; inference does not require this flag")
        return {"mode": "audited-legacy-v1", "parent_step": state["step"],
                "parent_source_sha256": previous["source_sha256"]}
    raise ValueError("Resume source/method mismatch: no audited migration exists for this "
                     "source profile; inference remains available, do not bypass validation")
