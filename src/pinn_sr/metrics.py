"""Numerical velocity metrics and batched training-observation fit."""
import numpy as np
import torch

from .runtime import require_finite


def metrics_from_sums(squared, absolute, reference_squared, points):
    """Pure reduction of per-component sums; no model, file or plotting side effects."""
    squared = np.asarray(squared, dtype=np.float64)
    absolute = np.asarray(absolute, dtype=np.float64)
    reference_squared = np.asarray(reference_squared, dtype=np.float64)
    if points <= 0 or np.any(reference_squared <= 0):
        raise ValueError("Metrics need nonempty points and nonzero reference norms")
    return {
        "points": points,
        "mae_uvw": (absolute / points).tolist(),
        "rmse_uvw": np.sqrt(squared / points).tolist(),
        "relative_l2_uvw": np.sqrt(squared / reference_squared).tolist(),
    }


def evaluate_fit(model, xyz, velocity, device, dtype, batch):
    """Evaluate labelled training points, restoring the caller's train/eval mode."""
    squared = np.zeros(3, dtype=np.float64)
    absolute = np.zeros(3, dtype=np.float64)
    was_training = model.net.training
    model.net.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(xyz), batch):
                inputs = torch.as_tensor(xyz[start:start + batch], device=device, dtype=dtype)
                output = model.net(inputs)
                require_finite([output], "evaluation predictions")
                error = output[:, :3].cpu().numpy().astype(np.float64) - velocity[start:start + batch]
                squared += np.sum(error ** 2, axis=0)
                absolute += np.sum(np.abs(error), axis=0)
    finally:
        model.net.train(was_training)
    denominator = np.sum(velocity.astype(np.float64) ** 2, axis=0)
    return {"evaluation": "training_observation_fit_not_independent_test",
            **metrics_from_sums(squared, absolute, denominator, len(xyz))}
