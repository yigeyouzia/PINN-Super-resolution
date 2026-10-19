"""Adam loop and resume state; model construction never creates an optimizer."""
import csv
import logging
from pathlib import Path
import sys
import time
import torch

from .checkpoint import FORMAT, restore_checkpoint, save_checkpoint
from .data import PhysicsSampler
from .experiment import create_run_dir, write_json
from .metrics import evaluate_fit
from .pinn import ConfigurablePINN
from .runtime import environment, require_finite, seed_everything, synchronize


def make_optimizer(model, config):
    return torch.optim.Adam(model.net.parameters(), lr=config["lr"])


def training(config, arrays, bounds, device, resume):
    seed_everything(config["seed"])
    if config["num_threads"]:
        torch.set_num_threads(config["num_threads"])
    dtype = getattr(torch, config["dtype"])
    xyz, velocity, phy = arrays
    model = ConfigurablePINN(config, bounds["ub"], bounds["lb"],
                             bounds["ub_velocity"], bounds["lb_velocity"], device)
    optimizer = make_optimizer(model, config)
    sampler = PhysicsSampler(len(phy), config["phy_batch_size"],
                             config["phy_sampling"], config["seed"])
    completed = 0
    if resume:
        model.net.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
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
            optimizer.zero_grad(set_to_none=True)
            total, data, physics = model.loss_terms(x_tensor, velocity_tensor, physics_tensor)
            require_finite((total, data, physics), "loss")
            total.backward()
            require_finite((p.grad for p in model.net.parameters() if p.grad is not None),
                           "gradients")
            optimizer.step()
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
                    "optimizer_state": optimizer.state_dict(), "config": config,
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
