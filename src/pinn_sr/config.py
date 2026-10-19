"""CLI defaults and compatibility contract (not experiment provenance)."""
import argparse
import math

METHOD_CONTRACT = "steady-ns3d-tanh-mre-sum-adam-v1"
SIGNATURE_FIELDS = (
    "model", "hidden_layers", "hidden_width", "dtype", "optimizer", "lr",
    "weight_data", "weight_physics", "seed", "phy_batch_size", "phy_sampling",
    "data_sha256",
)


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
    parser = argparse.ArgumentParser(description="Configurable, non-overwriting runner for the author's steady 3D PINN.",
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
    parser.add_argument("--allow-legacy-resume", action="store_true",
                        help="explicitly migrate an audited pre-refactor v1 checkpoint")
    return parser
