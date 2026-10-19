"""Thin orchestration of validation, provenance and the training entry point."""
import json
import sys
from .checkpoint import read_checkpoint, validate_resume
from .config import METHOD_CONTRACT, make_parser
from .data import load_inputs
from .experiment import resolve_path, source_hashes, validate_output_root
from .pinn_train import training
from .runtime import select_device


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
                       "source_sha256": source_hashes(),
                       "method_contract": METHOD_CONTRACT,
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
            config["resume_compatibility"] = validate_resume(
                config, resume, allow_legacy=args.allow_legacy_resume)
            if config["resume_compatibility"]["mode"] == "audited-legacy-v1":
                print("NOTE: explicitly migrating audited pre-refactor v1 state into a NEW run",
                      file=sys.stderr)
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
