"""Project paths, provenance and exclusive, non-overwriting output files."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import uuid

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()

def validate_output_root(output: Path, data_dir: Path, phy_file: Path):
    protected = [ROOT / name for name in ("data", "pinn", "src", "docs", ".git", "tests")]
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

def source_hashes():
    """Hash all maintained Python sources; author sources are regression-only."""
    paths = [ROOT / "train.py", ROOT / "src/__init__.py"]
    paths += sorted((ROOT / "src/pinn_sr").glob("*.py"))
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}
