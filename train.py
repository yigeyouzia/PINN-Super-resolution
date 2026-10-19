#!/usr/bin/env python3
"""Compatibility CLI; maintained implementation lives in src/pinn_sr."""
from src.pinn_sr.cli import main
from src.pinn_sr.checkpoint import restore_checkpoint

__all__ = ["main", "restore_checkpoint"]

if __name__ == "__main__":
    raise SystemExit(main())
