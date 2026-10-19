"""Read-only author oracle for tests only; never imported by production code."""
import functools
import importlib.util
import sys
from src.pinn_sr.experiment import ROOT


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
