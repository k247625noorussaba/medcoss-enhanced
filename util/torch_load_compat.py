# PyTorch >= 2.6 defaults torch.load(..., weights_only=True), which rejects pickles
# containing argparse.Namespace and other objects in MedCoSS checkpoints.
# PyTorch 1.11 has no weights_only kwarg — use try/except TypeError.

import torch

__all__ = ["torch_load_compat"]


def torch_load_compat(path, map_location=None, **kwargs):
    try:
        if map_location is None:
            return torch.load(path, weights_only=False, **kwargs)
        return torch.load(path, map_location=map_location, weights_only=False, **kwargs)
    except TypeError:
        if map_location is None:
            return torch.load(path, **kwargs)
        return torch.load(path, map_location=map_location, **kwargs)
