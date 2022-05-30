"""
Miscellaneous helper code.
"""

import torch


class DotDict(dict):
    """
    dot.notation access to dictionary attributes
    """

    def __getattr__(self, attr):
        return self[attr]

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_dict_to_torch(stuff, device="cpu"):
    """Set everything in the dict to the specified torch device."""
    if isinstance(stuff, dict):
        for k, v in stuff.items():
            stuff[k] = get_dict_to_torch(v, device)
        return stuff
    if isinstance(stuff, torch.Tensor):
        return stuff.to(device)
    return stuff


def get_dict_to_cpu(stuff):
    """Set everything in the dict to CPU."""
    if isinstance(stuff, dict):
        for k, v in stuff.items():
            stuff[k] = get_dict_to_cpu(v)
        return stuff
    if isinstance(stuff, torch.Tensor):
        return stuff.detach().cpu()
    return stuff


def is_not_none(var):
    """Return True if the variable var is None."""
    return not isinstance(var, type(None))


def get_masked_dict(d, mask):
    """Return a masked dictionary.
    TODO(ethan): add more asserts/checks so this doesn't have unpredictable behavior."""
    masked_dict = {}
    for key, value in d.items():
        masked_dict[key] = value[mask]
    return masked_dict
