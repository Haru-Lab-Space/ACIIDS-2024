"""
Utilities functions for casting variables to device.
"""

import torch


def cast_to_device(*variables, device: torch.device = None):
    """
    Casts variables to device.
    --------------------------
    Args:
        variables: Variables to cast to device.
        device: Device to cast variables to.

    Returns:
        Variables casted to device.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return tuple(var.to(device) for var in variables)

def cast_dict_to_device(variables, device: torch.device = None):
    """
    Casts variables to device.
    --------------------------
    Args:
        variables: Variables to cast to device.
        device: Device to cast variables to.

    Returns:
        Variables casted to device.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for key in variables.keys():
        variables[key] = variables[key].to(device)
    return variables