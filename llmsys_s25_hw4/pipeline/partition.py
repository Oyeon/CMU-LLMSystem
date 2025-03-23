# partition.py

from typing import List, Tuple
import torch
from torch import nn

class WithDevice(nn.Module):
    def __init__(self, module: nn.Module, device: torch.device):
        super().__init__()
        self._module = module
        self._device = torch.device(device)

    def forward(self, *args, **kwargs):
        return self._module(*args, **kwargs)

    @property
    def module(self):
        return self._module

    @property
    def device(self):
        return self._device

def _retrieve_device(module: nn.Module) -> torch.device:
    device = None
    for parameter in module.parameters():
        if device is None:
            device = parameter.device
        elif device != parameter.device:
            raise ValueError(
                f'nn.Module: {module}, should have all parameters on a single device.'
            )
    return device if device is not None else torch.device("cpu")

def _assemble_partition(modules: List[nn.Module]) -> nn.Sequential:
    modules_list: List[nn.Module] = []
    for m in modules:
        # If m is itself an nn.Sequential, flatten it
        if isinstance(m, nn.Sequential):
            modules_list.extend(m.children())
        else:
            modules_list.append(m)
    return nn.Sequential(*modules_list)

# ASSIGNMENT 4.2
def _split_module(modules: nn.Sequential) -> Tuple[List[nn.Sequential], List[torch.device]]:
    """
    Split an nn.Sequential module into partitions according to device. 
    Each partition is a sub-sequence of consecutive layers on the same device.
    The function returns (partitions, devices).
    """
    partitions = []
    devices = []

    current_partition = []
    current_device = None

    for name, child in modules.named_children():
        # 1) Determine child's device
        if isinstance(child, WithDevice):
            child_device = child.device
            actual_child = child.module
        else:
            child_device = _retrieve_device(child)
            actual_child = child
        
        # 2) If we're starting a new partition or device changed
        if current_device is None:
            # Initialize first partition
            current_device = child_device
            current_partition.append(child)
        elif child_device != current_device:
            # Finalize the old partition
            partitions.append(_assemble_partition(current_partition))
            devices.append(current_device)
            # Start new partition
            current_partition = [child]
            current_device = child_device
        else:
            # Same device => accumulate in current partition
            current_partition.append(child)

    # 3) Final partition if non-empty
    if current_partition:
        partitions.append(_assemble_partition(current_partition))
        devices.append(current_device)

    # Convert partitions to an nn.ModuleList for convenience
    partitions = nn.ModuleList(partitions)
    return partitions, devices
