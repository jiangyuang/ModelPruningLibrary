import torch.nn as nn
from ..nn.conv2d import DenseConv2d, SparseConv2d
from ..nn.linear import DenseLinear, SparseLinear

from typing import Callable


def is_prunable_fc(layer):
    return isinstance(layer, DenseLinear) or isinstance(layer, SparseLinear)


def is_prunable_conv(layer):
    return isinstance(layer, DenseConv2d) or isinstance(layer, SparseConv2d)


def is_prunable(layer):
    return is_prunable_fc(layer) or is_prunable_conv(layer)


def is_parameterized(layer):
    return is_prunable(layer) or isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d)


def collect_leaf_modules(module, criterion: Callable, layers: list, names: list, prefix: str = ""):
    for key, submodule in module._modules.items():
        new_prefix = prefix
        if prefix != "":
            new_prefix += '.'
        new_prefix += key
        # is leaf and satisfies criterion
        if submodule is not None:
            if len(submodule._modules.keys()) == 0 and criterion(submodule):
                layers.append(submodule)
                names.append(new_prefix)
            collect_leaf_modules(submodule, criterion, layers, names, prefix=new_prefix)
