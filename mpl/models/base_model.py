from abc import ABC, abstractmethod
from typing import Union, Sized, List, Tuple
from copy import deepcopy

import torch
from torch import nn as nn

from ..nn.linear import DenseLinear
from ..nn.conv2d import DenseConv2d
from .utils import collect_leaf_modules, is_parameterized


class BaseModel(nn.Module, ABC):
    def __init__(self):
        super(BaseModel, self).__init__()

        self.prunable_layers: list = []
        self.prunable_layer_prefixes: list = []

    def clone_from_model(self, original_model: nn.Module = None):
        # copying all submodules from original model
        for name, module in original_model._modules.items():
            self.add_module(name, deepcopy(module))

    def collect_prunable_layers(self) -> None:
        self.prunable_layers, self.prunable_layer_prefixes = self.find_layers(lambda x: is_parameterized(x))

    def convert_eligible_layers(self):
        # changing all conv2d and linear layers to customized ones
        for module_name, old_module in zip(self.prunable_layer_prefixes, self.prunable_layers):
            if isinstance(old_module, nn.Linear):
                self.set_module_by_name(module_name, DenseLinear.from_linear(old_module))
            elif isinstance(old_module, nn.Conv2d):
                self.set_module_by_name(module_name, DenseConv2d.from_conv2d(old_module))

    def find_layers(self, criterion) -> Tuple[List, List]:
        layers, names = [], []
        collect_leaf_modules(self, criterion, layers, names)
        return layers, names

    @abstractmethod
    def forward(self, inputs) -> torch.Tensor:
        pass

    def prune_by_threshold(self, thr_arg: Union[int, float, Sized]):
        prunable_layers = self.prunable_layers
        if isinstance(thr_arg, Sized):
            assert len(prunable_layers) == len(thr_arg)
        else:
            thr_arg = [thr_arg] * len(prunable_layers)
        for thr, layer in zip(thr_arg, prunable_layers):
            if thr is not None:
                layer.prune_by_threshold(thr)

        return self

    def prune_by_rank(self, rank_arg: Union[int, float, Sized]):
        prunable_layers = self.prunable_layers
        if isinstance(rank_arg, Sized):
            assert len(prunable_layers) == len(rank_arg)
        else:
            rank_arg = [rank_arg] * len(prunable_layers)
        for rank, layer in zip(rank_arg, prunable_layers):
            if rank is not None:
                layer.prune_by_rank(rank)

        return self

    def prune_by_pct(self, pct_arg: Union[int, float, Sized]):
        prunable_layers = self.prunable_layers
        if isinstance(pct_arg, Sized):
            assert len(prunable_layers) == len(pct_arg)
        else:
            pct_arg = [pct_arg] * len(prunable_layers)
        for pct, layer in zip(pct_arg, prunable_layers):
            if pct is not None:
                layer.prune_by_pct(pct)

        return self

    def random_prune_by_pct(self, pct_arg: Union[int, float, Sized]):
        prunable_layers = self.prunable_layers
        if isinstance(pct_arg, Sized):
            assert len(prunable_layers) == len(pct_arg)
        else:
            pct_arg = [pct_arg] * len(prunable_layers)
        for pct, layer in zip(pct_arg, prunable_layers):
            if pct is not None:
                layer.random_prune_by_pct(pct)

        return self

    def calc_num_prunable_params(self, count_bias=True, display=False):
        total_param_in_use = 0
        total_param = 0
        for layer, layer_prefix in zip(self.prunable_layers, self.prunable_layer_prefixes):
            num_bias = layer.bias.nelement() if layer.bias is not None and count_bias else 0
            num_weight = layer.num_weight
            num_params_in_use = num_weight + num_bias
            num_params = layer.weight.nelement() + num_bias
            total_param_in_use += num_params_in_use
            total_param += num_params

            if display:
                print("Layer name: {}. remaining/all: {}/{} = {}".format(layer_prefix, num_params_in_use, num_params,
                                                                         num_params_in_use / num_params))
        if display:
            print("Total: remaining/all: {}/{} = {}".format(total_param_in_use, total_param,
                                                            total_param_in_use / total_param))
        return total_param_in_use, total_param

    def nnz(self, count_bias=True):
        # number of parameters in use in prunable layers
        return self.calc_num_prunable_params(count_bias=count_bias)[0]

    def nelement(self, count_bias=True):
        # number of all parameters in prunable layers
        return self.calc_num_prunable_params(count_bias=count_bias)[1]

    def density(self, count_bias=True):
        total_param_in_use, total_param = self.calc_num_prunable_params(count_bias=count_bias)
        return total_param_in_use / total_param

    def _get_module_by_list(self, module_names: List):
        module = self
        for name in module_names:
            module = getattr(module, name)
        return module

    def get_module_by_name(self, module_name: str):
        return self._get_module_by_list(module_name.split('.'))

    def set_module_by_name(self, module_name: str, new_module):
        splits = module_name.split('.')
        self._get_module_by_list(splits[:-1]).__setattr__(splits[-1], new_module)

    def get_mask_by_name(self, param_name: str):
        if param_name.endswith("bias"):  # todo
            return None
        module = self._get_module_by_list(param_name.split('.')[:-1])
        return module.mask if hasattr(module, "mask") else None

    @torch.no_grad()
    def reinit_from_model(self, final_model):
        assert isinstance(final_model, self.__class__)
        for self_layer, layer in zip(self.prunable_layers, final_model.prunable_layers):
            self_layer.mask = layer.mask.clone().to(self_layer.mask.device)

    def to_sparse(self):
        self_copy = deepcopy(self)
        for module_name, old_module in zip(self.prunable_layer_prefixes, self.prunable_layers):
            self_copy.set_module_by_name(module_name, old_module.to_sparse())
        self.collect_prunable_layers()
        return self_copy

    def to(self, *args, **kwargs):
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None:
            # move masks to device
            for m in self.prunable_layers:
                m.move_data(device)
        return super(BaseModel, self).to(*args, **kwargs)
