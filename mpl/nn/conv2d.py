import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from ..autograd.functions import SparseConv2dFunction

from typing import Union, Tuple

__all__ = ["SparseConv2d", "DenseConv2d"]


class SparseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, weight, bias, mask):
        super(SparseConv2d, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(weight.clone(), requires_grad=False)
        self.mask = mask.clone()
        self.dense_weight_placeholder = nn.Parameter(torch.empty(size=self.weight.size()))
        self.dense_weight_placeholder.is_placeholder = True

        self.weight.dense = self.dense_weight_placeholder
        self.weight.mask = self.mask
        self.weight.is_sparse_param = True

        if bias is None:
            self.bias = torch.zeros(size=(out_channels,))
        else:
            self.bias = nn.Parameter(bias.clone())

    def forward(self, inp):
        return SparseConv2dFunction.apply(inp, self.weight, self.dense_weight_placeholder, self.kernel_size,
                                          self.bias, self.stride, self.padding)

    def __repr__(self):
        return "SparseConv2d({}, {}, kernel_size={}, " \
               "stride={}, padding={}, bias={})".format(self.in_channels,
                                                        self.out_channels,
                                                        self.kernel_size,
                                                        self.stride,
                                                        self.padding,
                                                        not torch.equal(self.bias, torch.zeros_like(self.bias)))

    def __str__(self):
        return self.__repr__()


class DenseConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride: Union[int, Tuple] = 1,
                 padding: Union[int, Tuple] = 0,
                 dilation: Union[int, Tuple] = 1, groups=1, bias=True,
                 padding_mode='zeros'):
        max_dilation = dilation if isinstance(dilation, int) else max(dilation)
        if max_dilation > 1:
            raise NotImplementedError("Dilation > 1 not implemented")
        if groups > 1:
            raise NotImplementedError("Groups > 1 not implemented")
        super(DenseConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                          dilation, groups, bias, padding_mode)
        self.mask = torch.ones_like(self.weight, dtype=torch.bool, device=self.weight.device)

    def forward(self, inp):
        return self._conv_forward(inp, self.weight * self.mask)

    def prune_by_threshold(self, thr):
        self.mask *= (torch.abs(self.weight) >= thr)

    def prune_by_rank(self, rank):
        if rank == 0:
            return
        weights_val = self.weight[self.mask == 1]
        sorted_abs_weights = torch.sort(torch.abs(weights_val))[0]
        thr = sorted_abs_weights[rank]
        self.prune_by_threshold(thr)

    def prune_by_pct(self, pct):
        if pct == 0:
            return
        prune_idx = int(self.num_weight * pct)
        self.prune_by_rank(prune_idx)

    def random_prune_by_pct(self, pct):
        prune_idx = int(self.num_weight * pct)
        rand = torch.rand_like(self.mask, device=self.mask.device)
        rand_val = rand[self.mask == 1]
        sorted_abs_rand = torch.sort(rand_val)[0]
        thr = sorted_abs_rand[prune_idx]
        self.mask *= (rand >= thr)

    @classmethod
    def from_conv2d(cls, conv2d_module: nn.Conv2d):
        new_conv2d = cls(conv2d_module.in_channels, conv2d_module.out_channels, conv2d_module.kernel_size,
                         conv2d_module.stride, conv2d_module.padding, conv2d_module.dilation, conv2d_module.groups,
                         bias=conv2d_module.bias is not None,
                         padding_mode=conv2d_module.padding_mode)

        new_conv2d.weight = nn.Parameter(conv2d_module.weight.clone())
        if conv2d_module.bias is not None:
            new_conv2d.bias = nn.Parameter(conv2d_module.bias.clone())

        return new_conv2d

    # This method will always remove zero elements, even if you wish to keep zeros in the sparse form
    def to_sparse(self):
        masked_weight = self.weight * self.mask
        mask = (masked_weight != 0.).view(self.out_channels, -1)
        weight = masked_weight.view(self.out_channels, -1).to_sparse()
        return SparseConv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, weight,
                            self.bias, mask)

    def move_data(self, device: torch.device):
        self.mask = self.mask.to(device)

    @property
    def num_weight(self):
        return torch.sum(self.mask).int().item()
