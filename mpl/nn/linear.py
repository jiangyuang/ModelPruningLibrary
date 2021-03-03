import torch
import torch.nn as nn
import torch.sparse as sparse
from ..autograd.functions import AddmmFunction

__all__ = ["SparseLinear", "DenseLinear"]


class SparseLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']

    def __init__(self, weight: sparse.FloatTensor, bias, mask):
        super(SparseLinear, self).__init__()
        if not weight.is_sparse:
            raise ValueError("Weight must be sparse")
        elif weight._nnz() > 0 and not weight.is_coalesced():
            raise ValueError("Weight must be coalesced")

        self.in_features = weight.size(1)
        self.out_features = weight.size(0)

        # in order to add to optimizer
        self.weight = nn.Parameter(weight.data.clone(), requires_grad=False)
        self.mask = mask.clone()
        # Don't move after creation to make it a leaf
        self.dense_weight_placeholder = nn.Parameter(torch.empty(size=self.weight.size(), device=self.weight.device))
        self.dense_weight_placeholder.is_placeholder = True

        # create links
        self.weight.dense = self.dense_weight_placeholder
        self.weight.mask = self.mask
        self.weight.is_sparse_param = True

        if bias is None:
            self.register_parameter('bias', None)
        else:
            assert bias.size() == torch.Size((weight.size(0), 1))
            self.bias = nn.Parameter(bias.data.clone())

    def _sparse_masked_select_abs(self, sparse_tensor: sparse.FloatTensor, thr):
        indices = sparse_tensor._indices()
        values = sparse_tensor._values()
        prune_mask = torch.abs(values) >= thr
        return torch.sparse_coo_tensor(indices=indices.masked_select(prune_mask).reshape(2, -1),
                                       values=values.masked_select(prune_mask),
                                       size=[self.out_features, self.in_features]).coalesce()

    def prune_by_threshold(self, thr):
        self.weight = nn.Parameter(self._sparse_masked_select_abs(self.weight, thr))

    def prune_by_rank(self, rank):
        weight_val = self.weight._values()
        sorted_abs_weight = torch.sort(torch.abs(weight_val))[0]
        thr = sorted_abs_weight[rank]
        self.prune_by_threshold(thr)

    def prune_by_pct(self, pct):
        if pct == 0:
            return
        prune_idx = int(self.weight._nnz() * pct)
        self.prune_by_rank(prune_idx)

    def move_data(self, device: torch.device):
        self.weight = self.weight.to(device)

    def forward(self, inp: torch.Tensor):
        return AddmmFunction.apply(self.bias, self.weight, self.dense_weight_placeholder, inp.t()).t()

    @property
    def num_weight(self) -> int:
        return self.weight._nnz()

    def __repr__(self):
        return "SparseLinear(in_features={}, out_features={}, bias={})".format(self.in_features, self.out_features,
                                                                               self.bias is not None)

    def __str__(self):
        return self.__repr__()


class DenseLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(DenseLinear, self).__init__(in_features, out_features, bias)
        self.mask = torch.ones_like(self.weight, dtype=torch.bool, device=self.weight.device)

    def forward(self, inp: torch.Tensor):
        return nn.functional.linear(inp, self.weight * self.mask, self.bias)

    def prune_by_threshold(self, thr):
        self.mask *= (self.weight.abs() >= thr)

    def prune_by_rank(self, rank):
        if rank == 0:
            return
        weight_val = self.weight[self.mask == 1.]
        sorted_abs_weight = weight_val.abs().sort()[0]
        thr = sorted_abs_weight[rank]
        self.prune_by_threshold(thr)

    def prune_by_pct(self, pct):
        prune_idx = int(self.num_weight * pct)
        self.prune_by_rank(prune_idx)

    def random_prune_by_pct(self, pct):
        prune_idx = int(self.num_weight * pct)
        rand = torch.rand(size=self.mask.size(), device=self.mask.device)
        rand_val = rand[self.mask == 1]
        sorted_abs_rand = rand_val.sort()[0]
        thr = sorted_abs_rand[prune_idx]
        self.mask *= (rand >= thr)

    @classmethod
    def from_linear(cls, linear_module: nn.Linear):
        new_linear = cls(linear_module.in_features, linear_module.out_features,
                         bias=linear_module.bias is not None)
        new_linear.weight = nn.Parameter(linear_module.weight.clone())
        if linear_module.bias is not None:
            new_linear.bias = nn.Parameter(linear_module.bias.clone())

        return new_linear

    # This method will always remove zero elements, even if you wish to keep zeros in the sparse form
    def to_sparse(self) -> SparseLinear:
        sparse_bias = None if self.bias is None else self.bias.reshape((-1, 1))
        masked_weight = self.weight * self.mask
        mask = masked_weight != 0.
        return SparseLinear(masked_weight.to_sparse(), sparse_bias, mask)

    def move_data(self, device: torch.device):
        self.mask = self.mask.to(device)

    def to(self, *args, **kwargs):
        device = torch._C._nn._parse_to(*args, **kwargs)[0]

        if device is not None:
            self.move_data(device)

        return super(DenseLinear, self).to(*args, **kwargs)

    @property
    def num_weight(self) -> int:
        return self.mask.sum().item()
