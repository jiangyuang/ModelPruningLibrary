import torch
from bases.nn.fc_layer import SparseFCLayer, DenseFCLayer


class SparseSequential(torch.nn.Sequential):
    def __init__(self, *args):
        super(SparseSequential, self).__init__(*args)
        for module in self.children():
            if not isinstance(module, SparseFCLayer):
                raise ValueError("SparseSequential object should contain only SparseFCLayer objects")

    def forward(self, inputs):
        inputs = inputs.t()
        for module in self._modules.values():
            inputs = module(inputs)
        inputs = inputs.t()
        return inputs

    def to_dense(self):
        list_modules = list(self.children())
        dense_list_modules = [DenseFCLayer.from_sparse(module) for module in list_modules]
        return DenseSequential(*dense_list_modules)


class DenseSequential(torch.nn.Sequential):
    def __init__(self, *args):
        super(DenseSequential, self).__init__(*args)
        # for module in self.children():
        #     if not isinstance(module, DenseFCLayer):
        #         raise ValueError("DenseSequential object should contain only DenseFCLayer objects")

    def to_sparse(self):
        list_modules = list(self.children())
        sparse_list_modules = [module.to_sparse() for module in list_modules]
        return SparseSequential(*sparse_list_modules)
