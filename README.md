# ModelPruningLibrary (Updated 3/3/2021)
## Plan for the Next Version
We plan to further complete ModelPruningLibrary with the following:
1. c++ implementation conv2d with groups > 1 and depthwise conv2d, as well as missing models in `torchvision.models`.
2. more optimizers as in `torch.optim`.
3. well-known pruning algorithms such as SNIP [[1]](#1).
4. we also plan to implement tools for federated learning (e.g. well-known datasets for FL).

Suggestions/comments are welcome!

## Description
This is a PyTorch-based library that implements
1. model pruning: various magnitude-based pruning algorithms (by percentage, random pruning, etc.);
2. conv2d module with **sparse kernels** as well as fully-connected module implementations;
3. SGD optimizer designed for our sparse modules;
4. two types of save-load functionalities for sparse tensors, determined automatically according to tensor's density (fraction of non-zero entries). If density < 1/32, we save value-index pairs, and otherwise, we use bitmap to save sparse tensors.

It is originally from the following paper:
- Jiang, Y., Wang, S., Ko, B. J., Lee, W. H., & Tassiulas, L. (2019). [Model pruning enables efficient federated learning on edge devices](https://arxiv.org/pdf/1909.12326.pdf). arXiv preprint arXiv:1909.12326.

When using this code for scientific publications, please kindly cite the above paper.

The library consists of the following components:
* **setup.py**: installs the c++ extension and `mpl` (model pruning library) module
* **extension**: the `extension.cpp` c++ file extends the current PyTorch implementation with **sparse kernels** (the installed module is called `sparse_conv2d`). However, please note that we only extend PyTorch's slow, cpu version of conv2d forward/backward with no groups and dilation = 1 (see PyTorch's c++ code [here](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/ConvolutionMM2d.cpp)). In other words, we do not use acceleration packages such as MKL (which are not available on Raspberry Pis on which our paper experimented). Do not compare the speed of our implementation with the acceleration packages. 
* **autograd**: the `AddmmFunction` and `SparseConv2dFunction` functions provide the forward and backward functions to our customized modules.
* **models**: this is similar to `torchvision`'s implementations ([link](https://github.com/pytorch/vision/tree/master/torchvision/models)). Note that we do not implement mnasnet, mobilenet and shufflenetv2 since they have groups > 1 in the models. We also implement popular models such as models in [leaf](https://github.com/TalwalkarLab/leaf/tree/master/models).
* **nn**: `conv2d.py` and `linear.py` implement the prunable modules and their `to_sparse` functionalities.
* **optim**: implements a compatible version of SGD optimizer.

Our code has been validated on Ubuntu 20.04. Contact me if you encounter any issues!

## Examples

### Setup Library:
```shell
sudo python3 setup.py install
```

   

### Importing and Using Model
```python3
from mpl.models import conv2

model = conv2()
print(model)
```

output:
```
Conv2(
  (features): Sequential(
    (0): DenseConv2d(1, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): DenseConv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): DenseLinear(in_features=3136, out_features=2048, bias=True)
    (1): ReLU(inplace=True)
    (2): DenseLinear(in_features=2048, out_features=62, bias=True)
  )
)
```

### Model Pruning:
```python3
import mpl.models

model = mpl.models.conv2()
print("Before pruning:")
model.calc_num_prunable_params(display=True)

print("After pruning:")
model.prune_by_pct([0.1, 0, None, 0.9])
model.calc_num_prunable_params(display=True)
```
output:
```
Before pruning:
Layer name: features.0. remaining/all: 832/832 = 1.0
Layer name: features.3. remaining/all: 51264/51264 = 1.0
Layer name: classifier.0. remaining/all: 6424576/6424576 = 1.0
Layer name: classifier.2. remaining/all: 127038/127038 = 1.0
Total: remaining/all: 6603710/6603710 = 1.0
After pruning:
Layer name: features.0. remaining/all: 752/832 = 0.9038461538461539
Layer name: features.3. remaining/all: 51264/51264 = 1.0
Layer name: classifier.0. remaining/all: 6424576/6424576 = 1.0
Layer name: classifier.2. remaining/all: 12760/127038 = 0.10044238731718069
Total: remaining/all: 6489352/6603710 = 0.9826827646883343
```
### Dense to Sparse Conversion:
```python3
from mpl.models import conv2

model = conv2()
print(model.to_sparse())
```
output:
```
Conv2(
  (features): Sequential(
    (0): SparseConv2d(1, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True)
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): SparseConv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True)
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): SparseLinear(in_features=3136, out_features=2048, bias=True)
    (1): ReLU(inplace=True)
    (2): SparseLinear(in_features=2048, out_features=62, bias=True)
  )
)
```
Note that `DenseConv2d` and `DenseLinear` layers are converted to `SparseConv2d` and `SparseLinear` layers, respectively.

### SGD Training with a Sparse Model:
```python3
from mpl.models import conv2
from mpl.optim import SGD
import torch

inp = torch.rand(size=(10, 1, 28, 28))
model = conv2().to_sparse()
optimizer = SGD(model.parameters(), lr=0.01)
optimizer.zero_grad()
model(inp).sum().backward()
optimizer.step()
```

### Save/Load a Tensor:
```python3
from mpl.utils.save_load import save, load
import torch

torch.manual_seed(0)
x = torch.randn(size=(1000, 1000))
mask = torch.rand_like(x) <= 0.5
x = (x * mask).to_sparse()
save(x, "sparse_x.pt")

x_loaded = load("sparse_x.pt")
```
Using our implementation, the size of `sparse_x.pt` file is 2.1 MB, while the default `torch.save` results in a file size of 10 MB (4.8x).

## References
<a id="1">[1]</a> 
Lee, Namhoon, Thalaiyasingam Ajanthan, and Philip HS Torr. "Snip: Single-shot network pruning based on connection sensitivity." arXiv preprint arXiv:1810.02340 (2018).
