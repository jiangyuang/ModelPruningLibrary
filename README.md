# ModelPruningLibrary
## Plan for New Version
We plan to launch a new version of ModelPruningLibrary. In the new version, we will introduce the following:
1. An easy-to-use framework for model pruning, including various types of layers (CNN, FC, etc.) and models. We will introduce functions such as automatically finding the parateterized (prunbale) layers, pruning by different criterions, and more.
2. An automatic `to_sparse` method for models. This will allow the conversion from sparse model using masks to genuine sparse models using sparse matrices.
3. Implementation of popular pruning methods.
4. Implementation of popular datasets that are not currently supported by PyTorch.

Suggestions/comments are welcome!

## Description
This is a library for model pruning and sparse fully-conneted layer implementations. It can be used for federated learning as described in the following paper:
- Y. Jiang, S. Wang, B. J. Ko, W.-H. Lee, L. Tassiulas, "[Model pruning enables efficient federated learning on edge devices](https://arxiv.org/abs/1909.12326)," arXiv preprint arXiv:1909.12326 (2019).

When using this code for scientific publications, please kindly cite the above paper.

The library consists of the following components:
* **bases**: the core component. The nn subfolder contains the implementation of **fc_layer**, **masked_conv2d**, **sequential** and **models**
    * *fc_layer*: implements linear, fully-connected layers with an optional activation function. There are two types of fc layers: *DenseFCLayer* and *SparseFCLayer* (both of which can convert to each other). *DenseFCLayer* uses dense matrix multiplication and applies a mask to the weight while *SparseFCLayer* uses sparse matrices for weights. Pruning methods including random pruning, initialization-based pruning, pruning by rank/threshold etc. are implemented.
    * *masked_conv2d* implements *MaskedConv2d*, which applies mask to the kernel of a convolutional layer. It can be pruned and be converted to the PyTorch Conv2d class.
    * *sequential* implements concatenation layer for both *DenseFCLayer* and *SparseFCLayer*.
    * *models* implements the base model and models inherited from base model. The base model has pruning methods, including pruning (or retaining) by threshold, by percent, by rank, by Taylor expansion or randomly.
* **configs**: configs for datasets (image dimensions, number of training data etc.), experiments (number of clients, batch size etc.), and network (IP address and port). The default server is local (127.0.0.1).
* **utils**: necessary tools for experiments.
* **datasets.loader**: downloads and loads a specific dataset. It provides 2 transformations: one-hot and flatten. A *torch.utils.data.DataLoader* class must be converted to a *DataIterator* class to be used (see the second example). Note that on a Windows machine, *n_workers* argument must be set to 0 to avoid bugs.

Next, we will present examples, and then the server code and client code for federated learning setting.

### EXAMPLES

1. Setup library

   ```bash
   sudo python3 setup.py install
   ```

   

2. Importing and using model

```python
from bases.nn.models import MNISTModel
model = MNISTModel(lr=0.1)
```

3. Using (MNIST) data:

```python
from datasets.loader import get_data_loader
from utils.data_iterator import DataIterator

train_loader, test_loader = get_data_loader(EXP_NAME, train_batch_size=MNIST.N_TRAIN, test_batch_size=MNIST.N_TEST,
                                            shuffle=False, flatten=True, one_hot=True, n_workers=32)
train_iter = DataIterator(data_loader=train_loader, batch_size=CLIENT_BATCH_SIZE) # Must use DataIterator
test_iter = DataIterator(data_loader=test_loader, batch_size=200)
```
4. Use DataIterator:

```python
x, y = train_iter.get_next_batch() # fetch minibatches (will start over if reaches end)
for x, y in train_iter:
  do_something(x, y) # or you can iterate this way (one pass)

```
5. Model pruning:

```python
model.prune_by_pct("classifier", [0.1, 0.1, 0.1]) # The first argument must be either "features" or "classifier". The second argument is pruning rate (layer wise). Here the model's classifier is pruned by magnitude by 10%, 10%, 10%, layerwise
model.random_prune_by_pct("classifier", [0.2, 0.3, 0.4]) # Same as above, but here pruning is random, and 20%, 30% and 40% of the parameters are pruned layerwise
```
6. Dense to sparse conversion:

```python
model = model.to_sparse()
```
7. Model training:

```python
for x, y in train_iter:
    model.complete_step(x, y) # including zeroing grad, compute loss, loss.backward(), and applying grad
```

8. Evaluate model:

```python
loss, acc = model.evaluate(test_iter) # will return both loss and accuracy
```

### Server Procedure

* Read arguments from command line and setup configuration.
  * -t: pruning type, 0 (default) for no pruning, 1 for initialization-based pruning, and 2 for random pruning.
  * -l: when pruning type is 1 or 2, thie argument must be specified. It is the *level* of pruning (e.g. if pruning rate is [0.1, 0.2, 0.3], at level l there are [1 - 0.1^l, 1 - 0.2^l, 1 - 0.3^l]) of the params left.
  * -s: the seed for random numbers.
* Initializes server and wait for connections.
* When the server receives connections from all clients, it sends init message, including the initial model, local updates, batch size, etc., to all clients.
* Server aggregates params and periodically evaluate model. It terminates the training when the max iteration or time is reached.

### Client Procedure

* Connect to server and receive init message from server
* Update the model multiple times (number specified by the local updates param). Each time the gradients are applied to the params applies ). Then, upload the model to server.
* Receives the aggregated model from server, repeat util the server sends *terminate* signal.

### Sample Terminal Commands for Server/Client Experiment

1. No pruning

```bash
# Server terminal command (no pruning)
python3 experiments/MNIST/server.py
# On each client terminal
python3 experiments/MNIST/client.py
```

2. Initialization-based pruning, level 10

```bash
# Server terminal command (initialization-based pruning, level 10)
python3 experiments/MNIST/server.py -t 1 -l 10
# On each Client terminal
python3 experiments/MNIST/client.py
```
