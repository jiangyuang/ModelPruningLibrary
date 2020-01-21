import torch
from torch.nn.parameter import Parameter
import itertools


class SparseFCLayer(torch.nn.Module):
    """
    Sparse matrix multiplication supports only (sparse, dense). That's why the matrix multiplication is reversed here.
    """

    def __init__(self, weights: torch.sparse.FloatTensor, biases=None, activation=None):
        super(SparseFCLayer, self).__init__()
        if not weights.is_sparse:
            raise ValueError("Left weights must be sparse")
        elif not weights.is_coalesced():
            raise ValueError("Left weights must be coalesced")

        # Dimension is reversed
        self.n_inputs = weights.size(1)
        self.n_outputs = weights.size(0)
        self._activation = activation

        self._weights = Parameter(weights)
        if biases is None:
            self._biases = Parameter(torch.Tensor(self.n_outputs, 1))
            torch.nn.init.zeros_(self._biases)
        else:
            self._biases = Parameter(biases)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        r"""Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `local_metadata.get("version", None)`.

        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.

        Arguments:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this module.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        """
        for hook in self._load_state_dict_pre_hooks.values():
            hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        local_name_params = itertools.chain(self._parameters.items(), self._buffers.items())
        local_state = {k: v.data for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]

                # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                if len(param.shape) == 0 and len(input_param.shape) == 1:
                    input_param = input_param[0]

                if input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                      'the shape in current model is {}.'
                                      .format(key, input_param.shape, param.shape))
                    continue

                if isinstance(input_param, Parameter):
                    # backwards compatibility for serialized parameters
                    input_param = input_param.data
                try:
                    # param.copy_(input_param)
                    with torch.no_grad():
                        if name in self._parameters.keys():
                            # self._parameters[name].zero_()
                            self._parameters[name].copy_(input_param)
                        elif name in self._buffers.keys():
                            # self._buffers[name].zero_()
                            self._buffers[name].copy_(input_param)
                except Exception:
                    error_msgs.append('While copying the parameter named "{}", '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}.'
                                      .format(key, param.size(), input_param.size()))
            elif strict:
                missing_keys.append(key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix):
                    input_name = key[len(prefix):]
                    input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
                    if input_name not in self._modules and input_name not in local_state:
                        unexpected_keys.append(key)

    def _sparse_masked_select_abs(self, sparse_tensor: torch.sparse.FloatTensor, thr):
        indices = sparse_tensor._indices()
        values = sparse_tensor._values()
        prune_mask = torch.abs(values) >= thr
        return torch.sparse_coo_tensor(indices=indices.masked_select(prune_mask).reshape(2, -1),
                                       values=values.masked_select(prune_mask),
                                       size=[self.n_outputs, self.n_inputs]).coalesce()

    def prune_by_threshold(self, thr):
        self._weights = Parameter(self._sparse_masked_select_abs(self._weights, thr))

    def prune_by_rank(self, rank):
        weights_val = self._weights._values()
        sorted_abs_weights = torch.sort(torch.abs(weights_val))[0]
        thr = sorted_abs_weights[rank]
        self.prune_by_threshold(thr)

    def prune_by_pct(self, pct):
        prune_idx = int(self._weights._nnz() * pct)
        self.prune_by_rank(prune_idx)

    def forward(self, inputs: torch.Tensor):
        ret = torch.sparse.addmm(self._biases, self._weights, inputs)
        return ret if self._activation is None else self._activation(ret)

    @property
    def weights(self):
        return self._weights

    @property
    def biases(self):
        return self._biases

    @property
    def activation(self):
        return self._activation

    @property
    def n_weights(self):
        return self._weights._nnz()

    def __str__(self):
        return "Sparse layer with size {} and activation {}".format((self.n_outputs, self.n_inputs), self._activation)


class DenseFCLayer(torch.nn.Module):
    def __init__(self, n_inputs=None, n_outputs=None, weights: torch.Tensor = None, use_biases=True, activation=None):
        super(DenseFCLayer, self).__init__()
        if n_inputs is not None and n_outputs is not None:
            self.n_inputs = n_inputs
            self.n_outputs = n_outputs
            self._activation = activation
            self._initial_weights = None

            self._weights = Parameter(torch.Tensor(n_inputs, n_outputs))
            self._init_weights()
            self._mask = torch.ones_like(self._weights)
            self._initial_weights = self._weights.clone()
            self.use_biases = use_biases

            if self.use_biases:
                self._biases = Parameter(torch.Tensor(n_outputs))
                self._init_biases()
        elif weights is not None:
            self.n_inputs = weights.size(0)
            self.n_outputs = weights.size(1)
            self._activation = activation
            self._initial_weights = weights

            self._weights = Parameter(weights)
            self._mask = torch.ones_like(self._weights)

            self._biases = Parameter(torch.Tensor(self.n_outputs))
            self._init_biases()
        else:
            raise ValueError("DenseFClayer class accepts either n_inputs/n_outputs or weights")

    def _init_weights(self):
        # Note the difference between init functions
        # torch.nn.init.xavier_normal_(self._weights)
        # torch.nn.init.xavier_uniform_(self._weights)
        # torch.nn.init.kaiming_normal_(self._weights)
        torch.nn.init.kaiming_uniform_(self._weights)

    def _init_biases(self):
        torch.nn.init.zeros_(self._biases)

    def prune_by_threshold(self, thr):
        self._mask *= (torch.abs(self._weights) >= thr).float()

    def prune_by_rank(self, rank):
        weights_val = self._weights[self._mask == 1]
        sorted_abs_weights = torch.sort(torch.abs(weights_val))[0]
        thr = sorted_abs_weights[rank]
        self.prune_by_threshold(thr)

    def prune_by_pct(self, pct):
        prune_idx = int(self.n_weights * pct)
        self.prune_by_rank(prune_idx)

    def prune_by_pct_taylor(self, pct):
        prune_idx = int(self.n_weights * pct)

        # by abs val
        wg = torch.abs(self._weights[self._mask == 1] * self._weights.grad[self._mask == 1])
        sorted_wg = torch.sort(wg)[0]
        thr = sorted_wg[prune_idx]
        print(thr)
        self._mask *= (torch.abs(self._weights * self._weights.grad) > thr).float()

        # by val
        # wg = self._weights[self._mask == 1] * self._weights.grad[self._mask == 1]
        # sorted_wg = torch.sort(wg)[0]
        # thr = sorted_wg[prune_idx]
        # self._mask *= (self._weights * self._weights.grad >= thr).float()

    def random_prune_by_pct(self, pct):
        prune_idx = int(self.n_weights * pct)
        rand = torch.rand(size=self._mask.size(), device=self._mask.device)
        rand_val = rand[self._mask == 1]
        sorted_abs_rand = torch.sort(rand_val)[0]
        thr = sorted_abs_rand[prune_idx]
        self._mask *= (rand >= thr).float()

    def reinitialize(self):
        self._weights = Parameter(self._initial_weights)
        self._init_biases()  # biases are reinitialized

    def to_sparse(self) -> SparseFCLayer:
        return SparseFCLayer((self._weights * self._mask).t().to_sparse(), self._biases.reshape((-1, 1)),
                             self._activation)

    @classmethod
    def from_sparse(cls, s_layer: SparseFCLayer):
        return cls(weights=s_layer.weights.t().to_dense(), activation=s_layer.activation)

    def to_device(self, device: torch.device):
        self._initial_weights = self._initial_weights.to(device)
        self._mask = self._mask.to(device)

    def forward(self, inputs: torch.Tensor, use_mask=True):
        masked_weights = self._weights
        if use_mask:
            masked_weights = self._weights * self._mask
        if self.use_biases:
            ret = torch.addmm(self._biases, inputs, masked_weights)
        else:
            ret = torch.mm(inputs, masked_weights)
        return ret if self._activation is None else self._activation(ret)

    @property
    def mask(self):
        return self._mask

    @property
    def weights(self):
        return self._weights

    @property
    def activation(self):
        return self._activation

    @property
    def n_weights(self):
        return torch.nonzero(self._mask).size(0)

    @property
    def biases(self):
        if self.use_biases:
            return self._biases
        else:
            return None

    def __str__(self):
        return "DenseFClayer with size {} and activation {}".format((self.n_inputs, self.n_outputs), self._activation)
