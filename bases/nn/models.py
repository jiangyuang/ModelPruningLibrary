import torch
from abc import ABC, abstractmethod

from bases.nn.fc_layer import DenseFCLayer
from bases.nn.sequential import DenseSequential, SparseSequential
from bases.nn.masked_conv2d import MaskedConv2d
from configs.constants import NAMES
from collections import namedtuple

from torch.nn.functional import binary_cross_entropy_with_logits

__all__ = ["MNISTModel", "FEMNISTLENETModel", "CIFAR10Model"]


class BaseModel(torch.nn.Module, ABC):
    def __init__(self, model_name: NAMES, lr, dict_layers: dict):
        super(BaseModel, self).__init__()
        self.model_name = model_name
        for layer_name, layer in dict_layers.items():
            self.add_module(layer_name, layer)

        self.lr = lr

    def load_state_dict(self, state_dict, strict=True):
        _IncompatibleKeys = namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(self)

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in missing_keys)))

        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                self.__class__.__name__, "\n\t".join(error_msgs)))
        return _IncompatibleKeys(missing_keys, unexpected_keys)

    @abstractmethod
    def forward(self, inputs) -> torch.Tensor:
        pass

    @abstractmethod
    def evaluate(self, test_iter):
        pass

    @abstractmethod
    def loss(self, inputs, labels: torch.IntTensor) -> torch.FloatTensor:
        pass

    def apply_grad(self, multiplier=1.):
        with torch.no_grad():
            for param in self.parameters():
                param.add_(-self.lr * multiplier * param.grad)  # includes both sparse and dense

    def complete_step(self, inputs, labels):
        self.zero_grad()
        loss = self.loss(inputs, labels)
        loss.backward()
        self.apply_grad()

    def prune_by_threshold(self, component: str, list_thr):
        if component.lower() == "features":
            assert hasattr(self, "features") and isinstance(self.features, torch.nn.Sequential)
            for idx, sub_sequential in enumerate(self.features):
                thr = list_thr[idx]
                if thr is not None:
                    assert isinstance(sub_sequential[0], MaskedConv2d)
                    sub_sequential[0].prune_by_threshold(thr)
        elif component.lower() == "classifier":
            assert hasattr(self, "classifier")
            for idx, submodule in enumerate(self.classifier):
                thr = list_thr[idx]
                if thr is not None:
                    submodule.prune_by_threshold(thr)
        else:
            raise ValueError("Invalid component {}".format(component))

        return self

    def prune_by_rank(self, component: str, list_rank):
        if component.lower() == "features":
            assert hasattr(self, "features") and isinstance(self.features, torch.nn.Sequential)
            for idx, sub_sequential in enumerate(self.features):
                rank = list_rank[idx]
                if rank is not None:
                    assert isinstance(sub_sequential[0], MaskedConv2d)
                    sub_sequential[0].prune_by_rank(rank)
        elif component.lower() == "classifier":
            assert hasattr(self, "classifier")
            for idx, submodule in enumerate(self.classifier):
                rank = list_rank[idx]
                if rank is not None:
                    submodule.prune_by_rank(rank)
        else:
            raise ValueError("Invalid component {}".format(component))

        return self

    def retain_by_rank(self, component: str, list_rank):
        if component.lower() == "features":
            assert hasattr(self, "features") and isinstance(self.features, torch.nn.Sequential)
            for idx, sub_sequential in enumerate(self.features):
                rank = list_rank[idx]
                if rank is not None:
                    assert isinstance(sub_sequential[0], MaskedConv2d)
                    sub_sequential[0].prune_by_rank(sub_sequential[0].n_weights - rank)
        elif component.lower() == "classifier":
            assert hasattr(self, "classifier")
            for idx, submodule in enumerate(self.classifier):
                rank = list_rank[idx]
                if rank is not None:
                    submodule.prune_by_rank(submodule.n_weights - rank)
        else:
            raise ValueError("Invalid component {}".format(component))

        return self

    def prune_by_pct(self, component: str, list_pct, n_level=1):
        if component.lower() == "features":
            assert hasattr(self, "features") and isinstance(self.features, torch.nn.Sequential)
            for idx, sub_sequential in enumerate(self.features):
                pct = list_pct[idx]
                if pct is not None:
                    assert isinstance(sub_sequential[0], MaskedConv2d)
                    for _ in range(n_level):
                        sub_sequential[0].prune_by_pct(pct)
        elif component.lower() == "classifier":
            assert hasattr(self, "classifier")
            for idx, submodule in enumerate(self.classifier):
                pct = list_pct[idx]
                if pct is not None:
                    for _ in range(n_level):
                        submodule.prune_by_pct(pct)
        else:
            raise ValueError("Invalid component {}".format(component))

        return self

    def prune_by_pct_taylor(self, component: str, list_pct, n_level=1):
        # if component.lower() == "features":
        #     assert hasattr(self, "features") and isinstance(self.features, torch.nn.Sequential)
        #     for idx, sub_sequential in enumerate(self.features):
        #         pct = list_pct[idx]
        #         if pct is not None:
        #             assert isinstance(sub_sequential[0], MaskedConv2d)
        #             for _ in range(n_level):
        #                 sub_sequential[0].prune_by_pct(pct)
        if component.lower() == "classifier":
            assert hasattr(self, "classifier")
            for idx, submodule in enumerate(self.classifier):
                pct = list_pct[idx]
                if pct is not None:
                    for _ in range(n_level):
                        submodule.prune_by_pct_taylor(pct)
        else:
            raise ValueError("Invalid component {}".format(component))

        return self

    def random_prune_by_pct(self, component: str, list_pct, n_level=1):
        if component.lower() == "features":
            assert hasattr(self, "features") and isinstance(self.features, torch.nn.Sequential)
            for idx, sub_sequential in enumerate(self.features):
                pct = list_pct[idx]
                if pct is not None:
                    assert isinstance(sub_sequential[0], MaskedConv2d)
                    for _ in range(n_level):
                        sub_sequential[0].random_prune_by_pct(pct)
        elif component.lower() == "classifier":
            assert hasattr(self, "classifier")
            for idx, submodule in enumerate(self.classifier):
                pct = list_pct[idx]
                if pct is not None:
                    for _ in range(n_level):
                        submodule.random_prune_by_pct(pct)
        else:
            raise ValueError("Invalid component {}".format(component))

        return self

    def reinitialize(self):
        """
        Only dense layers
        """
        assert hasattr(self, "classifier") and isinstance(self.classifier, DenseSequential)
        for submodule in self.classifier:
            submodule.reinitialize()

    def remove_mask(self):
        """
        Only CNN layers
        """
        assert hasattr(self, "features") and isinstance(self.features, torch.nn.Sequential)
        dict_modules = {}
        for name, module in self.named_children():
            if name == "features":
                new_features_list = []
                for sub_sequential in module:
                    assert isinstance(sub_sequential[0], MaskedConv2d)
                    new_sub_sequential_list = [sub_sequential[0].to_conv2d()]
                    for item in sub_sequential[1:]:
                        new_sub_sequential_list.append(item)
                    new_features_list.append(torch.nn.Sequential(*new_sub_sequential_list))
                dict_modules[name] = torch.nn.Sequential(*new_features_list)
            else:
                dict_modules[name] = module
        return self.__class__(self.lr, dict_modules)

    def to_sparse(self):
        """
        Only (dense) FC layers
        """
        assert hasattr(self, "classifier") and isinstance(self.classifier, DenseSequential)
        dict_modules = {}
        for name, module in self.named_children():
            if name == "classifier":
                dict_modules[name] = module.to_sparse()
            else:
                dict_modules[name] = module
        return self.__class__(self.lr, dict_modules)

    def to_dense(self):
        """
        Only (sparse) FC layers
        """
        assert hasattr(self, "classifier") and isinstance(self.classifier, SparseSequential)
        dict_modules = {}
        for name, module in self.named_children():
            if name == "classifier":
                dict_modules[name] = module.to_dense()
            else:
                dict_modules[name] = module
        return self.__class__(self.lr, dict_modules)

    def rm_mask_to_sparse(self):
        return self.remove_mask().to_sparse()

    def to(self, *args, **kwargs):
        device, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            if hasattr(self, "features") and isinstance(self.features, torch.nn.Sequential):
                for sub_sequential in self.features:
                    if isinstance(sub_sequential[0], MaskedConv2d):
                        sub_sequential[0].to_device(device)
            if hasattr(self, "classifier") and isinstance(self.classifier, DenseSequential):
                for submodule in self.classifier:
                    submodule.to_device(device)
        return super(BaseModel, self).to(*args, **kwargs)

    @property
    def n_element(self):
        n_element = {}
        total = 0
        if hasattr(self, "features"):
            num_params_f = 0
            for sub_sequential in self.features:
                for name, param in sub_sequential.named_parameters():
                    if name[-6:] == "weight":
                        num_params_f += torch.nonzero(sub_sequential[0].mask).size(0)
                    else:
                        num_params_f += param.nelement()
            n_element["features"] = num_params_f
            total += num_params_f

        num_params_c = 0
        for param in self.classifier.parameters():
            if isinstance(param, torch.sparse.FloatTensor):
                num_params_c += param._nnz()
            else:
                num_params_c += param.nelement()
        n_element["classifier"] = num_params_c
        total += num_params_c

        n_element["total"] = total
        return n_element


class MNISTModel(BaseModel):
    def __init__(self, lr, dict_layers: dict = None):
        if dict_layers is None:
            classifier = DenseSequential(DenseFCLayer(784, 300, activation=torch.relu),
                                         DenseFCLayer(300, 100, activation=torch.relu),
                                         DenseFCLayer(100, 10))

            dict_layers = {"classifier": classifier}

        super(MNISTModel, self).__init__(NAMES.MNIST, lr, dict_layers)

    def forward(self, inputs):
        outputs = self.classifier(inputs)
        return outputs

    def inc_size_to(self, n1, n2):
        self.classifier[0].inc_size_to(None, n1)
        self.classifier[2].inc_size_to(n1, n2)
        self.classifier[4].inc_size_to(n2, None)

    def evaluate(self, test_iter):
        test_loss = 0
        n_correct = 0
        n_total = 0
        with torch.no_grad():
            for inputs, labels in test_iter:
                # logits = inputs
                # for layer in self.classifier:
                #     if isinstance(layer, DenseFCLayer):
                #         if p is not None:
                #             logits *= p
                #         logits = layer(logits, use_mask=False)
                logits = self(inputs)

                batch_loss = binary_cross_entropy_with_logits(logits, labels)
                test_loss += batch_loss.item()

                labels_predicted = torch.argmax(logits, dim=1)
                labels = torch.argmax(labels, dim=1)

                n_total += labels.size(0)
                n_correct += torch.sum(torch.eq(labels_predicted, labels)).item()
        return test_loss, n_correct / n_total

    def loss(self, inputs, labels) -> torch.Tensor:
        return binary_cross_entropy_with_logits(self(inputs), labels)  # .t() == no .t()


class FEMNISTLENETModel(BaseModel):
    def __init__(self, lr, dict_layers: dict = None):
        if dict_layers is None:
            classifier = DenseSequential(DenseFCLayer(784, 300, activation=torch.nn.functional.relu6),
                                         DenseFCLayer(300, 100, activation=torch.nn.functional.relu6),
                                         DenseFCLayer(100, 62))

            dict_layers = {"classifier": classifier}

        super(FEMNISTLENETModel, self).__init__(NAMES.FEMNIST, lr, dict_layers)

    def forward(self, inputs):
        outputs = inputs.view(inputs.size(0), -1)
        outputs = self.classifier(outputs)
        return outputs

    def evaluate(self, test_iter):
        test_loss = 0
        n_correct = 0
        n_total = 0
        with torch.no_grad():
            for inputs, labels in test_iter:
                logits = self(inputs)
                batch_loss = binary_cross_entropy_with_logits(logits, labels)
                test_loss += batch_loss.item()

                labels_predicted = torch.argmax(logits, dim=1)
                labels = torch.argmax(labels, dim=1)

                n_total += labels.size(0)
                n_correct += torch.sum(torch.eq(labels_predicted, labels)).item()

        return test_loss, n_correct / n_total

    def loss(self, inputs, labels) -> torch.Tensor:
        return binary_cross_entropy_with_logits(self(inputs), labels)  # .t() == no .t()


class CIFAR10Model(BaseModel):
    def __init__(self, lr=0.05, dict_layers: dict = None):
        self.config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

        if dict_layers is None:
            features = self._make_feature_layers()
            classifier = DenseSequential(DenseFCLayer(512, 512, activation=torch.relu),
                                         DenseFCLayer(512, 512, activation=torch.relu),
                                         DenseFCLayer(512, 10))

            dict_layers = {"features": features, "classifier": classifier}

        super(CIFAR10Model, self).__init__(NAMES.CIFAR10, lr, dict_layers)

    def _make_feature_layers(self):
        layers = []
        in_channels = 3
        for param in self.config:
            if param == 'M':
                layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend([MaskedConv2d(in_channels, param, kernel_size=3, padding=1),
                               # torch.nn.BatchNorm2d(param),
                               torch.nn.ReLU(inplace=True)])
                in_channels = param

        return torch.nn.Sequential(*layers)

    def forward(self, inputs):
        outputs = self.features(inputs)
        outputs = outputs.view(outputs.size(0), -1)
        outputs = self.classifier(outputs)
        return outputs

    def evaluate(self, test_loader):
        test_loss = 0
        n_correct = 0
        n_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                inputs, labels = inputs.to(device), labels.to(device)
                logits = self(inputs)
                batch_loss = binary_cross_entropy_with_logits(logits, labels)
                test_loss += batch_loss.item()

                labels_predicted = torch.argmax(logits, dim=1)
                labels = torch.argmax(labels, dim=1)

                n_total += labels.size(0)
                n_correct += torch.sum(torch.eq(labels_predicted, labels)).item()

        return test_loss, n_correct / n_total

    def loss(self, inputs, labels) -> torch.Tensor:
        return binary_cross_entropy_with_logits(self(inputs), labels)  # .t() == no .t()

    def prune_by_pct(self, component: str, list_pct, n_level=1):
        if component.lower() == "features":
            assert hasattr(self, "features") and isinstance(self.features, torch.nn.Sequential)
            assert len(list_pct) == 8
            self.features[0].prune_by_pct(list_pct[0])
            self.features[3].prune_by_pct(list_pct[1])
            self.features[6].prune_by_pct(list_pct[2])
            self.features[8].prune_by_pct(list_pct[3])
            self.features[11].prune_by_pct(list_pct[4])
            self.features[13].prune_by_pct(list_pct[5])
            self.features[16].prune_by_pct(list_pct[6])
            self.features[18].prune_by_pct(list_pct[7])

        elif component.lower() == "classifier":
            assert hasattr(self, "classifier")
            for idx, submodule in enumerate(self.classifier):
                pct = list_pct[idx]
                if pct is not None:
                    for _ in range(n_level):
                        submodule.prune_by_pct(pct)
        else:
            raise ValueError("Invalid component {}".format(component))

        return self

    def to(self, *args, **kwargs):
        device, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            if hasattr(self, "features") and isinstance(self.features, torch.nn.Sequential):
                for component in self.features:
                    if isinstance(component, MaskedConv2d):
                        component.to_device(device)
            if hasattr(self, "classifier") and isinstance(self.classifier, DenseSequential):
                for submodule in self.classifier:
                    submodule.to_device(device)
        return super(BaseModel, self).to(*args, **kwargs)

    def remove_mask(self):
        """
        Only CNN layers
        """
        assert hasattr(self, "features") and isinstance(self.features, torch.nn.Sequential)
        dict_modules = {}
        for name, module in self.named_children():
            if name == "features":
                new_features_list = []
                for component in module:
                    if isinstance(component, MaskedConv2d):
                        new_features_list.append(component.to_conv2d())
                    else:
                        new_features_list.append(component)
                    # new_sub_sequential_list = [sub_sequential[0].to_conv2d()]
                    # for item in sub_sequential[1:]:
                    #     new_sub_sequential_list.append(item)
                    # new_features_list.append(torch.nn.Sequential(*new_sub_sequential_list))
                dict_modules[name] = torch.nn.Sequential(*new_features_list)
            else:
                dict_modules[name] = module
        return self.__class__(self.lr, dict_modules)
