import os
import fnmatch
import json

import torch
import torchvision
import torchvision.transforms as transforms
from utils.transforms import Flatten, OneHot
from configs.constants import MNIST, FashionMNIST, FEMNIST

from configs.constants import NAMES, get_dataset_name, get_dataset_constants


def get_data(name: str, flatten: bool = False, one_hot: bool = False):
    dataset_name = get_dataset_name(name)
    if dataset_name == NAMES.MNIST:
        trans = transforms.ToTensor()
        target_trans = None
        if flatten:
            trans = transforms.Compose([trans, Flatten()])
        if one_hot:
            target_trans = OneHot(MNIST.N_CLASSES)
        train_set = torchvision.datasets.MNIST(root='./datasets', train=True, download=True, transform=trans,
                                               target_transform=target_trans)
        test_set = torchvision.datasets.MNIST(root='./datasets', train=False, download=True, transform=trans,
                                              target_transform=target_trans)

        return train_set, test_set

    elif dataset_name == NAMES.FashionMNIST:
        trans = transforms.ToTensor()
        target_trans = None
        if flatten:
            trans = transforms.Compose([trans, Flatten()])
        if one_hot:
            target_trans = OneHot(FashionMNIST.N_CLASSES)
        train_set = torchvision.datasets.FashionMNIST(root='./datasets', train=True, download=True, transform=trans,
                                                      target_transform=target_trans)
        test_set = torchvision.datasets.FashionMNIST(root='./datasets', train=False, download=True, transform=trans,
                                                     target_transform=target_trans)
        return train_set, test_set
    elif dataset_name == NAMES.FEMNIST:
        femnist_path = "datasets/FEMNIST"
        train_inputs_path = os.path.join(femnist_path, "train", "all_others_train_inputs")
        train_labels_path = os.path.join(femnist_path, "train", "all_others_train_labels")
        test_inputs_path = os.path.join(femnist_path, "test", "all_others_test_inputs")
        test_labels_path = os.path.join(femnist_path, "test", "all_others_test_labels")

        if os.path.exists(train_inputs_path) and os.path.exists(train_labels_path) and os.path.exists(
                test_inputs_path) and os.path.exists(test_labels_path):
            train_inputs, train_labels, test_inputs, test_labels = torch.load(train_inputs_path), torch.load(
                train_labels_path), torch.load(test_inputs_path), torch.load(test_labels_path)
        else:
            print("Preparing for training data")
            train_inputs, train_labels = None, None
            sample_inputs, sample_labels = None, None
            for filename in os.listdir(os.path.join(femnist_path, "train")):
                if fnmatch.fnmatch(filename, "*.json"):
                    with open(os.path.join(femnist_path, "train", filename)) as file:
                        data = json.load(file)
                        for user_name, val in data["user_data"].items():
                            # key: user name
                            # val: dict {x: x_data, y: y_data}
                            if user_name in ["f1227_38", "f1232_34"]:
                                x = torch.tensor(val["x"]).reshape((-1, 1, 28, 28))
                                y = torch.tensor(val["y"])
                                if sample_inputs is None or sample_labels is None:
                                    sample_inputs, sample_labels = x, y
                                else:
                                    sample_inputs = torch.cat((sample_inputs, x), dim=0)
                                    sample_labels = torch.cat((sample_labels, y), dim=0)
                            else:
                                x = torch.tensor(val["x"]).reshape((-1, 1, 28, 28))
                                y = torch.tensor(val["y"])

                                if train_inputs is None or train_labels is None:
                                    train_inputs, train_labels = x, y
                                else:
                                    train_inputs = torch.cat((train_inputs, x), dim=0)
                                    train_labels = torch.cat((train_labels, y), dim=0)
            torch.save(train_inputs, os.path.join(femnist_path, "train", "all_others_train_inputs"))
            torch.save(train_labels, os.path.join(femnist_path, "train", "all_others_train_labels"))
            print("Saving sample data")
            torch.save(sample_inputs, os.path.join(femnist_path, "train", "samples_train_inputs"))
            torch.save(sample_labels, os.path.join(femnist_path, "train", "samples_train_labels"))

            print("Preparing for test data")
            test_inputs, test_labels = None, None
            for filename in os.listdir(os.path.join(femnist_path, "test")):
                if fnmatch.fnmatch(filename, "*.json"):
                    with open(os.path.join(femnist_path, "test", filename)) as file:
                        data = json.load(file)
                        for user_name, val in data["user_data"].items():
                            # key: user name
                            # val: dict {x: x_data, y: y_data}
                            x = torch.tensor(val["x"]).reshape((-1, 1, 28, 28))
                            y = torch.tensor(val["y"])

                            if test_inputs is None or test_labels is None:
                                test_inputs, test_labels = x, y
                            else:
                                test_inputs = torch.cat((test_inputs, x), dim=0)
                                test_labels = torch.cat((test_labels, y), dim=0)
            torch.save(test_inputs, os.path.join(femnist_path, "test", "all_others_test_inputs"))
            torch.save(test_labels, os.path.join(femnist_path, "test", "all_others_test_labels"))
        if flatten:
            train_inputs = train_inputs.reshape((-1, 28 * 28))
            test_inputs = test_inputs.reshape((-1, 28 * 28))
        if one_hot:
            train_labels = OneHot(FEMNIST.N_CLASSES)(train_labels)
            test_labels = OneHot(FEMNIST.N_CLASSES)(test_labels)
        return (train_inputs, train_labels), (test_inputs, test_labels)

    else:
        raise ValueError("{} is not supported.".format(name))


def get_data_loader(name: str, train_batch_size, test_batch_size=200, shuffle: bool = False, flatten: bool = False,
                    one_hot: bool = False, train_set_indices=None, test_set_indices=None, n_workers=0, pin_memory=True):
    train_set, test_set = get_data(name, flatten=flatten, one_hot=one_hot)
    if train_set_indices is not None:
        train_set = torch.utils.data.Subset(train_set, train_set_indices)
    if test_set_indices is not None:
        test_set = torch.utils.data.Subset(test_set, test_set_indices)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=shuffle,
                                               num_workers=n_workers, pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False,
                                              num_workers=n_workers, pin_memory=pin_memory)

    return train_loader, test_loader


def get_train_loader(name: str, train_batch_size, shuffle: bool = False, flatten: bool = False, one_hot: bool = False,
                     train_set_indices=None, n_workers=0, pin_memory=True):
    train_loader, _ = get_data_loader(name, train_batch_size=train_batch_size, test_batch_size=1, shuffle=shuffle,
                                      flatten=flatten, one_hot=one_hot, train_set_indices=train_set_indices,
                                      test_set_indices=None, n_workers=n_workers, pin_memory=pin_memory)
    return train_loader


def get_test_loader(name: str, test_batch_size=200, shuffle: bool = False, flatten: bool = False, one_hot: bool = False,
                    test_set_indices=None, n_workers=0, pin_memory=True):
    _, test_loader = get_data_loader(name, train_batch_size=1, test_batch_size=test_batch_size, shuffle=shuffle,
                                     flatten=flatten, one_hot=one_hot, train_set_indices=None,
                                     test_set_indices=test_set_indices, n_workers=n_workers, pin_memory=pin_memory)
    return test_loader


def get_samples(name: str, n_samples, random: bool = False, flatten: bool = False, one_hot: bool = False, device=None):
    if get_dataset_name(name) == NAMES.FEMNIST:
        n_classes = get_dataset_constants(name).N_CLASSES
        n_all_samples = 389
        femnist_path = "datasets/FEMNIST"
        samples_inputs_path = os.path.join(femnist_path, "train", "samples_train_inputs")
        samples_labels_path = os.path.join(femnist_path, "train", "samples_train_labels")
        if not (os.path.exists(samples_inputs_path) and os.path.exists(samples_labels_path)):
            get_data("FEMNIST")
        sample_inputs, sample_labels = torch.load(samples_inputs_path), torch.load(os.path.join(samples_labels_path))
        if random:
            perm = torch.randperm(sample_inputs.size(0))
            sample_inputs, sample_labels = sample_inputs[perm], sample_labels[perm]
    else:
        n_classes = get_dataset_constants(name).N_CLASSES
        n_all_samples = n_classes * n_samples

        train_loader, _ = get_data_loader(name, 1, shuffle=True if random else False)

        sample_inputs = None
        sample_labels = None
        counter = [0 for _ in range(n_classes)]

        for inputs, labels in train_loader:
            n = labels.item()
            if counter[n] < n_samples:
                if sample_inputs is None:
                    sample_inputs = inputs
                else:
                    sample_inputs = torch.cat((sample_inputs, inputs), dim=0)

                if sample_labels is None:
                    sample_labels = labels
                else:
                    sample_labels = torch.cat((sample_labels, labels), dim=0)

                counter[n] += 1

            if sum(counter) == n_all_samples:
                break

    # if shuffle:
    #     pmt = torch.randperm(n_all_samples)
    #     sample_inputs = sample_inputs[pmt]
    #     sample_labels = sample_labels[pmt]

    if flatten:
        n_features = get_dataset_constants(name).N_FEATURES
        sample_inputs = torch.reshape(sample_inputs, (n_all_samples, n_features))

    if one_hot:
        sample_labels = OneHot(n_classes)(sample_labels)

    if device is None:
        return sample_inputs, sample_labels.float()
    else:
        return sample_inputs.to(device), sample_labels.float().to(device)
