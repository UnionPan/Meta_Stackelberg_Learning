# This file contains functions for loading and preparing datasets,
# including functions for applying backdoor patterns and poisoning data.

import torch
import numpy as np
from torchvision import datasets, transforms
from math import floor
import random
from torch.utils.data import Dataset

def get_datasets(data, data_dir='../data'):
    """
    Loads and returns the specified dataset.
    
    Args:
        data (str): The name of the dataset to load (e.g., 'cifar10', 'mnist').
        data_dir (str): The directory where the data is stored.
        
    Returns:
        A tuple containing the train and test datasets.
    """
    train_dataset, test_dataset = None, None

    if data == 'fmnist':
        transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.2860], std=[0.3530])])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)

    elif data == 'mnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=apply_transform)

    elif data == 'cifar10':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
        train_dataset.targets, test_dataset.targets = torch.LongTensor(train_dataset.targets), torch.LongTensor(test_dataset.targets)

    return train_dataset, test_dataset

def add_pattern_bd(x, dataset='cifar10', pattern_type='square', agent_idx=-1):
    """
    Adds a backdoor pattern to an image. This is used to create poisoned data.
    The pattern can be a simple square or a "plus" sign.
    For CIFAR-10, it can also simulate a distributed backdoor attack (DBA) where
    the pattern is split among different agents.
    
    Args:
        x (numpy.ndarray): The input image.
        dataset (str): The dataset the image belongs to.
        pattern_type (str): The type of pattern to add ('square' or 'plus').
        agent_idx (int): The index of the agent adding the pattern. Used for DBA.
        
    Returns:
        The image with the backdoor pattern.
    """
    x = np.array(x.squeeze())

    # For CIFAR-10, a "plus" pattern can be added, either fully or partially for DBA.
    if dataset == 'cifar10':
        if pattern_type == 'plus':
            start_idx = 5
            size = 6
            if agent_idx == -1: # Full pattern
                # vertical line
                for d in range(0, 3):
                    for i in range(start_idx, start_idx+size+1):
                        x[i, start_idx][d] = 0
                # horizontal line
                for d in range(0, 3):
                    for i in range(start_idx-size//2, start_idx+size//2 + 1):
                        x[start_idx+size//2, i][d] = 0
            else: # Distributed Backdoor Attack (DBA)
                # The pattern is split into 4 parts based on the agent index.
                if agent_idx % 4 == 0: # Upper part of vertical line
                    for d in range(0, 3):
                        for i in range(start_idx, start_idx+(size//2)+1):
                            x[i, start_idx][d] = 0
                elif agent_idx % 4 == 1: # Lower part of vertical line
                    for d in range(0, 3):
                        for i in range(start_idx+(size//2)+1, start_idx+size+1):
                            x[i, start_idx][d] = 0
                elif agent_idx % 4 == 2: # Left part of horizontal line
                    for d in range(0, 3):
                        for i in range(start_idx-size//2, start_idx+size//4 + 1):
                            x[start_idx+size//2, i][d] = 0
                elif agent_idx % 4 == 3: # Right part of horizontal line
                    for d in range(0, 3):
                        for i in range(start_idx-size//4+1, start_idx+size//2 + 1):
                            x[start_idx+size//2, i][d] = 0

    elif dataset == 'mnist':
        if pattern_type == 'square':
            for i in range(5, 7):
                for j in range(6, 11):
                    x[i, j] = 255

    elif dataset == 'fmnist':
        if pattern_type == 'square':
            for i in range(21, 26):
                for j in range(21, 26):
                    x[i, j] = 255
        elif pattern_type == 'plus':
            start_idx = 5
            size = 2
            if agent_idx == -1:
                # vertical line
                for i in range(start_idx, start_idx+size+1):
                    x[i, start_idx] = 255
                # horizontal line
                for i in range(start_idx-size//2, start_idx+size//2 + 1):
                    x[start_idx+size//2, i] = 255
            else: # DBA
                if agent_idx % 4 == 0:
                    for i in range(start_idx, start_idx+(size//2)+1):
                        x[i, start_idx] = 255
                elif agent_idx % 4 == 1:
                    for i in range(start_idx+(size//2), start_idx+size+1):
                        x[i, start_idx] = 255
                elif agent_idx % 4 == 2:
                    for i in range(start_idx-size//2, start_idx+size//4+1):
                        x[start_idx+size//2, i] = 255
                elif agent_idx % 4 == 3:
                    for i in range(start_idx-size//4, start_idx+size//2+1):
                        x[start_idx+size//2, i] = 255

    return x

def poison_dataset(dataset, data, base_class, target_class, poison_frac, pattern_type, data_idxs=None, poison_all=False, agent_idx=-1):
    """
    Poisons a dataset by applying a backdoor pattern to a fraction of images
    from a specific class and changing their labels to the target class.
    
    Args:
        dataset: The dataset to poison.
        data (str): The name of the dataset.
        base_class (int): The class of images to poison.
        target_class (int): The target class for the backdoor.
        poison_frac (float): The fraction of images to poison.
        pattern_type (str): The type of backdoor pattern to use.
        data_idxs (list): A list of indices to sample from for poisoning.
        poison_all (bool): If True, poisons all images in data_idxs.
        agent_idx (int): The agent index, passed to add_pattern_bd for DBA.
    """
    all_idxs = (dataset.targets == base_class).nonzero().flatten().tolist()
    if data_idxs != None:
        all_idxs = list(set(all_idxs).intersection(data_idxs))

    poison_frac = 1 if poison_all else poison_frac
    poison_idxs = random.sample(all_idxs, floor(poison_frac*len(all_idxs)))
    for idx in poison_idxs:
        clean_img = dataset.data[idx]
        bd_img = add_pattern_bd(clean_img, data, pattern_type=pattern_type, agent_idx=agent_idx)
        dataset.data[idx] = torch.tensor(bd_img)
        dataset.targets[idx] = target_class
    return

class DatasetSplit(Dataset):
    """
    A wrapper for a PyTorch Dataset that represents a split of the data for a single client.
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs
        self.targets = torch.Tensor([self.dataset.targets[idx] for idx in idxs])

    def classes(self):
        return torch.unique(self.targets)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        inp, target = self.dataset[self.idxs[item]]
        return inp, target
