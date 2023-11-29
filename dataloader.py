# System imports
from collections import Counter, OrderedDict

# Third-party imports
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import torch
def get_label_configurations(n_classes, balanced):
    config = {
        3: {
            True: {0: 3000, 1: 3000, 2: 3000},
            False: {0: 3000, 1: 1000, 2: 500}
        },
        10: {
            True: {i: 3000 for i in range(10)},
            False: {i: 3000 if i < 3 else (1000 if i < 7 else 500) for i in range(10)}
        }
    }
    return config[n_classes][balanced], list(range(n_classes))

def print_label_distribution(dataloader):
    counter = Counter()
    for _, labels in dataloader:
        counter.update(labels.numpy())

    # For modified_train_labels_counter
    sorted_train_counter = OrderedDict(sorted(counter.items()))
    print(f"{sorted_train_counter}")



def get_dataloader(trainset, testset, samples_per_class, batch_size=32, drop_last=True):
    """
    Get the MNIST train and test DataLoaders with specified samples per class for trainset, 
    using the first 'n' samples. Testset will use all samples but only for the labels specified in samples_per_class.

    Args:
    - samples_per_class (dict): Dictionary with labels as keys and number of samples as values.

    Returns:
    - tuple: DataLoader for the MNIST train and test datasets with specified samples for trainset.
    """
    
    def get_class_indices(dataset, samples_per_class):
        """Get indices for each class."""
        class_indices = {label: [] for label in samples_per_class.keys()}
        for idx, (_, label) in enumerate(dataset):
            if label in samples_per_class:
                class_indices[label].append(idx)
        return class_indices

    def adjust_samples_evenly(samples_per_class, batch_size):
        """Distribute additional samples evenly among all classes."""
        total_samples = sum(samples_per_class.values())
        remaining_samples = total_samples % batch_size
        
        if drop_last and remaining_samples != 0:
            additional_samples = batch_size - remaining_samples
            samples_per_class_addition = additional_samples // len(samples_per_class)
            for label in samples_per_class:
                samples_per_class[label] += samples_per_class_addition

            # Distribute any remaining samples (if they can't be evenly divided)
            remaining_samples_after_even_distribution = additional_samples % len(samples_per_class)
            for label in list(samples_per_class.keys())[:remaining_samples_after_even_distribution]:
                samples_per_class[label] += 1

    def get_sampled_indices(class_indices, samples_per_class):
        """Get indices of the desired samples."""
        sampled_indices = []
        for label, count in samples_per_class.items():
            sampled_indices.extend(class_indices[label][:count])
        return sampled_indices

    def get_loader_from_indices(dataset, indices, batch_size, drop_last):
        """Create DataLoader from indices."""
        subset_data = Subset(dataset, indices)
        
        return DataLoader(subset_data, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    
    # Trainset processing
    class_indices_train = get_class_indices(trainset, samples_per_class)
    adjust_samples_evenly(samples_per_class, batch_size)
    sampled_indices_train = get_sampled_indices(class_indices_train, samples_per_class)
    train_loader = get_loader_from_indices(trainset, sampled_indices_train, batch_size, drop_last)

    # Testset processing: Use all samples with labels specified in samples_per_class
    labels = set(samples_per_class.keys())
    all_indices_for_labels = [idx for idx, (_, label) in enumerate(testset) if label in labels]
    test_loader = get_loader_from_indices(testset, all_indices_for_labels, batch_size, drop_last)
    
    return train_loader, test_loader



def get_specific_dataloader(dataset_path, name, n_classes, balanced, batch_size, drop_last=True):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    if name == 'MNIST':
        train_dataset = torchvision.datasets.MNIST(root=dataset_path, train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.MNIST(root=dataset_path, train=False, transform=transform, download=True)
    elif name == 'FashionMNIST':
        train_dataset = torchvision.datasets.FashionMNIST(root=dataset_path, train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.FashionMNIST(root=dataset_path, train=False, transform=transform, download=True)   
    elif name == 'CIFAR10':
        train_dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, transform=transform, download=True)
    
    train_label_sample_sizes, labels = get_label_configurations(n_classes, balanced)
    train_loader, test_loader = get_dataloader(train_dataset, test_dataset, train_label_sample_sizes, batch_size=batch_size, drop_last=True)
    return train_loader, test_loader