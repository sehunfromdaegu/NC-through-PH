# System imports
import os
import json

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Local imports
from dataloader import print_label_distribution, get_specific_dataloader
from models import resnet18
from train import train, test
from loss import PD0Loss, PD1Loss

def float_to_sci_notation(num):
    sci_str = "{:.1e}".format(num)
    coef, exp = sci_str.split('e')
    if coef.endswith(".0"):     # If the coefficient ends with ".0", remove it
        coef = coef[:-2] 
    exp_int = int(exp)
    return f"{coef}e{exp_int}"


def train_model(dataname, device, dataset_path='./data', n_classes=10, balanced=True, batch_size=64, n_statistical_analysis=40, epochs=200):
    '''
    Train a ResNet-18 model on a specified dataset and save the model state for each analysis iteration.

    Parameters:
    dataname (str): Name of the dataset used for analysis.
    device (str): The device (CPU/GPU) used for training the model.
    dataset_path (str, optional): Path to the dataset. Defaults to './data'.
    n_classes (int, optional): Number of classes in the dataset. Only the options 3 or 10 are available. Defaults to 10.
    balanced (bool, optional): Whether the dataset is balanced or not. Defaults to True.
    batch_size (int, optional): Batch size for training. Defaults to 64.
    n_statistical_analysis (int, optional): Number of statistical analyses to perform. Defaults to 40.
    epochs (int, optional): Number of epochs for training. Defaults to 200.

    '''

    directory = f'./experiments/{dataname}/trained_models'
    # Check if the directory exists
    if not os.path.exists(directory):
        # Create the directory if it doesn't exist
        os.makedirs(directory)

    filename = f"./experiments/{dataname}/{n_classes}classes_balanced({balanced})_batch({batch_size}).npz"
    print('train base models...')
   
    num_channels = 3 if dataname == 'CIFAR10' else 1

    train_loader, test_loader = get_specific_dataloader(dataset_path, dataname, n_classes, balanced, batch_size, drop_last=True)

    print_label_distribution(train_loader)
    print_label_distribution(test_loader)

    train_accuracy_history = []
    test_accuracy_history = []
    for i in tqdm(range(n_statistical_analysis)):
        model = resnet18(num_classes=n_classes, num_channels=num_channels).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(),  lr=1e-3, weight_decay=1e-4)
        for epoch in range(epochs):
            loss, train_accuracy = train(model, optimizer, criterion, train_loader)

        # Evaluate the model on the train, test set
        test_loss, test_accuracy = test(model, criterion, test_loader, return_ETF_variance=False)

        train_accuracy_history.append(train_accuracy)
        test_accuracy_history.append(test_accuracy)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, f'{directory}/({i})_{n_classes}classes_balanced({balanced})_batch({batch_size})_epochs({epochs}).pth')

    print("Train accuracy: {:.4f} +/- {:.4f}".format(np.mean(train_accuracy_history), np.std(train_accuracy_history)))
    print("Test accuracy : {:.4f} +/- {:.4f}".format(np.mean(test_accuracy_history), np.std(test_accuracy_history)))


def train_model_topology(config, dataname, device, dataset_path='./data', saved_model_location=None, save_results=True):
    '''
    Train a ResNet-18 model with topological loss. 
    This function should be used after statistical_analysis_no_topology() has been run.

    Parameters:
    config (dict): A dictionary containing configuration parameters like number of classes, batch size, epochs, etc.
    dataname (str): Name of the dataset used for analysis.
    device (str): The device (CPU/GPU) used for training the model.
    dataset_path (str): Path to the dataset.
    saved_model_location (str, optional): Path to the location of saved models for further analysis. If None, models will be trained from scratch.
    save_results (bool, optional): Flag to determine if the results should be saved. Defaults to True.

    '''

    directory = f'./experiments/{dataname}/trained_models'
    # Check if the directory exists
    if not os.path.exists(directory):
        # Create the directory if it doesn't exist
        os.makedirs(directory)

    # Unpacking the config dictionary
    n_classes = config['n_classes']
    balanced = config['balanced']
    batch_size = config['batch_size']
    epochs = config['epochs']
    additional_epochs = config['additional_epochs']
    n_statistical_analysis = config['n_statistical_analysis']
    topology_loss = config['topology_loss']
    regular_simplex = config['regular_simplex']
    topology_lr = config['topology_lr']

    num_channels = 3 if dataname == 'CIFAR10' else 1
    
    print(f'train({dataname}) with topological loss...')
    print(config)
    setting = (f'{n_classes}classes'
        f'_balanced({balanced})'
        f'_topology_loss({topology_loss})'
        f'_regularETF({regular_simplex})'
        f'_batch({batch_size})'
        f'_lr({float_to_sci_notation(topology_lr)})'
        f'_epochs({epochs})'
        f'_additional_epochs({additional_epochs})')
    # filename is the path to save the results
    filename = directory + '/' + setting + '.npz'  
    torch.manual_seed(42)

    # sanity check
    if saved_model_location is not None:
        for i in range(n_statistical_analysis):
            saved_model_name = f"({i})_{n_classes}classes_balanced({balanced})_batch({batch_size})_epochs({epochs}).pth"
            saved_model_path = saved_model_location + saved_model_name
            if not os.path.exists(saved_model_path):
                print(f"Model {saved_model_path} does not exist")
                return

    if topology_loss == 'CE':
        topology_criterion = nn.CrossEntropyLoss()
    elif topology_loss == 'PD0':
        topology_criterion = PD0Loss(pow=2., regular_simplex=regular_simplex)
    elif topology_loss == 'PD1':
        topology_criterion = PD1Loss(regular_simplex=regular_simplex)

    train_loader, test_loader = get_specific_dataloader(dataset_path, dataname, n_classes, balanced, batch_size, drop_last=True)

    print_label_distribution(train_loader)
    print_label_distribution(test_loader)
    
    test_accuracy_history = []
    test_accuracy_ETF_history = []

    for i in tqdm(range(n_statistical_analysis)):
        criterion = nn.CrossEntropyLoss()
        if saved_model_location is None:
            model = resnet18(num_classes=n_classes, num_channels=num_channels).to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            for epoch in range(epochs):
                loss, train_accuracy = train(model, optimizer, criterion, train_loader)
        else:
            saved_model_name = f"({i})_{n_classes}classes_balanced({balanced})_batch({batch_size})_epochs({epochs}).pth"
            state_dict = torch.load(f"{saved_model_location}" + saved_model_name)
            model = resnet18(num_classes=n_classes, num_channels=num_channels).to(device)
            model.load_state_dict(state_dict['model_state_dict'])
            loss, train_accuracy = test(model, criterion, train_loader, return_ETF_variance=False)
        
        # Evaluate the model on the test set
        test_loss, test_accuracy = test(model, criterion, test_loader, return_ETF_variance=False)

        # Topology optimization
        topology_optimizer = optim.Adam(model.parameters(), lr=topology_lr, weight_decay=topology_lr*0.1)
        if topology_loss == 'CE':
            topology_optimizer = optimizer

        for epoch in range(additional_epochs):
            train_loss, train_accuracy_ETF = train(model, topology_optimizer, topology_criterion, train_loader, return_ETF_variance=False)
            test_loss, test_accuracy_ETF = test(model, topology_criterion, test_loader, return_ETF_variance=False)

        torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': topology_optimizer.state_dict(),
        },  f'{directory}/({i})_{setting}.pth' )

        test_accuracy_history.append(test_accuracy)
        test_accuracy_ETF_history.append(test_accuracy_ETF)

    
    def calculate_stats(values):
        mean = np.mean(values)
        std = np.std(values)
        return mean, std

    test_accuracy_mean, test_accuracy_std = calculate_stats(test_accuracy_history)
    test_accuracy_ETF_mean, test_accuracy_ETF_std = calculate_stats(test_accuracy_ETF_history)

    print(f"Test accuracy: {test_accuracy_mean:.4f} +/- {test_accuracy_std:.4f}")
    print(f"Test accuracy ETF: {test_accuracy_ETF_mean:.4f} +/- {test_accuracy_ETF_std:.4f}")

    # Save the data
    config_json = json.dumps(config)
    if save_results:
        np.savez(filename, 
                test_accuracy_history=test_accuracy_history,
                test_accuracy_ETF_history=test_accuracy_ETF_history,
                config=config_json)
        # loaded_data = np.load(filename)
        # loaded_config = json.loads(loaded_data['config'].item())
    print(' ')
