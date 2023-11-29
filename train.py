# Importing the required libraries
import torch
import torch.nn as nn
from tqdm import tqdm
from loss import compute_ETF_variance
import numpy as np


def train(model, optimizer, criterion, train_loader, return_ETF_variance=False):
    model.train()
    device = next(model.parameters()).device
    
    correct = 0
    total_loss = 0
    ETF_variance = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, last_layer = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_loss += loss.item()

        if return_ETF_variance:
            output, last_layer = model(data)
            ETF_variance += compute_ETF_variance(last_layer, target).cpu().detach().numpy()

    total_loss /= len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    
    if return_ETF_variance:
        ETF_variance /= len(train_loader)
        return total_loss, accuracy, ETF_variance
    else:    
        return total_loss, accuracy

def test(model, criterion, test_loader, return_ETF_variance=False):
    # Evaluate the model on the test set
    model.eval()
    device = next(model.parameters()).device
    test_loss = 0
    correct = 0
    ETF_variance = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, last_layer = model(data)
            if return_ETF_variance:
                ETF_variance += compute_ETF_variance(last_layer, target).cpu().detach().numpy()
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    if return_ETF_variance:
        ETF_variance /= len(test_loader)
        return test_loss, accuracy, ETF_variance
    else:
        return test_loss, accuracy

