# Importing the required libraries
import numpy as np
import torch
import torch.nn as nn
import gudhi as gd
from gudhi.wasserstein import wasserstein_distance

def pairwise_class_distances(X, labels=None):
    """
    Compute pairwise distances d(X_i,X_j), where X_i is samples of label i
    
    Parameters:
    - X: A tensor of shape (n, d) representing the dataset, where n is number of samples and d is number of features.
    - labels (optional): A tensor of shape (n,) containing labels for each sample in X. If not provided, each sample is assumed to have a unique label.
    
    Returns:
    - A tensor containing the pairwise distances between samples of each unique class label.
    """    
    labels = labels if labels is not None else torch.arange(X.shape[0])
    unique_labels = torch.unique(labels)

    if torch.isnan(X).any():
        print("NaNs in input data")

    if any([(labels == label_i).sum() == 0 or (labels == label_j).sum() == 0 for label_i in unique_labels for label_j in unique_labels]):
        print("Empty subsets found")
    
    # # Efficiently check for empty subsets
    # label_counts = torch.bincount(labels, minlength=len(unique_labels))
    # if (label_counts == 0).any():
    #     print("Empty subsets found")
    
    pairs = [(label_i, label_j) 
            for i, label_i in enumerate(unique_labels)
            for j, label_j in enumerate(unique_labels) 
            if i < j]
    
    distances = []
    for label_i, label_j in pairs:
        X_i, X_j = X[labels == label_i], X[labels == label_j]
        dists = torch.cdist(X_i, X_j, p=2)
        distances.append(dists.min()) 

    return torch.stack(distances), pairs


def PD0_deaths(X):
    if len(X) == 1:
        return torch.tensor(0.0, device=X.device)
    else:
        # Gudhi Rips complex computation
        rips = gd.RipsComplex(points=X.detach().cpu().numpy())
        st = rips.create_simplex_tree(max_dimension=0)
        st.compute_persistence()

        # Get persistence generators indices
        i = st.flag_persistence_generators()
        
        # i[0] consists of 0 dimensional persistence pairs (vertex_index, edge_vertex1_index, edge_vetrex2_index). 
        # First vertex corresponds to birth, edge(vertices) corresponds to death.
        persistence_pairs = i[0]

        # persistence_pairs[:, (1, 2)] is the set of edge indices. Each edge is a pair of vertices.
        indices = torch.tensor(persistence_pairs[:, (1, 2)], dtype=torch.long, device=X.device, requires_grad=False)
        killing_edges = X[indices[:, 0]] - X[indices[:, 1]]
        
        # Compute deaths
        deaths = torch.linalg.vector_norm(killing_edges, dim=-1, ord=2)
        return deaths
    

# This function is same as total persistence in 0 dimension
def PD0_deaths_sum(X):
    deaths = PD0_deaths(X)
    squared_deaths = torch.square(deaths)
    return squared_deaths.sum()

def sum_squared_pairwise_distances(X):
    # Compute pairwise distances using torch.cdist
    distances = torch.cdist(X, X)
    
    # Square the distances
    squared_distances = distances**2
    
    # Since squared_distances is symmetric and we don't want to count distances twice
    # (and we don't want to count the distance of a vector with itself),
    # we mask the upper triangular part (including the diagonal) and sum the remaining values.
    mask = torch.triu(torch.ones_like(squared_distances), diagonal=1)
    sum_of_squared_distances = torch.sum(squared_distances[mask.bool()])
    
    return sum_of_squared_distances




def compute_ETF_vectors(X, labels):
    '''
    The function computes the within-class mean vectors for each label.
    '''
    # Implicitly assume that the last layer is a 2D tensor of shape (n_samples, n_features)
    unique_labels = torch.unique(labels)
    mean_tensors = []
    
    for label in unique_labels:
        indices = torch.nonzero(labels == label, as_tuple=True)[0]
        mean_tensors.append(torch.mean(X[indices], dim=0))
    
    within_class_means = torch.stack(mean_tensors, dim=0)
    return within_class_means, unique_labels

def compute_ETF_variance(X, labels):
    '''
    The function implements 'TrainVariance' in the paper 
    'Limitations of Neural Collapse for Understading Generalization in Deep Learning'. Page 6.
    It measures the "level" of neural collapse.
    '''
    # Implicitly assume that the last layer is a 2D tensor of shape (n_samples, n_features)
    unique_labels = torch.unique(labels)
    n_labels = len(unique_labels)
    n_features = X.shape[1]

    # Initialize tensor to hold the mean vectors for each label
    within_class_means = torch.zeros(n_labels, n_features, device=X.device).detach()

    within_class_variance_sum = torch.tensor(0.0, device=X.device)
    # Compute the mean vectors for each label
    for i, label in enumerate(unique_labels):
        indices = (labels == label)
        within_class_means[i] = torch.mean(X[indices], dim=0)
        within_class_variance_sum += (torch.norm(X[indices] - within_class_means[i], p=2, dim=1)**2).sum()
    within_class_variance = within_class_variance_sum/len(X)

    # Compute the squared Euclidean norm between the barycenter and each within class mean
    barycenter = torch.mean(within_class_means, dim=0)
    between_class_variance_sum = torch.linalg.vector_norm(within_class_means - barycenter, ord=2, dim=1)**2
    between_class_variance = torch.mean(between_class_variance_sum)

    return within_class_variance/between_class_variance


def cosine_similarity_standard_deviation(ETF_vectors):
    # Compute the pairwise cosine similarities
    barycenter = torch.mean(ETF_vectors, dim=0)
    ETF_vectors_centered = ETF_vectors - barycenter
    cosine_sim = torch.nn.functional.cosine_similarity(ETF_vectors_centered.unsqueeze(0), ETF_vectors_centered.unsqueeze(1), dim=-1)
    
    # Remove the diagonal elements (self similarity)
    row_indices, col_indices = torch.triu_indices(row=ETF_vectors_centered.size(0), col=ETF_vectors_centered.size(0), offset=1)
    cosine_values = cosine_sim[row_indices, col_indices]
    # Compute the standard deviation
    std_dev = torch.std(cosine_values)
    
    return std_dev.item()

def NC2_metric(ETF_vectors, num_classes):
    barycenter = torch.mean(ETF_vectors, dim=0)
    ETF_centered = ETF_vectors - barycenter
    distances = torch.norm(ETF_centered, dim=1)
    avg_distance = torch.mean(distances)
    ETF_centered_scaled = ETF_centered / avg_distance

    pairwise_distances = pairwise_class_distances(ETF_centered_scaled)[0]
    matching_costs = pairwise_distances - torch.sqrt(2*num_classes/(num_classes-1))
    
    return (1/num_classes)*torch.sum(matching_costs**2)


# also considers 1d persistence with graph filtration
class PD1Loss(nn.Module):
    def __init__(self, lambda_pd=0.01, regular_simplex = True, only_within_class_variability=False):
        super(PD1Loss, self).__init__()
        self.lambda_pd = lambda_pd
        self.regular_simplex = regular_simplex
        self.only_within_class_variability = only_within_class_variability

    def forward(self, last_layer, labels):
        unique_labels = torch.unique(labels)
        topology_loss = 0.

        # Compute within-class loss(0 dimensional total persistence)
        for label in unique_labels:
            indices = (labels==label)
            X_i = last_layer[indices]
            topology_loss += PD0_deaths_sum(X_i) # test version.

        if self.only_within_class_variability:
            loss = self.lambda_pd * topology_loss
            return loss
        
        # If there are more than one labels, compute between-class loss(0 and 1 dimensional total persistence)
        # In implementation, we omit 1d features that arises within a class, and only consider 1d features that arises between classes.
        if len(unique_labels) > 1:
            pairwise_distances, label_pairs = pairwise_class_distances(last_layer, labels)
            ETF_vectors, ETF_labels = compute_ETF_vectors(last_layer, labels)
            ETF_pairwise_distances, ETF_label_pairs = pairwise_class_distances(ETF_vectors, ETF_labels)
            assert label_pairs == ETF_label_pairs, "Label pairs do not match"
                
            if self.regular_simplex:
                ETF_average_pairwise_distances = ETF_pairwise_distances.mean().item()
                topology_loss += torch.square(pairwise_distances - ETF_average_pairwise_distances).sum()
            else:
                topology_loss += torch.square(pairwise_distances - ETF_pairwise_distances).sum()
        
        loss = self.lambda_pd * topology_loss
        return loss

class PD0Loss(nn.Module):
    def __init__(self, lambda_pd = 0.01, pow=2., regular_simplex = True):
        super(PD0Loss, self).__init__()
        self.lambda_pd = lambda_pd
        self.pow = pow
        self.regular_simplex = regular_simplex

    def forward(self, last_layer, labels):
        if torch.isnan(last_layer).any():
            print(f"(PD0Loss) NaNs in last_layer")
            return torch.tensor(float('nan'))
        unique_labels = torch.unique(labels)
        # Compute persistence diagram
        PD0 = self.persistence_diagram(last_layer)
        ETF_vectors, ETF_labels = compute_ETF_vectors(last_layer, labels)
        ETF_PD0 = self.ETF_ideal_PD(ETF_vectors)

        # Compute and return Wasserstein distance as loss
        Wasserstein_distance = wasserstein_distance(PD0, ETF_PD0, order=self.pow, internal_p=1, enable_autodiff=True, keep_essential_parts=False)
        if Wasserstein_distance == 0:
            print("Wasserstein_distance is zero in PD0Loss")
        
        return self.lambda_pd * torch.pow(Wasserstein_distance, self.pow)
    
    def persistence_diagram(self, X):
        deaths = PD0_deaths(X)
        zeros = torch.zeros_like(deaths)
        persistence_diagram = torch.stack([zeros, deaths], dim=-1)
        return persistence_diagram
        
    # this is a temporary version    
    def ETF_ideal_PD(self, ETF_vectors):
        n_vectors = len(ETF_vectors)
        if self.regular_simplex:
            average_pairwise_distance = self.average_pairwise_distance(ETF_vectors)
            ETF_PD0 =  torch.tensor([[0.0000, average_pairwise_distance]] * (n_vectors - 1)).to(ETF_vectors.device).detach()
            return ETF_PD0
        else:
            pairwise_distances = torch.cdist(ETF_vectors, ETF_vectors, p=2)
            pairwise_distances_1d_array = torch.unique(pairwise_distances, sorted=True)
            deaths = pairwise_distances_1d_array[:n_vectors-1]
            ETF_PD0 = torch.tensor([[0.0000, deaths[i]] for i in range(n_vectors-1)]).to(ETF_vectors.device).detach()
            return ETF_PD0

    def average_pairwise_distance(self, X):
        # Compute the pairwise distance matrix
        dist_matrix = torch.cdist(X, X, p=2)

        # Get the upper triangular part of the matrix (excluding the diagonal)
        rows, cols = torch.triu_indices(X.shape[0], X.shape[0], offset=1)
        pairwise_distances = dist_matrix[rows, cols]

        # Calculate the average pairwise distance
        avg_pairwise_distance = pairwise_distances.mean()

        return avg_pairwise_distance.item()
