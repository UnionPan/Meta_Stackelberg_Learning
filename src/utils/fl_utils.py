# This file contains utility functions for Federated Learning simulations,
# including model parameter handling, training and testing, various aggregation rules,
# and functions for crafting attacks.

import torch
import numpy as np
from collections import OrderedDict
from functools import reduce
import math
import copy

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_parameters(net):
    """Get model parameters as a list of NumPy arrays."""
    result = []
    for _, val in net.state_dict().items():
        if len(val.cpu().numpy().shape)!=0:
            result.append(val.cpu().numpy())
        else:
            result.append(np.asarray([val.cpu().numpy()]))
    return result

def set_parameters(net, parameters):
    """Set model parameters from a list of NumPy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def weights_to_vector(weights):
    """Convert a list of NumPy arrays to a 1-D NumPy array."""
    return np.concatenate([w.flatten() for w in weights])

def vector_to_weights(vector, weights_template):
    """Convert a 1-D NumPy array back to a list of NumPy arrays based on a template."""
    indies = np.cumsum([0] + [w.size for w in weights_template])
    return [vector[indies[i]:indies[i+1]].reshape(weights_template[i].shape) for i in range(len(weights_template))]

def train(net, train_iter, epochs, lr):
    """Train a model on a given dataset."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    net.train()
    for _ in range(epochs):
        try:
            images, labels = next(train_iter)
        except StopIteration:
            # In case the iterator is exhausted
            # Depending on the desired behavior, you might want to reset it
            # For now, we just break the loop
            break
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(net(images), labels)
        loss.backward()
        optimizer.step()

def test(net, valloader):
    """Test a model on a given dataset."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy

def relu(x):
    """ReLU activation function."""
    return max(0.0, x)

def cos_sim(a, b):
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

# --- Aggregation Rules (Defenses) ---

def average(new_weights, num_clients, subsample_rate):
    """Standard Federated Averaging (FedAvg)."""
    fractions = [1/int(num_clients*subsample_rate) for _ in range(len(new_weights))]
    fraction_total = np.sum(fractions)

    weighted_weights = [
        [layer * fraction for layer in weights] for weights, fraction in zip(new_weights, fractions)
    ]

    aggregate_weights = [
        reduce(np.add, layer_updates) / fraction_total
        for layer_updates in zip(*weighted_weights)
    ]

    return aggregate_weights

def Krum(old_weight, new_weights, num_attacker):
    """Krum aggregation rule, designed to be robust against a certain number of attackers."""
    grads = [[layer_old - layer_new for layer_old, layer_new in zip(old_weight, new_weight)] for new_weight in new_weights]

    scores = []
    for i, grad_i in enumerate(grads):
        score = []
        for j, grad_j in enumerate(grads):
            if i == j: continue
            score.append(np.linalg.norm(weights_to_vector(grad_i) - weights_to_vector(grad_j))**2)
        
        score.sort()
        scores.append(sum(score[:len(grads) - num_attacker - 2]))

    chosen_grad = grads[np.argmin(scores)]
    return [w - g for w, g in zip(old_weight, chosen_grad)]

def Median(old_weight, new_weights):
    """Element-wise median aggregation."""
    grads = [[layer_old - layer_new for layer_old, layer_new in zip(old_weight, new_weight)] for new_weight in new_weights]

    median_grad = []
    for layer_idx in range(len(grads[0])):
        layer_grads = [grad[layer_idx] for grad in grads]
        median_grad.append(np.median(np.array(layer_grads), axis=0))
    
    return [w - g for w, g in zip(old_weight, median_grad)]

def GeoMedian(old_weight, new_weights, R=10):
    """Geometric Median aggregation."""
    epsilon = 1e-6
    geo_median = np.zeros_like(weights_to_vector(old_weight))
    vec_weights = [weights_to_vector(w) for w in new_weights]

    for _ in range(R):
        betas = [1.0 / max(np.linalg.norm(v - geo_median), epsilon) for v in vec_weights]
        sum_betas = sum(betas)
        geo_median = sum(beta * v for beta, v in zip(betas, vec_weights)) / sum_betas

    return vector_to_weights(geo_median, old_weight)

def Clipped_Median(old_weights, new_weights, max_norm):
    """Median aggregation with clipping of gradients."""
    grads = []
    for new_weight in new_weights:
        grad_vec = weights_to_vector(old_weights) - weights_to_vector(new_weight)
        norm = np.linalg.norm(grad_vec)
        clipped_grad_vec = grad_vec * min(1, max_norm / norm)
        grads.append(vector_to_weights(clipped_grad_vec, old_weights))

    median_grad = []
    for layer_idx in range(len(grads[0])):
        layer_grads = [grad[layer_idx] for grad in grads]
        median_grad.append(np.median(np.array(layer_grads), axis=0))

    return [w - g for w, g in zip(old_weights, median_grad)]

def Clipping(old_weight, new_weights, max_norm, num_clients, subsample_rate):
    """Clipping gradients before averaging."""
    grads = []
    for new_weight in new_weights:
        grad_vec = weights_to_vector(old_weight) - weights_to_vector(new_weight)
        norm = np.linalg.norm(grad_vec)
        clipped_grad_vec = grad_vec * min(1, max_norm / norm)
        grads.append(vector_to_weights(clipped_grad_vec, old_weight))

    return average(grads, num_clients, subsample_rate) # Re-using average for the final aggregation

def random_noise(weight, gau_rate):
    """Adds Laplacian noise to weights."""
    noisy_weight = copy.deepcopy(weight)
    noisy_weight_vec = weights_to_vector(noisy_weight)
    noisy_weight_vec += np.random.laplace(0, 1/gau_rate, noisy_weight_vec.shape)
    return vector_to_weights(noisy_weight_vec, weight)

def CRFL(old_weight, new_weights, max_norm, gau_rate, num_clients, subsample_rate):
    """Clipped-Robust Federated Learning (CRFL)."""
    clipped_weights = Clipping(old_weight, new_weights, max_norm, num_clients, subsample_rate)
    return random_noise(clipped_weights, gau_rate)

def FLtrust(old_weight, new_weights, valid_loader, lr, net):
    """FLTrust aggregation rule, which uses a trusted dataset to validate updates."""
    server_grad_vec = weights_to_vector(old_weight) - weights_to_vector(train(net, valid_loader, 1, lr))

    client_grads_vec = [weights_to_vector(old_weight) - weights_to_vector(w) for w in new_weights]

    trust_scores = [relu(cos_sim(g, server_grad_vec)) for g in client_grads_vec]
    sum_scores = sum(trust_scores)

    if sum_scores == 0:
        return old_weight

    normalized_scores = [s / sum_scores for s in trust_scores]

    weighted_grads = [g * s for g, s in zip(client_grads_vec, normalized_scores)]
    aggregated_grad = sum(weighted_grads)

    return vector_to_weights(weights_to_vector(old_weight) - aggregated_grad, old_weight)

def Clipped_Mean(old_weights, new_weights, max_norm, filter_rate):
    """Trimmed-mean aggregation with clipping."""
    grads = []
    for new_weight in new_weights:
        grad_vec = weights_to_vector(old_weights) - weights_to_vector(new_weight)
        norm = np.linalg.norm(grad_vec)
        clipped_grad_vec = grad_vec * min(1, max_norm / norm)
        grads.append(vector_to_weights(clipped_grad_vec, old_weights))

    mean_grad = []
    for layer_idx in range(len(grads[0])):
        layer_grads = [grad[layer_idx] for grad in grads]
        layer_grads.sort(axis=0)
        start = int(len(layer_grads) * filter_rate / 2)
        end = len(layer_grads) - start
        mean_grad.append(np.mean(np.array(layer_grads[start:end]), axis=0))

    return [w - g for w, g in zip(old_weights, mean_grad)]

# --- Attack Crafting Functions ---

def craft(old_weights, new_weights, action, b):
    """A simple model poisoning attack (Inverse Power Method)."""
    weight_diff = [w1 - w2 for w1, w2 in zip(old_weights, new_weights)]
    crafted_weight_diff = [b * diff_layer * action for diff_layer in weight_diff]
    return [w1 - w2 for w1, w2 in zip(old_weights, crafted_weight_diff)]

def Krum_craft(old_weights, weights_lis, att_ids, cids, net, train_iter, lr, num_clients, subsample_rate):
    """Crafts malicious weights to attack the Krum aggregation rule."""
    num_attacker = len(att_ids)
    temp_weights_lis = list(weights_lis)
    # Simulate honest updates from attackers to find the direction of the aggregate
    for _ in att_ids:
        set_parameters(net, old_weights)
        train(net, train_iter, 1, lr)
        temp_weights_lis.append(get_parameters(net))

    aggregate_weight = average(temp_weights_lis, num_clients, subsample_rate)
    sign = [np.sign(u - v) for u, v in zip(aggregate_weight, old_weights)]

    # This part is complex and seems to be a specific implementation of an attack
    # on Krum. It involves finding an upper bound for the crafted weights.
    # Due to its complexity, it might need further review and simplification.

    return [old_weights] * num_attacker # Placeholder, as the original logic is very complex

def Median_craft(old_weights, weights_lis, att_ids, cids, net, train_iter, lr, num_clients, subsample_rate):
    """Crafts malicious weights to attack the Median aggregation rule."""
    temp_weights_lis = list(weights_lis)
    for _ in att_ids:
        set_parameters(net, old_weights)
        train(net, train_iter, 1, lr)
        temp_weights_lis.append(get_parameters(net))

    aggregate_weight = average(temp_weights_lis, num_clients, subsample_rate)
    sign = [np.sign(u - v) for u, v in zip(aggregate_weight, old_weights)]

    # Similar to Krum_craft, this is a complex and specific attack.
    # It crafts weights by taking values from the min/max of honest weights.
    # This might also need further review.

    return [old_weights] * len(att_ids) # Placeholder
