"""
Enhanced active learning module with multiple query strategies
"""
import numpy as np
import tensorflow as tf
from typing import Tuple, List
from scipy.stats import entropy


def select_uncertain_samples(
    model: tf.keras.Model,
    unlabeled_data: np.ndarray,
    batch_size: int,
    strategy: str = "uncertainty"
) -> np.ndarray:
    """
    Select most informative samples using specified query strategy
    
    Args:
        model: Trained Keras model
        unlabeled_data: Unlabeled samples to query from
        batch_size: Number of samples to select
        strategy: Query strategy ('uncertainty', 'margin', 'entropy', 'bald')
        
    Returns:
        Indices of selected samples
    """
    strategies = {
        "uncertainty": _uncertainty_sampling,
        "margin": _margin_sampling,
        "entropy": _entropy_sampling,
        "bald": _bald_sampling
    }
    
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(strategies.keys())}")
    
    return strategies[strategy](model, unlabeled_data, batch_size)


def _uncertainty_sampling(
    model: tf.keras.Model,
    unlabeled_data: np.ndarray,
    batch_size: int
) -> np.ndarray:
    """
    Least confidence sampling - select samples with lowest max probability
    
    Args:
        model: Trained model
        unlabeled_data: Unlabeled samples
        batch_size: Number of samples to select
        
    Returns:
        Indices of most uncertain samples
    """
    predictions = model.predict(unlabeled_data, verbose=0)
    uncertainty = 1 - np.max(predictions, axis=1)
    uncertain_indices = np.argsort(-uncertainty)[:batch_size]
    return uncertain_indices


def _margin_sampling(
    model: tf.keras.Model,
    unlabeled_data: np.ndarray,
    batch_size: int
) -> np.ndarray:
    """
    Margin sampling - select samples with smallest difference between top 2 predictions
    
    Args:
        model: Trained model
        unlabeled_data: Unlabeled samples
        batch_size: Number of samples to select
        
    Returns:
        Indices of samples with smallest margin
    """
    predictions = model.predict(unlabeled_data, verbose=0)
    
    # Get top 2 probabilities for each sample
    top2 = np.partition(predictions, -2, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]  # Difference between top 2
    
    # Select samples with smallest margin (most uncertain)
    uncertain_indices = np.argsort(margin)[:batch_size]
    return uncertain_indices


def _entropy_sampling(
    model: tf.keras.Model,
    unlabeled_data: np.ndarray,
    batch_size: int
) -> np.ndarray:
    """
    Entropy-based sampling - select samples with highest prediction entropy
    
    Args:
        model: Trained model
        unlabeled_data: Unlabeled samples
        batch_size: Number of samples to select
        
    Returns:
        Indices of samples with highest entropy
    """
    predictions = model.predict(unlabeled_data, verbose=0)
    
    # Calculate entropy for each prediction
    prediction_entropy = entropy(predictions.T)
    
    # Select samples with highest entropy
    uncertain_indices = np.argsort(-prediction_entropy)[:batch_size]
    return uncertain_indices


def _bald_sampling(
    model: tf.keras.Model,
    unlabeled_data: np.ndarray,
    batch_size: int,
    mc_iterations: int = 10
) -> np.ndarray:
    """
    BALD (Bayesian Active Learning by Disagreement) sampling using MC Dropout
    
    Args:
        model: Trained model with dropout layers
        unlabeled_data: Unlabeled samples
        batch_size: Number of samples to select
        mc_iterations: Number of Monte Carlo forward passes
        
    Returns:
        Indices of samples with highest BALD score
    """
    # Perform multiple forward passes with dropout enabled
    mc_predictions = []
    for _ in range(mc_iterations):
        predictions = model(unlabeled_data, training=True)
        mc_predictions.append(predictions.numpy())
    
    mc_predictions = np.array(mc_predictions)  # Shape: (mc_iterations, n_samples, n_classes)
    
    # Calculate mean prediction
    mean_predictions = np.mean(mc_predictions, axis=0)
    
    # Calculate predictive entropy (uncertainty)
    predictive_entropy = entropy(mean_predictions.T)
    
    # Calculate expected entropy (aleatoric uncertainty)
    entropies = entropy(mc_predictions.transpose(0, 2, 1))  # Entropy for each MC iteration
    expected_entropy = np.mean(entropies, axis=0)
    
    # BALD score = predictive entropy - expected entropy (mutual information)
    bald_scores = predictive_entropy - expected_entropy
    
    # Select samples with highest BALD score
    uncertain_indices = np.argsort(-bald_scores)[:batch_size]
    return uncertain_indices


def calculate_diversity_score(samples: np.ndarray) -> float:
    """
    Calculate diversity score of selected samples
    
    Args:
        samples: Selected samples
        
    Returns:
        Diversity score (higher is more diverse)
    """
    if len(samples) < 2:
        return 0.0
    
    # Flatten samples
    flat_samples = samples.reshape(len(samples), -1)
    
    # Calculate pairwise distances
    distances = []
    for i in range(len(flat_samples)):
        for j in range(i + 1, len(flat_samples)):
            dist = np.linalg.norm(flat_samples[i] - flat_samples[j])
            distances.append(dist)
    
    return np.mean(distances) if distances else 0.0

