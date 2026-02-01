"""
Utility functions for active learning project
"""
import numpy as np
import tensorflow as tf
from typing import Tuple, Dict
import json
from pathlib import Path


def save_model_checkpoint(model: tf.keras.Model, filepath: str) -> None:
    """
    Save model checkpoint
    
    Args:
        model: Keras model to save
        filepath: Path to save model
    """
    model.save(filepath)
    print(f"Model saved to {filepath}")


def load_model_checkpoint(filepath: str) -> tf.keras.Model:
    """
    Load model checkpoint
    
    Args:
        filepath: Path to saved model
        
    Returns:
        Loaded Keras model
    """
    model = tf.keras.models.load_model(filepath)
    print(f"Model loaded from {filepath}")
    return model


def save_session_state(state_dict: Dict, filepath: str) -> None:
    """
    Save session state to file
    
    Args:
        state_dict: Dictionary of session state
        filepath: Path to save state
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable_state = {}
    for key, value in state_dict.items():
        if isinstance(value, np.ndarray):
            serializable_state[key] = value.tolist()
        elif isinstance(value, (list, dict, str, int, float, bool, type(None))):
            serializable_state[key] = value
        else:
            # Skip non-serializable objects like models
            continue
    
    with open(filepath, 'w') as f:
        json.dump(serializable_state, f, indent=2)
    print(f"Session state saved to {filepath}")


def calculate_class_distribution(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """
    Calculate class distribution
    
    Args:
        labels: Array of class labels
        num_classes: Number of classes
        
    Returns:
        Array of class counts
    """
    return np.bincount(labels.flatten(), minlength=num_classes)


def compute_label_efficiency(
    labeled_counts: list,
    accuracies: list
) -> Tuple[float, float]:
    """
    Compute label efficiency metrics
    
    Args:
        labeled_counts: List of labeled sample counts
        accuracies: List of corresponding accuracies
        
    Returns:
        Tuple of (samples_to_target, improvement_rate)
    """
    if len(labeled_counts) < 2 or len(accuracies) < 2:
        return 0.0, 0.0
    
    # Calculate improvement rate (accuracy gain per sample)
    sample_diff = labeled_counts[-1] - labeled_counts[0]
    accuracy_diff = accuracies[-1] - accuracies[0]
    
    improvement_rate = accuracy_diff / sample_diff if sample_diff > 0 else 0.0
    
    # Estimate samples needed to reach target accuracy (e.g., 0.9)
    target_accuracy = 0.9
    current_accuracy = accuracies[-1]
    
    if improvement_rate > 0 and current_accuracy < target_accuracy:
        samples_to_target = (target_accuracy - current_accuracy) / improvement_rate
    else:
        samples_to_target = 0.0
    
    return samples_to_target, improvement_rate


def get_model_info(model: tf.keras.Model) -> Dict:
    """
    Get comprehensive model information
    
    Args:
        model: Keras model
        
    Returns:
        Dictionary with model information
    """
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    return {
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "non_trainable_parameters": int(non_trainable_params),
        "num_layers": len(model.layers),
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
    }


def plot_uncertainty_distribution(uncertainties: np.ndarray, save_path: str = None):
    """
    Plot distribution of uncertainty scores
    
    Args:
        uncertainties: Array of uncertainty scores
        save_path: Optional path to save plot
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(uncertainties, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Uncertainty Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Uncertainty Scores', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_unc = np.mean(uncertainties)
    median_unc = np.median(uncertainties)
    ax.axvline(mean_unc, color='red', linestyle='--', label=f'Mean: {mean_unc:.3f}')
    ax.axvline(median_unc, color='green', linestyle='--', label=f'Median: {median_unc:.3f}')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
