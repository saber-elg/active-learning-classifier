"""
Data preprocessing and augmentation for active learning
"""
import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import tensorflow as tf
from typing import Tuple


def load_data(
    initial_labeled_ratio: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads and splits the CIFAR-10 dataset into labeled and unlabeled sets
    
    Args:
        initial_labeled_ratio: Ratio of initially labeled samples
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (x_labeled, y_labeled, x_unlabeled, x_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Normalize the dataset
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Flatten labels
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # Create labeled and unlabeled datasets
    test_size = 1.0 - initial_labeled_ratio
    x_labeled, x_unlabeled, y_labeled, y_unlabeled = train_test_split(
        x_train, y_train, test_size=test_size, random_state=random_state, stratify=y_train
    )

    return x_labeled, y_labeled, x_unlabeled, x_test, y_test


def get_data_augmentation() -> tf.keras.Sequential:
    """
    Create data augmentation pipeline for training
    
    Returns:
        Sequential model with augmentation layers
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])


def create_augmented_dataset(
    x_data: np.ndarray,
    y_data: np.ndarray,
    batch_size: int = 32
) -> tf.data.Dataset:
    """
    Create augmented TensorFlow dataset
    
    Args:
        x_data: Input images
        y_data: Labels
        batch_size: Batch size for training
        
    Returns:
        Augmented TensorFlow dataset
    """
    augmentation = get_data_augmentation()
    
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    dataset = dataset.shuffle(buffer_size=len(x_data))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

