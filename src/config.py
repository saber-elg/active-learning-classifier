"""
Configuration settings for the Active Learning Image Classifier
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    input_shape: Tuple[int, int, int] = (32, 32, 3)
    num_classes: int = 10
    initial_filters: int = 32
    learning_rate: float = 0.001
    dropout_rate: float = 0.3
    l2_regularization: float = 0.0001
    use_batch_norm: bool = True


@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 25
    batch_size: int = 32
    validation_split: float = 0.2
    early_stopping_patience: int = 5
    lr_reduction_patience: int = 3
    lr_reduction_factor: float = 0.5


@dataclass
class ActiveLearningConfig:
    """Active learning configuration"""
    initial_labeled_ratio: float = 0.1  # 10% initially labeled
    query_batch_size: int = 10
    query_strategy: str = "uncertainty"  # uncertainty, margin, entropy, bald
    max_iterations: int = 10
    enable_data_augmentation: bool = True


@dataclass
class AppConfig:
    """Main application configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    active_learning: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    
    class_names: dict = field(default_factory=lambda: {
        0: "Airplane", 1: "Automobile", 2: "Bird", 3: "Cat", 4: "Deer",
        5: "Dog", 6: "Frog", 7: "Horse", 8: "Ship", 9: "Truck"
    })


# Global configuration instance
config = AppConfig()
