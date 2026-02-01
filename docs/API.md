# API Reference

Complete API documentation for the Active Learning Classifier package.

## Table of Contents
- [Model Module](#model-module)
- [Active Learning Module](#active-learning-module)
- [Data Preprocessing Module](#data-preprocessing-module)
- [Configuration Module](#configuration-module)
- [Utilities Module](#utilities-module)

---

## Model Module

### `build_model`

Build a CNN model with residual connections for image classification.

```python
def build_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    initial_filters: int = 32,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.3,
    l2_reg: float = 0.0001,
    use_batch_norm: bool = True
) -> tf.keras.Model
```

**Parameters:**
- `input_shape` (Tuple[int, int, int]): Shape of input images (height, width, channels)
- `num_classes` (int): Number of output classes
- `initial_filters` (int, optional): Number of filters in first conv layer. Default: 32
- `learning_rate` (float, optional): Learning rate for Adam optimizer. Default: 0.001
- `dropout_rate` (float, optional): Dropout rate for regularization. Default: 0.3
- `l2_reg` (float, optional): L2 regularization factor. Default: 0.0001
- `use_batch_norm` (bool, optional): Whether to use batch normalization. Default: True

**Returns:**
- `tf.keras.Model`: Compiled Keras model

**Example:**
```python
from src.model import build_model

model = build_model(
    input_shape=(32, 32, 3),
    num_classes=10,
    initial_filters=64,
    learning_rate=0.001,
    dropout_rate=0.4
)

# Train the model
model.fit(x_train, y_train, epochs=10)
```

---

## Active Learning Module

### `select_uncertain_samples`

Select most informative samples from unlabeled pool using specified query strategy.

```python
def select_uncertain_samples(
    model: tf.keras.Model,
    unlabeled_data: np.ndarray,
    batch_size: int,
    strategy: str = "uncertainty"
) -> np.ndarray
```

**Parameters:**
- `model` (tf.keras.Model): Trained Keras model for making predictions
- `unlabeled_data` (np.ndarray): Array of unlabeled samples with shape (n_samples, height, width, channels)
- `batch_size` (int): Number of samples to select
- `strategy` (str, optional): Query strategy to use. Options: "uncertainty", "margin", "entropy", "bald". Default: "uncertainty"

**Returns:**
- `np.ndarray`: Indices of selected samples (shape: (batch_size,))

**Raises:**
- `ValueError`: If strategy is not one of the supported strategies

**Example:**
```python
from src.active_learning import select_uncertain_samples

# Select 10 most uncertain samples
indices = select_uncertain_samples(
    model=trained_model,
    unlabeled_data=x_unlabeled,
    batch_size=10,
    strategy="uncertainty"
)

# Get the selected samples
samples_to_label = x_unlabeled[indices]
```

### Query Strategies

#### Uncertainty Sampling
Selects samples with the lowest maximum probability (least confident predictions).

**Formula:** `uncertainty = 1 - max(P(y|x))`

**Use Case:** General purpose, works well for most scenarios.

#### Margin Sampling
Selects samples with the smallest margin between the top two predicted classes.

**Formula:** `margin = P(y₁|x) - P(y₂|x)` where y₁ and y₂ are top-2 classes

**Use Case:** Best for datasets with similar classes or binary-like decisions.

#### Entropy Sampling
Selects samples with the highest prediction entropy.

**Formula:** `entropy = -Σ P(yᵢ|x) log P(yᵢ|x)`

**Use Case:** Multi-class problems where overall uncertainty is important.

#### BALD (Bayesian Active Learning by Disagreement)
Selects samples with maximum mutual information between predictions and model parameters.

**Formula:** `BALD = H[y|x] - E[H[y|x,θ]]`

**Use Case:** When you want to maximize information gain, more computationally expensive.

### `calculate_diversity_score`

Calculate diversity score of selected samples.

```python
def calculate_diversity_score(
    selected_samples: np.ndarray,
    all_samples: np.ndarray
) -> float
```

**Parameters:**
- `selected_samples` (np.ndarray): Samples that were selected
- `all_samples` (np.ndarray): All available samples

**Returns:**
- `float`: Diversity score between 0 and 1 (higher is more diverse)

**Example:**
```python
from src.active_learning import calculate_diversity_score

diversity = calculate_diversity_score(
    selected_samples=x_unlabeled[indices],
    all_samples=x_unlabeled
)
print(f"Selection diversity: {diversity:.3f}")
```

---

## Data Preprocessing Module

### `load_data`

Load and preprocess CIFAR-10 dataset with train/test split.

```python
def load_data(
    initial_labeled_ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
```

**Parameters:**
- `initial_labeled_ratio` (float, optional): Fraction of training data to initially label. Default: 0.1

**Returns:**
- Tuple containing:
  - `x_labeled` (np.ndarray): Initially labeled images
  - `y_labeled` (np.ndarray): Labels for initially labeled images
  - `x_unlabeled` (np.ndarray): Unlabeled images
  - `x_test` (np.ndarray): Test images
  - `y_test` (np.ndarray): Test labels

**Example:**
```python
from src.data_preprocessing import load_data

x_labeled, y_labeled, x_unlabeled, x_test, y_test = load_data(
    initial_labeled_ratio=0.1
)

print(f"Labeled samples: {len(x_labeled)}")
print(f"Unlabeled samples: {len(x_unlabeled)}")
print(f"Test samples: {len(x_test)}")
```

### `create_augmented_dataset`

Create augmented dataset with random transformations.

```python
def create_augmented_dataset(
    x_train: np.ndarray,
    y_train: np.ndarray,
    augmentation_factor: int = 2
) -> Tuple[np.ndarray, np.ndarray]
```

**Parameters:**
- `x_train` (np.ndarray): Training images
- `y_train` (np.ndarray): Training labels
- `augmentation_factor` (int, optional): How many augmented copies to create per sample. Default: 2

**Returns:**
- Tuple containing:
  - `x_augmented` (np.ndarray): Augmented images (original + augmented)
  - `y_augmented` (np.ndarray): Corresponding labels

**Augmentation Techniques:**
- Random rotation (±15 degrees)
- Random horizontal/vertical shift (10%)
- Random zoom (90-110%)
- Random horizontal flip

**Example:**
```python
from src.data_preprocessing import create_augmented_dataset

x_aug, y_aug = create_augmented_dataset(
    x_train=x_labeled,
    y_train=y_labeled,
    augmentation_factor=3
)

print(f"Original size: {len(x_labeled)}")
print(f"Augmented size: {len(x_aug)}")
```

---

## Configuration Module

### Configuration Classes

#### `ModelConfig`

Configuration for model architecture.

```python
@dataclass
class ModelConfig:
    input_shape: Tuple[int, int, int] = (32, 32, 3)
    num_classes: int = 10
    initial_filters: int = 32
    learning_rate: float = 0.001
    dropout_rate: float = 0.3
    l2_regularization: float = 0.0001
    use_batch_norm: bool = True
```

**Attributes:**
- `input_shape`: Image dimensions (height, width, channels)
- `num_classes`: Number of output classes
- `initial_filters`: Starting number of convolutional filters
- `learning_rate`: Optimizer learning rate
- `dropout_rate`: Dropout probability for regularization
- `l2_regularization`: L2 penalty coefficient
- `use_batch_norm`: Enable batch normalization

#### `TrainingConfig`

Configuration for training process.

```python
@dataclass
class TrainingConfig:
    epochs: int = 25
    batch_size: int = 32
    validation_split: float = 0.2
    early_stopping_patience: int = 5
    lr_reduction_patience: int = 3
    lr_reduction_factor: float = 0.5
```

**Attributes:**
- `epochs`: Maximum training epochs
- `batch_size`: Samples per gradient update
- `validation_split`: Fraction of training data for validation
- `early_stopping_patience`: Epochs to wait before stopping if no improvement
- `lr_reduction_patience`: Epochs to wait before reducing learning rate
- `lr_reduction_factor`: Factor to reduce learning rate by

#### `ActiveLearningConfig`

Configuration for active learning process.

```python
@dataclass
class ActiveLearningConfig:
    initial_labeled_ratio: float = 0.1
    query_batch_size: int = 10
    query_strategy: str = "uncertainty"
    max_iterations: int = 10
    enable_data_augmentation: bool = True
```

**Attributes:**
- `initial_labeled_ratio`: Initial fraction of labeled data
- `query_batch_size`: Number of samples to query per iteration
- `query_strategy`: Strategy to use ("uncertainty", "margin", "entropy", "bald")
- `max_iterations`: Maximum active learning iterations
- `enable_data_augmentation`: Enable data augmentation during training

#### `AppConfig`

Main configuration container.

```python
@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    active_learning: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    class_names: dict = field(default_factory=lambda: {...})
```

### Usage

```python
from src.config import config

# Access configuration
print(f"Learning rate: {config.model.learning_rate}")
print(f"Batch size: {config.training.batch_size}")
print(f"Query strategy: {config.active_learning.query_strategy}")

# Modify configuration
config.model.learning_rate = 0.0001
config.active_learning.query_batch_size = 20
config.training.epochs = 50

# Use in model building
model = build_model(
    input_shape=config.model.input_shape,
    num_classes=config.model.num_classes,
    learning_rate=config.model.learning_rate
)
```

---

## Utilities Module

### Visualization Functions

#### `plot_learning_curves`

Plot training and validation metrics over epochs.

```python
def plot_learning_curves(
    history: tf.keras.callbacks.History,
    metrics: List[str] = ["accuracy", "loss"]
) -> plt.Figure
```

**Parameters:**
- `history`: Training history object from model.fit()
- `metrics`: List of metrics to plot

**Returns:**
- `plt.Figure`: Matplotlib figure object

**Example:**
```python
from src.utils import plot_learning_curves

history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)
fig = plot_learning_curves(history)
plt.show()
```

#### `plot_confusion_matrix`

Create confusion matrix heatmap.

```python
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None
) -> plt.Figure
```

**Parameters:**
- `y_true`: True labels
- `y_pred`: Predicted labels
- `class_names`: List of class names for axis labels

**Returns:**
- `plt.Figure`: Matplotlib figure object

**Example:**
```python
from src.utils import plot_confusion_matrix

y_pred = model.predict(x_test).argmax(axis=1)
fig = plot_confusion_matrix(y_test, y_pred, class_names=config.class_names)
plt.show()
```

---

## Complete Workflow Example

```python
from src.config import config
from src.data_preprocessing import load_data, create_augmented_dataset
from src.model import build_model
from src.active_learning import select_uncertain_samples
from tensorflow.keras.utils import to_categorical

# 1. Load data
x_labeled, y_labeled, x_unlabeled, x_test, y_test = load_data(
    initial_labeled_ratio=0.1
)

# 2. Build model
model = build_model(
    input_shape=config.model.input_shape,
    num_classes=config.model.num_classes
)

# 3. Active learning loop
for iteration in range(config.active_learning.max_iterations):
    # Train model
    if config.active_learning.enable_data_augmentation:
        x_aug, y_aug = create_augmented_dataset(x_labeled, y_labeled)
    else:
        x_aug, y_aug = x_labeled, y_labeled
    
    y_aug_cat = to_categorical(y_aug, config.model.num_classes)
    
    history = model.fit(
        x_aug, y_aug_cat,
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        validation_split=config.training.validation_split,
        verbose=1
    )
    
    # Evaluate
    y_test_cat = to_categorical(y_test, config.model.num_classes)
    _, test_acc, _ = model.evaluate(x_test, y_test_cat)
    print(f"Iteration {iteration}: Test Accuracy = {test_acc:.4f}")
    
    # Query new samples
    if len(x_unlabeled) > 0:
        indices = select_uncertain_samples(
            model, x_unlabeled,
            batch_size=config.active_learning.query_batch_size,
            strategy=config.active_learning.query_strategy
        )
        
        # Simulate labeling (in practice, get human labels)
        new_x = x_unlabeled[indices]
        new_y = get_labels(new_x)  # Your labeling function
        
        # Update pools
        x_labeled = np.concatenate([x_labeled, new_x])
        y_labeled = np.concatenate([y_labeled, new_y])
        x_unlabeled = np.delete(x_unlabeled, indices, axis=0)

print("Active learning complete!")
```

---

## Error Handling

### Common Exceptions

**ValueError**
- Invalid query strategy specified
- Batch size larger than unlabeled pool
- Invalid configuration values

**Example:**
```python
try:
    indices = select_uncertain_samples(
        model, x_unlabeled,
        batch_size=10,
        strategy="invalid_strategy"
    )
except ValueError as e:
    print(f"Error: {e}")
    # Use default strategy
    indices = select_uncertain_samples(
        model, x_unlabeled,
        batch_size=10,
        strategy="uncertainty"
    )
```

---

## Type Hints

All functions include comprehensive type hints for better IDE support and type checking with mypy:

```python
from typing import Tuple, List
import numpy as np
import tensorflow as tf

def example_function(
    x: np.ndarray,
    y: np.ndarray,
    config: dict
) -> Tuple[tf.keras.Model, float]:
    ...
```

Run type checking:
```bash
mypy src/
```

---

## Performance Tips

1. **Batch Processing**: Process unlabeled data in batches to manage memory
2. **GPU Acceleration**: Ensure TensorFlow detects your GPU
3. **Data Augmentation**: Use on-the-fly augmentation to save memory
4. **Model Checkpointing**: Save only best models to save disk space

```python
# Check GPU availability
import tensorflow as tf
print("GPUs Available:", tf.config.list_physical_devices('GPU'))

# Enable mixed precision for faster training
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

---

**Last Updated**: February 2024  
**API Version**: 1.0.0
