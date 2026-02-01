# System Architecture

This document provides a detailed technical overview of the Active Learning Classifier system architecture.

## Table of Contents
- [Overview](#overview)
- [System Components](#system-components)
- [Data Flow](#data-flow)
- [Model Architecture](#model-architecture)
- [Active Learning Pipeline](#active-learning-pipeline)
- [Design Decisions](#design-decisions)

---

## Overview

The Active Learning Classifier is designed as a modular, production-ready system for efficient image classification with minimal labeled data. The architecture follows MLOps best practices with clear separation of concerns.

### Key Design Principles
1. **Modularity**: Each component has a single responsibility
2. **Configurability**: Centralized configuration management
3. **Testability**: All components are unit-testable
4. **Extensibility**: Easy to add new query strategies or models
5. **Observability**: Comprehensive metrics and logging

---

## System Components

### 1. Data Pipeline (`data_preprocessing.py`)

**Responsibilities:**
- Load and preprocess CIFAR-10 dataset
- Split data into labeled/unlabeled pools
- Apply data augmentation
- Normalize pixel values

**Key Functions:**
```python
load_data(initial_labeled_ratio: float) -> Tuple
create_augmented_dataset(x_train, y_train, augmentation_factor: int)
```

**Data Augmentation Techniques:**
- Random rotations (±15°)
- Random shifts (10% horizontal/vertical)
- Random zoom (90-110%)
- Random horizontal flips

### 2. Model Architecture (`model.py`)

**Responsibilities:**
- Build CNN with residual connections
- Configure optimizer and loss function
- Support various architectural configurations

**Architecture Components:**
- **Residual Block**: Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → Add → ReLU
- **Global Average Pooling**: Reduces spatial dimensions
- **Dense Layers**: Final classification with dropout

**Configuration Parameters:**
```python
input_shape: Tuple[int, int, int] = (32, 32, 3)
num_classes: int = 10
initial_filters: int = 32
learning_rate: float = 0.001
dropout_rate: float = 0.3
l2_regularization: float = 0.0001
use_batch_norm: bool = True
```

### 3. Active Learning Module (`active_learning.py`)

**Responsibilities:**
- Implement multiple query strategies
- Select most informative samples
- Calculate sample diversity metrics

**Query Strategies:**

#### Uncertainty Sampling
```python
score = 1 - max(predictions)  # Lower confidence = higher score
```

#### Margin Sampling
```python
sorted_probs = sort(predictions)
score = sorted_probs[-1] - sorted_probs[-2]  # Smaller margin = higher uncertainty
```

#### Entropy Sampling
```python
score = -sum(p * log(p) for p in predictions)  # Higher entropy = more uncertain
```

#### BALD (Bayesian Active Learning by Disagreement)
```python
# Use MC Dropout for multiple predictions
predictions = [model.predict(x, training=True) for _ in range(T)]
epistemic_uncertainty = mutual_information(predictions)
```

### 4. Configuration Management (`config.py`)

**Responsibilities:**
- Centralize all configuration
- Type-safe configuration classes
- Easy experimentation

**Configuration Hierarchy:**
```
AppConfig
├── ModelConfig
├── TrainingConfig
└── ActiveLearningConfig
```

### 5. Utilities (`utils.py`)

**Responsibilities:**
- Visualization functions
- Metric calculations
- Helper functions

---

## Data Flow

### Training Loop
```
┌─────────────────┐
│ Labeled Pool    │
│ (x_labeled,     │
│  y_labeled)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data            │
│ Augmentation    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Training  │
│ (fit with       │
│  callbacks)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Trained Model   │
└─────────────────┘
```

### Active Learning Loop
```
┌─────────────────┐
│ Unlabeled Pool  │
│ (x_unlabeled)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Inference │
│ (predict_proba) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Query Strategy  │
│ (select most    │
│  informative)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Human           │
│ Annotation      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Update Pools    │
│ labeled +=      │
│ unlabeled -=    │
└─────────────────┘
```

---

## Model Architecture

### Detailed Layer Configuration

```
Input Layer (32, 32, 3)
│
├─ Residual Block 1
│  ├─ Conv2D(32, 3x3, padding='same') + L2
│  ├─ BatchNormalization
│  ├─ ReLU
│  ├─ Conv2D(32, 3x3, padding='same') + L2
│  ├─ BatchNormalization
│  └─ Add (skip connection) + ReLU
│
├─ MaxPooling2D(2x2)
│
├─ Residual Block 2
│  ├─ Conv2D(64, 3x3, padding='same') + L2
│  ├─ BatchNormalization
│  ├─ ReLU
│  ├─ Conv2D(64, 3x3, padding='same') + L2
│  ├─ BatchNormalization
│  ├─ Conv2D(64, 1x1) [shortcut projection]
│  └─ Add (skip connection) + ReLU
│
├─ MaxPooling2D(2x2)
│
├─ Residual Block 3
│  ├─ Conv2D(128, 3x3, padding='same') + L2
│  ├─ BatchNormalization
│  ├─ ReLU
│  ├─ Conv2D(128, 3x3, padding='same') + L2
│  ├─ BatchNormalization
│  ├─ Conv2D(128, 1x1) [shortcut projection]
│  └─ Add (skip connection) + ReLU
│
├─ GlobalAveragePooling2D
│  (Reduces to 128 features)
│
├─ Dense(128) + L2
├─ ReLU
├─ Dropout(0.3)
│
└─ Dense(10, activation='softmax')
```

### Parameters Count
- **Total params**: ~1.2M
- **Trainable params**: ~1.2M
- **Non-trainable params**: 0

---

## Active Learning Pipeline

### 1. Initialization
```python
# Start with small labeled set (10% of data)
x_labeled, y_labeled, x_unlabeled = load_data(initial_labeled_ratio=0.1)
model = build_model(...)
```

### 2. Training Phase
```python
# Train on current labeled data
model.fit(
    x_labeled, y_labeled,
    validation_split=0.2,
    callbacks=[EarlyStopping, ReduceLROnPlateau]
)
```

### 3. Query Phase
```python
# Select most informative samples
indices = select_uncertain_samples(
    model, x_unlabeled,
    batch_size=10,
    strategy="uncertainty"
)
```

### 4. Annotation Phase
```python
# Present samples to human annotator
samples_to_label = x_unlabeled[indices]
labels = get_human_labels(samples_to_label)  # Via Streamlit UI
```

### 5. Update Phase
```python
# Move newly labeled samples to labeled pool
x_labeled = np.concatenate([x_labeled, samples_to_label])
y_labeled = np.concatenate([y_labeled, labels])
x_unlabeled = np.delete(x_unlabeled, indices, axis=0)
```

### 6. Evaluation Phase
```python
# Track metrics
test_accuracy = model.evaluate(x_test, y_test)
label_efficiency = test_accuracy / len(x_labeled)
```

---

## Design Decisions

### Why Residual Connections?
- **Problem**: Deep networks suffer from vanishing gradients
- **Solution**: Skip connections allow gradients to flow directly
- **Benefit**: Train deeper, more accurate models

### Why Global Average Pooling?
- **Problem**: Flatten layers have many parameters
- **Solution**: Average each feature map to single value
- **Benefit**: Reduces overfitting, fewer parameters

### Why Batch Normalization?
- **Problem**: Internal covariate shift slows training
- **Solution**: Normalize activations within each batch
- **Benefit**: Faster convergence, regularization effect

### Why Multiple Query Strategies?
- **Problem**: No single strategy is optimal for all datasets
- **Solution**: Provide multiple strategies to choose from
- **Benefit**: Flexibility for different use cases

### Why Centralized Configuration?
- **Problem**: Magic numbers scattered throughout code
- **Solution**: Single source of truth for all settings
- **Benefit**: Easy experimentation, reproducibility

---

## Performance Considerations

### Memory Management
- **Batch Processing**: Process unlabeled data in batches
- **Data Augmentation**: On-the-fly augmentation (no storage overhead)
- **Model Checkpointing**: Save only best models

### Computational Efficiency
- **Vectorization**: Use NumPy operations over loops
- **GPU Acceleration**: Automatic with TensorFlow
- **Lazy Loading**: Load data only when needed

### Scalability
- **Horizontal Scaling**: Can distribute training across GPUs
- **Vertical Scaling**: Adjust batch size based on available memory
- **Streaming**: Support for large datasets via generators

---

## Extension Points

### Adding New Query Strategies
1. Implement strategy function in `active_learning.py`
2. Add to `strategies` dictionary
3. Update configuration options
4. Add tests

### Adding New Models
1. Implement model builder in `model.py`
2. Ensure consistent interface (input_shape, num_classes)
3. Update configuration
4. Benchmark performance

### Adding New Datasets
1. Implement loader in `data_preprocessing.py`
2. Handle dataset-specific preprocessing
3. Update configuration for dataset properties
4. Add data augmentation if needed

---

## Monitoring & Observability

### Metrics Tracked
- **Training Metrics**: Loss, accuracy per epoch
- **Validation Metrics**: Prevent overfitting
- **Test Metrics**: True performance measurement
- **AL Metrics**: Label efficiency, diversity

### Visualization
- Learning curves (accuracy/loss vs. epoch)
- Confusion matrices
- Per-class performance
- Sample diversity over time

### Logging
- Model architecture summary
- Training configuration
- Query strategy performance
- Annotation statistics

---

## Security Considerations

### Data Privacy
- No personally identifiable information (PII) collected
- Dataset stored locally
- No external API calls

### Model Security
- Input validation prevents adversarial examples
- Model checkpoints stored securely
- No model served without authentication in production

---

## Future Enhancements

### Planned Features
1. **Model Ensembles**: Combine multiple models for better uncertainty
2. **Distributed Training**: Multi-GPU support
3. **Online Learning**: Continuous model updates
4. **AutoML**: Automatic hyperparameter tuning
5. **Explainability**: SHAP/LIME for model interpretability

### Research Directions
1. **Self-Supervised Pre-training**: Better initial representations
2. **Active Feature Selection**: Select most informative features
3. **Batch Active Learning**: Select diverse batches
4. **Budget-Aware AL**: Optimize under annotation cost constraints

---

## References

1. Settles, B. (2009). Active Learning Literature Survey
2. He, K. et al. (2016). Deep Residual Learning for Image Recognition
3. Gal, Y. (2016). Uncertainty in Deep Learning (PhD Thesis)
4. Houlsby, N. et al. (2011). Bayesian Active Learning for Classification

---

**Last Updated**: February 2024
**Maintainer**: Mohamed-Saber Elguelta
