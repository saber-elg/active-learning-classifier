"""
Enhanced CNN model with modern architecture improvements
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, 
    Dense, Dropout, BatchNormalization, Add, Activation
)
from tensorflow.keras.regularizers import l2
from typing import Tuple


def residual_block(x: tf.Tensor, filters: int, l2_reg: float = 0.0001) -> tf.Tensor:
    """
    Residual block with batch normalization and skip connections
    
    Args:
        x: Input tensor
        filters: Number of filters for conv layers
        l2_reg: L2 regularization factor
        
    Returns:
        Output tensor
    """
    shortcut = x
    
    # First conv layer
    x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Second conv layer
    x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    
    # Adjust shortcut if needed
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding='same', kernel_regularizer=l2(l2_reg))(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    # Add skip connection
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x


def build_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    initial_filters: int = 32,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.3,
    l2_reg: float = 0.0001,
    use_batch_norm: bool = True
) -> tf.keras.Model:
    """
    Build an enhanced CNN with residual connections and batch normalization
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of classification categories
        initial_filters: Number of filters in first layer
        learning_rate: Learning rate for optimizer
        dropout_rate: Dropout rate for regularization
        l2_reg: L2 regularization factor
        use_batch_norm: Whether to use batch normalization

    Returns:
        Compiled Keras model
    """
    inputs = Input(shape=input_shape)
    
    # Initial conv layer
    x = Conv2D(initial_filters, (3, 3), padding='same', kernel_regularizer=l2(l2_reg))(inputs)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Residual block 1
    x = residual_block(x, initial_filters, l2_reg)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(dropout_rate)(x)
    
    # Residual block 2
    x = residual_block(x, initial_filters * 2, l2_reg)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(dropout_rate)(x)
    
    # Residual block 3
    x = residual_block(x, initial_filters * 4, l2_reg)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(dropout_rate)(x)
    
    # Global average pooling instead of flatten
    x = GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = Dense(128, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = Dropout(dropout_rate + 0.1)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax', kernel_regularizer=l2(l2_reg))(x)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
    )
    
    return model
