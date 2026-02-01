"""
Sample test file for the Active Learning classifier
"""
import pytest
import numpy as np
import tensorflow as tf
from src.model import build_model
from src.active_learning import select_uncertain_samples
from src.config import config


class TestModel:
    """Test cases for model building"""
    
    def test_model_creation(self):
        """Test that model can be created with default config"""
        model = build_model(
            input_shape=config.model.input_shape,
            num_classes=config.model.num_classes,
            initial_filters=config.model.initial_filters,
            learning_rate=config.model.learning_rate,
            dropout_rate=config.model.dropout_rate,
            l2_reg=config.model.l2_regularization,
            use_batch_norm=config.model.use_batch_norm
        )
        assert model is not None
        assert len(model.layers) > 0
    
    def test_model_output_shape(self):
        """Test that model outputs correct shape"""
        model = build_model(
            input_shape=config.model.input_shape,
            num_classes=config.model.num_classes
        )
        test_input = np.random.random((1, 32, 32, 3))
        output = model.predict(test_input, verbose=0)
        assert output.shape == (1, config.model.num_classes)
    
    def test_model_trainable(self):
        """Test that model is trainable"""
        model = build_model(
            input_shape=config.model.input_shape,
            num_classes=config.model.num_classes
        )
        assert model.trainable is True


class TestActiveLearning:
    """Test cases for active learning strategies"""
    
    def test_uncertainty_sampling(self):
        """Test uncertainty sampling returns correct number of samples"""
        model = build_model(
            input_shape=config.model.input_shape,
            num_classes=config.model.num_classes
        )
        unlabeled_data = np.random.random((100, 32, 32, 3))
        batch_size = 10
        
        indices = select_uncertain_samples(
            model, unlabeled_data, batch_size, strategy="uncertainty"
        )
        assert len(indices) == batch_size
        assert len(np.unique(indices)) == batch_size  # All unique
    
    def test_invalid_strategy(self):
        """Test that invalid strategy raises error"""
        model = build_model(
            input_shape=config.model.input_shape,
            num_classes=config.model.num_classes
        )
        unlabeled_data = np.random.random((100, 32, 32, 3))
        
        with pytest.raises(ValueError):
            select_uncertain_samples(
                model, unlabeled_data, 10, strategy="invalid_strategy"
            )


class TestConfig:
    """Test cases for configuration"""
    
    def test_config_exists(self):
        """Test that configuration is accessible"""
        assert config is not None
        assert config.model is not None
        assert config.training is not None
        assert config.active_learning is not None
    
    def test_config_values(self):
        """Test that configuration has expected values"""
        assert config.model.num_classes == 10
        assert config.model.input_shape == (32, 32, 3)
        assert 0 < config.training.validation_split < 1
