# Contributing to Active Learning Classifier

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## 🎯 Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and professional in all interactions.

## 🚀 How to Contribute

### Reporting Bugs

Before creating a bug report:
1. Check the [existing issues](https://github.com/saber-elg/active-learning-classifier/issues)
2. Ensure you're using the latest version
3. Verify the issue is reproducible

When creating a bug report, include:
- Clear, descriptive title
- Steps to reproduce
- Expected vs. actual behavior
- System information (OS, Python version, TensorFlow version)
- Error messages and stack traces
- Screenshots if applicable

### Suggesting Features

Feature requests are welcome! Please:
- Check if the feature has already been suggested
- Provide a clear use case
- Explain how it aligns with the project goals
- Consider implementation complexity

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/saber-elg/active-learning-classifier.git
   cd active-learning-classifier
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

4. **Make your changes**
   - Write clean, documented code
   - Follow existing code style
   - Add tests for new functionality
   - Update documentation as needed

5. **Run tests and checks**
   ```bash
   # Run tests
   pytest tests/ -v --cov=src
   
   # Format code
   black src/ tests/ app.py
   
   # Type checking
   mypy src/
   
   # Linting
   flake8 src/ tests/ app.py
   ```

6. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add amazing new feature"
   ```
   
   Use conventional commit messages:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `test:` Test additions/changes
   - `refactor:` Code refactoring
   - `style:` Formatting changes
   - `perf:` Performance improvements

7. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then open a Pull Request on GitHub

## 📋 Development Guidelines

### Code Style

- Follow PEP 8 style guide
- Use `black` for automatic formatting
- Maximum line length: 100 characters
- Use type hints for function signatures
- Write docstrings for all public functions/classes

Example:
```python
def select_uncertain_samples(
    model: tf.keras.Model,
    unlabeled_data: np.ndarray,
    batch_size: int,
    strategy: str = "uncertainty"
) -> np.ndarray:
    """
    Select most informative samples using specified query strategy.
    
    Args:
        model: Trained Keras model
        unlabeled_data: Unlabeled samples to query from
        batch_size: Number of samples to select
        strategy: Query strategy to use
        
    Returns:
        Indices of selected samples
        
    Raises:
        ValueError: If strategy is not supported
    """
    # Implementation here
    pass
```

### Testing

- Write tests for all new functionality
- Maintain >80% code coverage
- Use descriptive test names
- Test edge cases and error conditions

Example:
```python
def test_uncertainty_sampling_returns_correct_count():
    """Test that uncertainty sampling returns requested number of samples"""
    model = build_model((32, 32, 3), 10)
    data = np.random.random((100, 32, 32, 3))
    
    result = select_uncertain_samples(model, data, batch_size=10)
    
    assert len(result) == 10
    assert len(np.unique(result)) == 10  # All unique
```

### Documentation

- Update README.md for user-facing changes
- Add docstrings for all public APIs
- Update ARCHITECTURE.md for design changes
- Include code examples in docstrings

## 🏗️ Project Architecture

When making changes, consider:
- **Modularity**: Keep components loosely coupled
- **Testability**: Write testable code
- **Configuration**: Use `config.py` for settings
- **Error Handling**: Provide clear error messages
- **Performance**: Profile before optimizing

## 📝 Commit Message Guidelines

Format:
```
<type>(<scope>): <subject>

<body>

<footer>
```

Example:
```
feat(active_learning): add BALD query strategy

Implement Bayesian Active Learning by Disagreement (BALD)
using MC Dropout for uncertainty estimation.

- Add _bald_sampling function
- Update select_uncertain_samples to support BALD
- Add tests for BALD strategy
- Update documentation

Closes #42
```

## 🔍 Review Process

All submissions require review:
1. Automated tests must pass
2. Code coverage must not decrease
3. At least one maintainer approval required
4. No merge conflicts
5. Documentation updated

## 🎓 Resources

- [TensorFlow Best Practices](https://www.tensorflow.org/guide/keras/train_and_evaluate)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [pytest Documentation](https://docs.pytest.org/)
- [Active Learning Survey Paper](https://www.cs.cmu.edu/~bsettles/pub/settles.activelearning.pdf)

## 💬 Questions?

- Open a [Discussion](https://github.com/saber-elg/active-learning-classifier/discussions)
- Reach out via email: your.email@example.com

## 🙏 Thank You!

Every contribution, no matter how small, is valued and appreciated!
