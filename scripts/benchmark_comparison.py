"""
Benchmark script to compare Active Learning vs Random Sampling
Run this to see the benefits of active learning
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.data_preprocessing import load_data
from src.model import build_model
from src.active_learning import select_uncertain_samples
from src.config import config


def train_and_evaluate(x_train, y_train, x_test, y_test, epochs=10):
    """Train model and return test accuracy"""
    model = build_model(
        config.model.input_shape,
        config.model.num_classes,
        initial_filters=config.model.initial_filters,
        learning_rate=config.model.learning_rate,
        dropout_rate=config.model.dropout_rate,
        l2_reg=config.model.l2_regularization,
        use_batch_norm=config.model.use_batch_norm
    )
    
    y_train_cat = to_categorical(y_train, config.model.num_classes)
    y_test_cat = to_categorical(y_test, config.model.num_classes)
    
    history = model.fit(
        x_train, y_train_cat,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    _, test_acc, _ = model.evaluate(x_test, y_test_cat, verbose=0)
    return test_acc, model


def active_learning_experiment(
    x_labeled, y_labeled, x_unlabeled, y_unlabeled, x_test, y_test,
    iterations=10, batch_size=100, strategy="uncertainty"
):
    """Run active learning experiment"""
    accuracies = []
    labeled_counts = []
    
    for i in range(iterations):
        print(f"AL Iteration {i+1}/{iterations}")
        
        # Train and evaluate
        test_acc, model = train_and_evaluate(x_labeled, y_labeled, x_test, y_test)
        accuracies.append(test_acc)
        labeled_counts.append(len(x_labeled))
        
        print(f"  Labeled: {len(x_labeled)}, Test Acc: {test_acc:.4f}")
        
        if len(x_unlabeled) == 0:
            break
        
        # Select uncertain samples
        query_size = min(batch_size, len(x_unlabeled))
        uncertain_indices = select_uncertain_samples(
            model, x_unlabeled, query_size, strategy=strategy
        )
        
        # Add to labeled set (simulating oracle labeling)
        x_labeled = np.vstack([x_labeled, x_unlabeled[uncertain_indices]])
        y_labeled = np.concatenate([y_labeled, y_unlabeled[uncertain_indices]])
        
        # Remove from unlabeled set
        mask = np.ones(len(x_unlabeled), dtype=bool)
        mask[uncertain_indices] = False
        x_unlabeled = x_unlabeled[mask]
        y_unlabeled = y_unlabeled[mask]
        
        del model  # Free memory
        tf.keras.backend.clear_session()
    
    return accuracies, labeled_counts


def random_sampling_experiment(
    x_labeled, y_labeled, x_unlabeled, y_unlabeled, x_test, y_test,
    iterations=10, batch_size=100
):
    """Run random sampling baseline experiment"""
    accuracies = []
    labeled_counts = []
    
    for i in range(iterations):
        print(f"Random Iteration {i+1}/{iterations}")
        
        # Train and evaluate
        test_acc, model = train_and_evaluate(x_labeled, y_labeled, x_test, y_test)
        accuracies.append(test_acc)
        labeled_counts.append(len(x_labeled))
        
        print(f"  Labeled: {len(x_labeled)}, Test Acc: {test_acc:.4f}")
        
        if len(x_unlabeled) == 0:
            break
        
        # Random selection
        query_size = min(batch_size, len(x_unlabeled))
        random_indices = np.random.choice(len(x_unlabeled), query_size, replace=False)
        
        # Add to labeled set
        x_labeled = np.vstack([x_labeled, x_unlabeled[random_indices]])
        y_labeled = np.concatenate([y_labeled, y_unlabeled[random_indices]])
        
        # Remove from unlabeled set
        mask = np.ones(len(x_unlabeled), dtype=bool)
        mask[random_indices] = False
        x_unlabeled = x_unlabeled[mask]
        y_unlabeled = y_unlabeled[mask]
        
        del model  # Free memory
        tf.keras.backend.clear_session()
    
    return accuracies, labeled_counts


def run_comparison(iterations=10, batch_size=100):
    """Run complete comparison experiment"""
    print("=" * 60)
    print("Active Learning vs Random Sampling Comparison")
    print("=" * 60)
    
    # Load data
    x_labeled, y_labeled, x_unlabeled, x_test, y_test = load_data(
        initial_labeled_ratio=0.02  # Start with only 2% labeled
    )
    
    # Keep track of initial unlabeled labels for simulation
    (x_train_full, y_train_full), _ = tf.keras.datasets.cifar10.load_data()
    x_train_full = x_train_full.astype('float32') / 255.0
    y_train_full = y_train_full.flatten()
    
    # Create unlabeled labels (for simulation)
    unlabeled_mask = np.isin(
        x_train_full.reshape(len(x_train_full), -1),
        x_labeled.reshape(len(x_labeled), -1)
    ).all(axis=1) == False
    
    y_unlabeled = y_train_full[unlabeled_mask]
    
    print(f"\nInitial Setup:")
    print(f"  Labeled: {len(x_labeled)}")
    print(f"  Unlabeled: {len(x_unlabeled)}")
    print(f"  Test: {len(x_test)}")
    print(f"  Iterations: {iterations}")
    print(f"  Batch Size: {batch_size}\n")
    
    # Run Active Learning
    print("\n" + "=" * 60)
    print("Running ACTIVE LEARNING experiment...")
    print("=" * 60)
    al_acc, al_counts = active_learning_experiment(
        x_labeled.copy(), y_labeled.copy(),
        x_unlabeled.copy(), y_unlabeled.copy(),
        x_test, y_test,
        iterations=iterations,
        batch_size=batch_size,
        strategy="uncertainty"
    )
    
    # Run Random Sampling
    print("\n" + "=" * 60)
    print("Running RANDOM SAMPLING baseline...")
    print("=" * 60)
    random_acc, random_counts = random_sampling_experiment(
        x_labeled.copy(), y_labeled.copy(),
        x_unlabeled.copy(), y_unlabeled.copy(),
        x_test, y_test,
        iterations=iterations,
        batch_size=batch_size
    )
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy comparison
    ax1.plot(al_counts, al_acc, 'b-o', linewidth=2, markersize=8, label='Active Learning')
    ax1.plot(random_counts, random_acc, 'r-s', linewidth=2, markersize=8, label='Random Sampling')
    ax1.set_xlabel('Number of Labeled Samples', fontsize=14)
    ax1.set_ylabel('Test Accuracy', fontsize=14)
    ax1.set_title('Active Learning vs Random Sampling', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Improvement over random
    improvement = [(al - rand) * 100 for al, rand in zip(al_acc, random_acc)]
    ax2.bar(range(len(improvement)), improvement, color='green', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Iteration', fontsize=14)
    ax2.set_ylabel('Accuracy Improvement (%)', fontsize=14)
    ax2.set_title('Active Learning Advantage', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('al_vs_random_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved as 'al_vs_random_comparison.png'")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nFinal Results:")
    print(f"  Active Learning:  {al_acc[-1]:.4f}")
    print(f"  Random Sampling:  {random_acc[-1]:.4f}")
    print(f"  Improvement:      {(al_acc[-1] - random_acc[-1]) * 100:.2f}%")
    
    print(f"\nLabel Efficiency:")
    target_acc = 0.6  # Target accuracy
    al_samples = next((count for count, acc in zip(al_counts, al_acc) if acc >= target_acc), "Not reached")
    rand_samples = next((count for count, acc in zip(random_counts, random_acc) if acc >= target_acc), "Not reached")
    
    print(f"  Samples to reach {target_acc:.1%} accuracy:")
    print(f"    Active Learning:  {al_samples}")
    print(f"    Random Sampling:  {rand_samples}")
    
    if isinstance(al_samples, int) and isinstance(rand_samples, int):
        reduction = (1 - al_samples / rand_samples) * 100
        print(f"  Labeling Reduction: {reduction:.1f}%")
    
    return al_acc, al_counts, random_acc, random_counts


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run comparison with 8 iterations, adding 100 samples each time
    run_comparison(iterations=8, batch_size=100)
    
    plt.show()
