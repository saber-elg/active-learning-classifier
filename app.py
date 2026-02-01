import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from typing import List
import pandas as pd

from src.data_preprocessing import load_data, create_augmented_dataset
from src.model import build_model
from src.active_learning import select_uncertain_samples, calculate_diversity_score
from src.config import config

INPUT_SHAPE = config.model.input_shape
NUM_CLASSES = config.model.num_classes
CLASS_NAMES = config.class_names




def initialize_session_state():
    """Initialize session state with model and data"""
    if "model" not in st.session_state:
        st.session_state.model = build_model(
            INPUT_SHAPE,
            NUM_CLASSES,
            initial_filters=config.model.initial_filters,
            learning_rate=config.model.learning_rate,
            dropout_rate=config.model.dropout_rate,
            l2_reg=config.model.l2_regularization,
            use_batch_norm=config.model.use_batch_norm
        )
        (
            st.session_state.x_labeled,
            st.session_state.y_labeled,
            st.session_state.x_unlabeled,
            st.session_state.x_test,
            st.session_state.y_test,
        ) = load_data(initial_labeled_ratio=config.active_learning.initial_labeled_ratio)
        
        # Initialize metrics tracking
        st.session_state.train_accuracy_history = []
        st.session_state.val_accuracy_history = []
        st.session_state.test_accuracy_history = []
        st.session_state.labeled_count_history = [len(st.session_state.x_labeled)]
        st.session_state.al_iterations = 0
        st.session_state.diversity_scores = []

def train_model(epochs: int, batch_size: int, use_augmentation: bool = True):
    """Train the model with optional data augmentation"""
    with st.spinner("Training in progress..."):
        x_labeled = st.session_state.x_labeled
        y_labeled = st.session_state.y_labeled
        model = st.session_state.model

        early_stopping = EarlyStopping(
            monitor="val_accuracy",
            patience=config.training.early_stopping_patience,
            restore_best_weights=True
        )
        lr_scheduler = ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.training.lr_reduction_factor,
            patience=config.training.lr_reduction_patience
        )

        progress_bar = st.progress(0)
        progress_callback = tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: progress_bar.progress((epoch + 1) / epochs)
        )

        try:
            if use_augmentation and config.active_learning.enable_data_augmentation:
                # Use augmented dataset
                y_labeled_cat = to_categorical(y_labeled, NUM_CLASSES)
                train_size = int(len(x_labeled) * (1 - config.training.validation_split))
                
                x_train, x_val = x_labeled[:train_size], x_labeled[train_size:]
                y_train, y_val = y_labeled_cat[:train_size], y_labeled_cat[train_size:]
                
                train_dataset = create_augmented_dataset(x_train, y_train, batch_size)
                
                history = model.fit(
                    train_dataset,
                    epochs=epochs,
                    validation_data=(x_val, y_val),
                    callbacks=[early_stopping, lr_scheduler, progress_callback],
                    verbose=1,
                )
            else:
                # Standard training
                y_labeled_cat = to_categorical(y_labeled, NUM_CLASSES)
                history = model.fit(
                    x_labeled,
                    y_labeled_cat,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=config.training.validation_split,
                    callbacks=[early_stopping, lr_scheduler, progress_callback],
                    verbose=1,
                )
            
            # Update metrics
            st.session_state.train_accuracy_history.extend(history.history.get("accuracy", []))
            st.session_state.val_accuracy_history.extend(history.history.get("val_accuracy", []))
            
            # Evaluate on test set
            y_test_cat = to_categorical(st.session_state.y_test, NUM_CLASSES)
            test_loss, test_acc, _ = model.evaluate(
                st.session_state.x_test, y_test_cat, verbose=0
            )
            st.session_state.test_accuracy_history.append(test_acc)
            
            st.success(f"Training completed! Test Accuracy: {test_acc:.4f}")
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

def label_uncertain_samples(query_strategy: str = "uncertainty"):
    """Label uncertain samples and add them to training set"""
    with st.spinner("Finding uncertain samples..."):
        model = st.session_state.model
        x_unlabeled = st.session_state.x_unlabeled
        
        if len(x_unlabeled) == 0:
            st.warning("No unlabeled samples remaining!")
            return

        batch_size = min(config.active_learning.query_batch_size, len(x_unlabeled))
        uncertain_indices = select_uncertain_samples(
            model, x_unlabeled, batch_size, strategy=query_strategy
        )
        uncertain_samples = x_unlabeled[uncertain_indices]
        
        # Calculate diversity
        diversity = calculate_diversity_score(uncertain_samples)
        st.info(f"Selected {len(uncertain_samples)} samples with diversity score: {diversity:.2f}")

        # Store in session state for form persistence
        if 'pending_labels' not in st.session_state:
            st.session_state.pending_labels = None
            st.session_state.pending_indices = None
            st.session_state.pending_samples = None

        st.session_state.pending_indices = uncertain_indices
        st.session_state.pending_samples = uncertain_samples

        with st.form(key="labeling_form"):
            labels = []
            for i, sample in enumerate(uncertain_samples):
                prediction = model.predict(sample[np.newaxis, ...], verbose=0)[0]
                pred_class = np.argmax(prediction)
                confidence = np.max(prediction)
                
                # Show top 3 predictions
                top3_idx = np.argsort(prediction)[-3:][::-1]
                top3_probs = prediction[top3_idx]

                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.image(sample, caption=f"Sample {i + 1}", width=250)
                with col2:
                    st.write(f"**Top Prediction:**")
                    st.write(f"{CLASS_NAMES[pred_class]}")
                    st.write(f"Confidence: {confidence:.2%}")
                with col3:
                    st.write("**Top 3:**")
                    for idx, prob in zip(top3_idx, top3_probs):
                        st.write(f"{CLASS_NAMES[idx]}: {prob:.1%}")
                
                label = st.selectbox(
                    f"True Label for Sample {i + 1}",
                    options=list(CLASS_NAMES.keys()),
                    format_func=lambda x: CLASS_NAMES[x],
                    key=f"label_{i}",
                    index=pred_class
                )
                labels.append(label)
                st.divider()

            submit_button = st.form_submit_button("✅ Submit Labels and Update Dataset")
            if submit_button:
                # Add labeled samples to training set
                st.session_state.x_labeled = np.vstack([
                    st.session_state.x_labeled,
                    st.session_state.pending_samples
                ])
                st.session_state.y_labeled = np.concatenate([
                    st.session_state.y_labeled,
                    np.array(labels)
                ])
                
                # Remove from unlabeled set
                mask = np.ones(len(st.session_state.x_unlabeled), dtype=bool)
                mask[st.session_state.pending_indices] = False
                st.session_state.x_unlabeled = st.session_state.x_unlabeled[mask]
                
                # Update metrics
                st.session_state.al_iterations += 1
                st.session_state.labeled_count_history.append(len(st.session_state.x_labeled))
                st.session_state.diversity_scores.append(diversity)
                
                # Clear pending
                st.session_state.pending_labels = None
                st.session_state.pending_indices = None
                st.session_state.pending_samples = None
                
                st.success(f"✅ Successfully added {len(labels)} samples to training set!")
                st.balloons()
                st.rerun()

def evaluate_model():
    """Comprehensive model evaluation with detailed metrics"""
    with st.spinner("Evaluating model..."):
        model = st.session_state.model
        x_test = st.session_state.x_test
        y_test = st.session_state.y_test
        y_test_categorical = to_categorical(y_test, NUM_CLASSES)

        loss, accuracy, top3_accuracy = model.evaluate(x_test, y_test_categorical, verbose=0)
        predictions = model.predict(x_test, verbose=0)
        y_pred = np.argmax(predictions, axis=1)

        st.subheader("📊 Model Performance")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Test Loss", f"{loss:.4f}")
        with col2:
            st.metric("Test Accuracy", f"{accuracy:.4f}")
        with col3:
            st.metric("Top-3 Accuracy", f"{top3_accuracy:.4f}")

        # Classification report
        st.subheader("📋 Classification Report")
        report = classification_report(
            y_test, y_pred,
            target_names=CLASS_NAMES.values(),
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen'))

        # Confusion matrix
        st.subheader("🔥 Confusion Matrix")
        fig, ax = plt.subplots(figsize=(12, 10))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES.values(),
            yticklabels=CLASS_NAMES.values(),
            ax=ax
        )
        plt.title("Confusion Matrix", fontsize=16)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Per-class accuracy
        st.subheader("📈 Per-Class Accuracy")
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        class_acc_df = pd.DataFrame({
            'Class': CLASS_NAMES.values(),
            'Accuracy': class_accuracies
        }).sort_values('Accuracy', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(class_acc_df['Class'], class_acc_df['Accuracy'], color='skyblue')
        ax.set_xlabel('Accuracy', fontsize=12)
        ax.set_title('Per-Class Accuracy', fontsize=14)
        ax.set_xlim([0, 1])
        for i, v in enumerate(class_acc_df['Accuracy']):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


st.set_page_config(page_title="Active Learning Image Classifier", layout="wide", page_icon="🤖")

# Title and description
st.title("🤖 Active Learning for Image Classification")
st.markdown("""
**Efficiently train models with minimal labeled data using active learning strategies.**
Select the most informative samples to label and watch your model improve!
""")

initialize_session_state()

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Active Learning Controls")
    
    st.subheader("Training Settings")
    epochs = st.slider("Epochs", 1, 100, config.training.epochs)
    batch_size = st.slider("Batch Size", 8, 128, config.training.batch_size)
    use_augmentation = st.checkbox("Use Data Augmentation", value=True)
    
    st.subheader("Query Strategy")
    query_strategy = st.selectbox(
        "Sampling Strategy",
        options=["uncertainty", "margin", "entropy", "bald"],
        help="""
        - **Uncertainty**: Least confidence sampling
        - **Margin**: Smallest difference between top 2 predictions
        - **Entropy**: Highest prediction entropy
        - **BALD**: Bayesian Active Learning by Disagreement
        """
    )
    
    st.divider()
    
    # Action buttons
    st.subheader("Actions")
    train_button = st.button("🏋️ Train Model", use_container_width=True)
    label_button = st.button("🏷️ Label Samples", use_container_width=True)
    evaluate_button = st.button("📊 Evaluate Model", use_container_width=True)
    
    # Reset button
    if st.button("🔄 Reset Session", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Main dashboard
tab1, tab2, tab3 = st.tabs(["📈 Metrics Dashboard", "📚 Active Learning Progress", "ℹ️ Model Info"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Labeled Samples",
            len(st.session_state.x_labeled),
            delta=None if not st.session_state.labeled_count_history or len(st.session_state.labeled_count_history) < 2 
                  else len(st.session_state.x_labeled) - st.session_state.labeled_count_history[-2]
        )
    with col2:
        st.metric("Unlabeled Samples", len(st.session_state.x_unlabeled))
    with col3:
        st.metric("AL Iterations", st.session_state.al_iterations)
    with col4:
        current_test_acc = st.session_state.test_accuracy_history[-1] if st.session_state.test_accuracy_history else 0
        st.metric(
            "Test Accuracy",
            f"{current_test_acc:.4f}",
            delta=None if len(st.session_state.test_accuracy_history) < 2 
                  else f"{current_test_acc - st.session_state.test_accuracy_history[-2]:.4f}"
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📉 Learning Curves")
        if st.session_state.val_accuracy_history:
            fig, ax = plt.subplots(figsize=(10, 6))
            epochs_range = range(1, len(st.session_state.val_accuracy_history) + 1)
            ax.plot(epochs_range, st.session_state.train_accuracy_history, 'b-', label='Train Accuracy', alpha=0.8)
            ax.plot(epochs_range, st.session_state.val_accuracy_history, 'r-', label='Val Accuracy', alpha=0.8)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_title('Training and Validation Accuracy', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No training history available. Train the model first!")
    
    with col2:
        st.subheader("📊 Test Accuracy vs Labeled Samples")
        if st.session_state.test_accuracy_history:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(
                st.session_state.labeled_count_history[:len(st.session_state.test_accuracy_history)],
                st.session_state.test_accuracy_history,
                'go-',
                linewidth=2,
                markersize=8
            )
            ax.set_xlabel('Number of Labeled Samples', fontsize=12)
            ax.set_ylabel('Test Accuracy', fontsize=12)
            ax.set_title('Label Efficiency Curve', fontsize=14)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No test accuracy history. Evaluate the model first!")

with tab2:
    st.subheader("🎯 Active Learning Progress")
    
    if st.session_state.al_iterations > 0:
        progress_df = pd.DataFrame({
            'Iteration': range(st.session_state.al_iterations + 1),
            'Labeled Samples': st.session_state.labeled_count_history[:st.session_state.al_iterations + 1],
            'Test Accuracy': [0] + st.session_state.test_accuracy_history if st.session_state.test_accuracy_history else [0] * (st.session_state.al_iterations + 1),
            'Diversity Score': [0] + st.session_state.diversity_scores if st.session_state.diversity_scores else [0] * (st.session_state.al_iterations + 1)
        })
        st.dataframe(progress_df, use_container_width=True)
        
        # Diversity scores over iterations
        if st.session_state.diversity_scores:
            st.subheader("🌈 Sample Diversity Over Iterations")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(range(1, len(st.session_state.diversity_scores) + 1), 
                   st.session_state.diversity_scores, 'mo-', linewidth=2)
            ax.set_xlabel('AL Iteration', fontsize=12)
            ax.set_ylabel('Diversity Score', fontsize=12)
            ax.set_title('Sample Diversity', fontsize=14)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        st.info("Start labeling samples to track active learning progress!")

with tab3:
    st.subheader("🏗️ Model Architecture")
    
    if st.session_state.model:
        # Model summary
        col1, col2 = st.columns(2)
        with col1:
            total_params = st.session_state.model.count_params()
            trainable_params = sum([tf.size(w).numpy() for w in st.session_state.model.trainable_weights])
            st.metric("Total Parameters", f"{total_params:,}")
            st.metric("Trainable Parameters", f"{trainable_params:,}")
        
        with col2:
            st.write("**Architecture Features:**")
            st.write("✅ Residual Connections")
            st.write("✅ Batch Normalization")
            st.write("✅ L2 Regularization")
            st.write("✅ Global Average Pooling")
        
        # Show model summary
        with st.expander("View Full Model Summary"):
            model_summary = []
            st.session_state.model.summary(print_fn=lambda x: model_summary.append(x))
            st.text('\n'.join(model_summary))
    
    st.subheader("⚙️ Configuration")
    config_info = {
        "Learning Rate": config.model.learning_rate,
        "Dropout Rate": config.model.dropout_rate,
        "L2 Regularization": config.model.l2_regularization,
        "Initial Filters": config.model.initial_filters,
        "Batch Norm": config.model.use_batch_norm,
        "Data Augmentation": config.active_learning.enable_data_augmentation,
    }
    st.json(config_info)

# Execute actions
if train_button:
    train_model(epochs, batch_size, use_augmentation)

if label_button:
    label_uncertain_samples(query_strategy)

if evaluate_button:
    evaluate_model()
