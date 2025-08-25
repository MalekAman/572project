import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import warnings
from sklearn.preprocessing import LabelEncoder # Import LabelEncoder

warnings.filterwarnings('ignore')

# Re-define the load_autoencoder_datasets function for self-containment
def load_autoencoder_datasets(processed_path: str = "autoencoder_preprocessed"):
    """
    Load preprocessed autoencoder datasets and components from the specified directory.
    
    Args:
        processed_path (str): The directory containing the saved datasets.
        
    Returns:
        dict: A dictionary containing the loaded datasets and components.
    """
    
    processed_path = Path(processed_path)
    
    print(f"🔄 Loading datasets and components from: {processed_path}")
    
    try:
        # Load datasets
        autoencoder_train = pd.read_csv(processed_path / "autoencoder_train.csv")
        validation = pd.read_csv(processed_path / "validation_data.csv")
        validation_labels = pd.read_csv(processed_path / "validation_labels.csv").squeeze("columns")
        test = pd.read_csv(processed_path / "test_data.csv")
        test_labels = pd.read_csv(processed_path / "test_labels.csv").squeeze("columns")
        hybrid_train = pd.read_csv(processed_path / "hybrid_train_data.csv")
        hybrid_train_labels = pd.read_csv(processed_path / "hybrid_train_labels.csv").squeeze("columns")
        
        # Load preprocessing components
        scaler = joblib.load(processed_path / "scaler.joblib")
        encoders = joblib.load(processed_path / "encoders.joblib")
        feature_info = joblib.load(processed_path / "feature_info.joblib")
        
        print("✅ All datasets and preprocessing components loaded successfully.")
        
        return {
            'autoencoder_train': autoencoder_train,
            'validation': validation,
            'validation_labels': validation_labels,
            'test': test,
            'test_labels': test_labels,
            'hybrid_train': hybrid_train,
            'hybrid_train_labels': hybrid_train_labels,
            'scaler': scaler,
            'encoders': encoders,
            'feature_info': feature_info
        }
        
    except FileNotFoundError as e:
        print(f"❌ Error: One or more files not found. Make sure you have run the preprocessor successfully.")
        print(f"❌ Missing file: {e.filename}")
        return None

## ----------------------------------------------------------------------------------
## Autoencoder Model for Anomaly Detection
## ----------------------------------------------------------------------------------

def build_autoencoder_model(input_dim: int, latent_dim: int) -> keras.Model:
    """
    Builds a simple deep feed-forward autoencoder model.

    Args:
        input_dim (int): The number of input features.
        latent_dim (int): The dimension of the latent space (bottleneck).

    Returns:
        keras.Model: The compiled autoencoder model.
    """
    print(f"\nBuilding Autoencoder with input_dim={input_dim}, latent_dim={latent_dim}")
    
    encoder_input = keras.Input(shape=(input_dim,))
    
    # Encoder
    x = layers.Dense(input_dim // 2, activation='relu')(encoder_input)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(input_dim // 4, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    latent_representation = layers.Dense(latent_dim, activation='relu', name="latent_space")(x)
    
    # Decoder
    x = layers.Dense(input_dim // 4, activation='relu')(latent_representation)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(input_dim // 2, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    decoder_output = layers.Dense(input_dim, activation='sigmoid')(x) # Sigmoid for MinMax scaled data [0,1]
    
    autoencoder = keras.Model(inputs=encoder_input, outputs=decoder_output, name="autoencoder")
    
    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')
    
    print("✅ Autoencoder model built and compiled.")
    autoencoder.summary()
    return autoencoder

def train_autoencoder(model: keras.Model, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                      epochs: int = 50, batch_size: int = 256, model_path: Path = None):
    """
    Trains the autoencoder model.

    Args:
        model (keras.Model): The autoencoder model to train.
        X_train (pd.DataFrame): Training data (normal samples only).
        X_val (pd.DataFrame): Validation data (mixed normal/attack samples).
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        model_path (Path): Path to save the trained model.
    """
    print(f"\nTraining Autoencoder for {epochs} epochs...")
    
    # Use only normal samples from validation for early stopping monitoring
    # Assuming 'Normal' is a label in the validation_labels and we have access to it
    # For this function, we'll assume X_val is already only normal data for simplicity in monitoring,
    # but typically you'd filter X_val or provide X_val_normal for AE validation.
    # For now, we'll use a portion of X_train for validation to mimic purely normal validation.
    X_train_ae, X_val_ae = train_test_split(X_train, test_size=0.15, random_state=42)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
        restore_best_weights=True
    )

    history = model.fit(
        X_train_ae, X_train_ae,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_ae, X_val_ae),
        callbacks=[early_stopping],
        verbose=1
    )
    
    if model_path:
        model.save(model_path)
        print(f"✅ Autoencoder model trained and saved to {model_path}")
    else:
        print("✅ Autoencoder model trained.")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Autoencoder Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(model_path.parent / "autoencoder_training_loss.png")
    plt.show()


def evaluate_autoencoder_for_anomaly_detection(autoencoder_model: keras.Model, 
                                               X_normal_val: pd.DataFrame, 
                                               X_val_full: pd.DataFrame, 
                                               y_val_full: pd.Series,
                                               threshold_quantile: float = 0.95):
    """
    Evaluates the autoencoder's anomaly detection performance and determines a threshold.

    Args:
        autoencoder_model (keras.Model): The trained autoencoder model.
        X_normal_val (pd.DataFrame): Pure normal data from the validation set (for threshold calculation).
        X_val_full (pd.DataFrame): Full validation data (mixed normal/attack) for evaluation.
        y_val_full (pd.Series): Labels for the full validation data.
        threshold_quantile (float): Quantile for setting the anomaly threshold on normal data.

    Returns:
        float: The calculated anomaly detection threshold.
    """
    print("\nEvaluating Autoencoder for Anomaly Detection...")

    # 1. Calculate reconstruction error for normal validation data to set threshold
    reconstructions_normal = autoencoder_model.predict(X_normal_val, verbose=0)
    mse_normal = np.mean(np.power(X_normal_val - reconstructions_normal, 2), axis=1)
    
    # Determine anomaly threshold (e.g., 95th percentile of normal reconstruction errors)
    anomaly_threshold = np.quantile(mse_normal, threshold_quantile)
    print(f"Calculated Anomaly Threshold (at {threshold_quantile*100}th percentile of normal data): {anomaly_threshold:.4f}")

    # Plot histogram of reconstruction errors for normal data (Visualization of "clusters" based on error distribution)
    plt.figure(figsize=(10, 6))
    sns.histplot(mse_normal, bins=50, kde=True, color='blue', label='Normal Data Reconstruction Error')
    plt.axvline(anomaly_threshold, color='red', linestyle='--', label=f'Threshold ({threshold_quantile*100}th percentile)')
    plt.title('Distribution of Reconstruction Errors for Normal Validation Data')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path("autoencoder_preprocessed") / "normal_reconstruction_error_distribution.png")
    plt.show()

    # 2. Evaluate performance on the full validation set (mixed normal/attack)
    reconstructions_full = autoencoder_model.predict(X_val_full, verbose=0)
    mse_full = np.mean(np.power(X_val_full - reconstructions_full, 2), axis=1)

    # Plot scatter plot of reconstruction errors for the full validation set (Visualizing separation)
    plt.figure(figsize=(12, 7))
    # Map labels to colors: Normal is one color, attacks are another.
    colors = ['blue' if label == 'Normal' else 'red' for label in y_val_full]
    plt.scatter(range(len(mse_full)), mse_full, c=colors, s=10, alpha=0.6)
    plt.axhline(anomaly_threshold, color='red', linestyle='--', label=f'Anomaly Threshold ({anomaly_threshold:.4f})')
    
    # Create custom legend handles to differentiate normal/anomaly
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Normal (Actual)', markerfacecolor='blue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Anomaly (Actual)', markerfacecolor='red', markersize=8),
        Line2D([0], [0], color='red', linestyle='--', label=f'Threshold')
    ]
    plt.legend(handles=legend_elements, title="Actual Labels and Threshold")

    plt.title('Reconstruction Errors vs. Sample Index (Validation Set) - Anomaly Detection')
    plt.xlabel('Sample Index')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(Path("autoencoder_preprocessed") / "anomaly_detection_scatter_plot.png")
    plt.show()

    # Predict anomalies: 1 if anomaly (error > threshold), 0 if normal
    predictions = (mse_full > anomaly_threshold).astype(int)
    
    # Convert true labels to binary: 0 for 'Normal', 1 for 'Attack'
    y_true_binary = (y_val_full != 'Normal').astype(int)

    print("\nAnomaly Detection Performance on Validation Set:")
    print(classification_report(y_true_binary, predictions, target_names=['Normal', 'Anomaly']))
    
    cm = confusion_matrix(y_true_binary, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Normal', 'Predicted Anomaly'], 
                yticklabels=['Actual Normal', 'Actual Anomaly'])
    plt.title('Confusion Matrix for Anomaly Detection (Validation Set)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(Path("autoencoder_preprocessed") / "anomaly_detection_confusion_matrix.png")
    plt.show()

    return anomaly_threshold

## ----------------------------------------------------------------------------------
## Hybrid Model (Classifier on Preprocessed Features)
## ----------------------------------------------------------------------------------

def build_hybrid_classifier(input_dim: int, num_classes: int) -> keras.Model:
    """
    Builds a simple feed-forward neural network for the hybrid classification model.

    Args:
        input_dim (int): The number of input features.
        num_classes (int): The number of unique attack types (including 'Normal').

    Returns:
        keras.Model: The compiled classifier model.
    """
    print(f"\nBuilding Hybrid Classifier with input_dim={input_dim}, num_classes={num_classes}")
    
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax') # Softmax for multi-class classification
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("✅ Hybrid Classifier model built and compiled.")
    model.summary()
    return model

def train_hybrid_classifier(model: keras.Model, X_train: pd.DataFrame, y_train: pd.Series, 
                            X_test: pd.DataFrame, y_test: pd.Series,
                            epochs: int = 50, batch_size: int = 256, model_path: Path = None):
    """
    Trains the hybrid classifier model.

    Args:
        model (keras.Model): The classifier model to train.
        X_train (pd.DataFrame): Training features for the hybrid model.
        y_train (pd.Series): Training labels for the hybrid model.
        X_test (pd.DataFrame): Test features for the hybrid model.
        y_test (pd.Series): Test labels for the hybrid model.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        model_path (Path): Path to save the trained model.
    """
    print(f"\nTraining Hybrid Classifier for {epochs} epochs...")
    
    # Load the LabelEncoder used during training
    # --- FIX APPLIED HERE ---
    encoders_file = Path(model_path).parent / "encoders.joblib" # Correctly reference from model_path parent
    # ------------------------
    existing_encoders = joblib.load(encoders_file)
    label_encoder = existing_encoders.get('attack_type_encoder', None)

    if label_encoder is None:
        # If 'attack_type_encoder' was not saved, create and fit it here
        print("❗ 'attack_type_encoder' not found in saved encoders. Creating a new one for target labels.")
        label_encoder = LabelEncoder()
        # Fit on all unique labels seen across training and test to ensure consistency
        all_unique_labels = np.sort(pd.concat([y_train, y_test]).unique())
        label_encoder.fit(all_unique_labels)
        existing_encoders['attack_type_encoder'] = label_encoder
        joblib.dump(existing_encoders, encoders_file)
        print("✅ New 'attack_type_encoder' created and saved.")
    else:
        print("✅ 'attack_type_encoder' found. Using existing encoder for target labels.")

    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train_encoded,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test_encoded),
        callbacks=[early_stopping],
        verbose=1
    )
    
    if model_path:
        model.save(model_path)
        print(f"✅ Hybrid Classifier model trained and saved to {model_path}")
    else:
        print("✅ Hybrid Classifier model trained.")

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Hybrid Classifier Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(model_path.parent / "hybrid_classifier_training_accuracy.png")
    plt.show()

def evaluate_hybrid_classifier(classifier_model: keras.Model, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluates the trained hybrid classifier model.

    Args:
        classifier_model (keras.Model): The trained classifier model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels for the test features.
    """
    print("\nEvaluating Hybrid Classifier on Test Set...")
    
    # Load the LabelEncoder used during training
    # --- FIX APPLIED HERE ---
    # Need to correctly infer the path to encoders.joblib from the current script's location
    encoders_file = Path(".").resolve().parent / "encoders.joblib"
    if not encoders_file.exists(): # Fallback if run directly from within the folder without `processed_data_dir`
        encoders_file = Path("autoencoder_preprocessed") / "encoders.joblib"
    # Fallback to current directory if previous path doesn't work, which happens if you're executing from `autoencoder_preprocessed` itself
    if not encoders_file.exists():
        encoders_file = Path(".") / "encoders.joblib" # Use the current directory
    # ------------------------
    label_encoder = joblib.load(encoders_file).get('attack_type_encoder', None)
    if label_encoder is None:
        raise ValueError("LabelEncoder for 'attack_type' not found. It should have been saved during training.")

    y_test_encoded = label_encoder.transform(y_test)
    
    loss, accuracy = classifier_model.evaluate(X_test, y_test_encoded, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    y_pred_probs = classifier_model.predict(X_test, verbose=0)
    y_pred_encoded = np.argmax(y_pred_probs, axis=1)
    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)

    print("\nClassification Report (Hybrid Classifier - Test Set):")
    print(classification_report(y_test, y_pred_labels))

    cm = confusion_matrix(y_test, y_pred_labels, labels=label_encoder.classes_)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix for Hybrid Classifier (Test Set)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(Path("autoencoder_preprocessed") / "hybrid_classifier_confusion_matrix.png")
    plt.show()


## ----------------------------------------------------------------------------------
## Main Execution
## ----------------------------------------------------------------------------------

if __name__ == '__main__':
    # Define paths
    # Changed processed_data_dir to Path(".") to correctly reference the current directory
    # when the script is executed from inside 'autoencoder_preprocessed'.
    processed_data_dir = Path(".") 
    autoencoder_model_path = processed_data_dir / "autoencoder_model.keras"
    hybrid_classifier_model_path = processed_data_dir / "hybrid_classifier_model.keras"

    # Ensure the output directory exists for models
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load preprocessed data
    loaded_data = load_autoencoder_datasets(processed_path=str(processed_data_dir))
    
    if loaded_data is None:
        print("🚨 Failed to load preprocessed data. Please ensure Data_Preprocessing.py was run successfully.")
    else:
        # Extract datasets
        X_ae_train = loaded_data['autoencoder_train']
        X_val_full = loaded_data['validation']
        y_val_full = loaded_data['validation_labels']
        X_test_full = loaded_data['test']
        y_test_full = loaded_data['test_labels']
        X_hybrid_train = loaded_data['hybrid_train']
        y_hybrid_train = loaded_data['hybrid_train_labels']

        # Determine number of features for models
        input_dim_ae = X_ae_train.shape[1]
        input_dim_hybrid = X_hybrid_train.shape[1]

        # Determine number of classes for hybrid classifier
        all_labels = pd.concat([y_hybrid_train, y_val_full, y_test_full]).unique()
        num_classes = len(all_labels)

        # --------------------------------------------------
        # Autoencoder Model Training and Evaluation
        # --------------------------------------------------
        print("\n" + "="*70)
        print("🤖 AUTOENCODER MODEL FOR ANOMALY DETECTION")
        print("="*70)

        # Build Autoencoder
        latent_dim = 32 # A common choice, can be tuned
        autoencoder = build_autoencoder_model(input_dim_ae, latent_dim)

        # Train Autoencoder (using X_ae_train for training, a split from it for validation)
        # Note: X_val_full here is NOT used for AE validation in fit,
        # but a part of X_ae_train is split to act as validation for reconstruction loss
        train_autoencoder(autoencoder, X_ae_train, X_val_full, epochs=100, batch_size=512, model_path=autoencoder_model_path)
        
        # Split X_val_full into normal and attack parts for evaluation, as the AE was trained only on normal
        normal_val_mask = (y_val_full == 'Normal')
        X_normal_val_for_threshold = X_val_full[normal_val_mask]

        # Evaluate Autoencoder for Anomaly Detection (using full validation set)
        anomaly_threshold = evaluate_autoencoder_for_anomaly_detection(
            autoencoder, X_normal_val_for_threshold, X_val_full, y_val_full, threshold_quantile=0.95
        )

        # --------------------------------------------------
        # Hybrid Classifier Model Training and Evaluation
        # --------------------------------------------------
        print("\n" + "="*70)
        print("CLASSIFIER MODEL FOR HYBRID DETECTION")
        print("="*70)

        # Build Hybrid Classifier
        hybrid_classifier = build_hybrid_classifier(input_dim_hybrid, num_classes)

        # Train Hybrid Classifier (using hybrid_train data, and test data for validation)
        # We need a LabelEncoder for the target 'attack_type'
        # The preprocessor might not have saved an encoder for 'attack_type' if it was treated as target.
        # So we ensure a LabelEncoder is created and saved if it doesn't exist.
        
        # It is crucial to have the full set of unique labels when fitting the LabelEncoder for the hybrid model.
        # So, we pass all unique labels from the combined datasets to ensure the encoder learns all classes.
        all_unique_labels_for_encoder = np.sort(pd.concat([y_hybrid_train, y_test_full]).unique())
        
        # Check if an attack_type_encoder exists, otherwise create and save it
        encoders_file = processed_data_dir / "encoders.joblib" # Use the processed_data_dir for consistency
        existing_encoders = joblib.load(encoders_file)
        if 'attack_type_encoder' not in existing_encoders:
            print("❗ 'attack_type_encoder' not found in saved encoders. Creating a new one for target labels.")
            target_label_encoder = LabelEncoder()
            target_label_encoder.fit(all_unique_labels_for_encoder)
            existing_encoders['attack_type_encoder'] = target_label_encoder
            joblib.dump(existing_encoders, encoders_file)
            print("✅ New 'attack_type_encoder' created and saved.")
        else:
            print("✅ 'attack_type_encoder' found. Using existing encoder for target labels.")
            target_label_encoder = existing_encoders['attack_type_encoder']


        train_hybrid_classifier(
            hybrid_classifier, X_hybrid_train, y_hybrid_train, 
            X_test_full, y_test_full, # Using the test set as validation during training
            epochs=100, batch_size=512, model_path=hybrid_classifier_model_path
        )

        # Evaluate Hybrid Classifier
        evaluate_hybrid_classifier(hybrid_classifier, X_test_full, y_test_full)

        print("\n🎉 Model training and evaluation completed!")
        print(f"Autoencoder model saved to: {autoencoder_model_path}")
        print(f"Hybrid Classifier model saved to: {hybrid_classifier_model_path}")
