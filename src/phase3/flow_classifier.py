# flow_classifier.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

class FlowClassifier:
    """
    Supervised Classifier for Flow-Level Data (Phase 3 - Second Stage)
    """
    
    def __init__(self, data_path: str = "D:/iot_project/flow_data/processed_flows/preprocessed_flows"):
        self.data_path = Path(data_path)
        self.model_path = self.data_path / "flow_classifier_model.keras"
        
        print("🤖 FlowClassifier Initialized")
        print(f"📁 Data path: {self.data_path}")
    
    def load_processed_data(self):
        """Load the preprocessed flow data"""
        print("\n" + "="*50)
        print("📂 LOADING PROCESSED FLOW DATA")
        print("="*50)
        
        X_train = pd.read_csv(self.data_path / "X_train_flow.csv")
        X_test = pd.read_csv(self.data_path / "X_test_flow.csv")
        y_train = pd.read_csv(self.data_path / "y_train_flow.csv").squeeze()
        y_test = pd.read_csv(self.data_path / "y_test_flow.csv").squeeze()
        
        print(f"Training data: {X_train.shape}")
        print(f"Test data: {X_test.shape}")
        print(f"Training labels: {y_train.shape}")
        print(f"Test labels: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def build_classifier(self, input_dim: int, num_classes: int) -> keras.Model:
        """Build the flow classifier model"""
        print(f"\nBuilding Flow Classifier with input_dim={input_dim}, num_classes={num_classes}")
        
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Flow Classifier model built and compiled")
        model.summary()
        return model
    
    def train_classifier(self, model: keras.Model, X_train, y_train, X_test, y_test, epochs: int = 50):
        """Train the flow classifier"""
        print("\n" + "="*50)
        print("🚀 TRAINING FLOW CLASSIFIER")
        print("="*50)
        
        # Load label encoder to convert string labels to integers
        encoders = joblib.load(self.data_path / "flow_encoders.joblib")
        label_encoder = encoders.get('attack_type_encoder')
        
        if label_encoder is None:
            # Create label encoder if not found
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            all_labels = pd.concat([y_train, y_test])
            label_encoder.fit(all_labels)
        
        # Encode labels
        y_train_encoded = label_encoder.transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        # Training callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001
        )
        
        # Train the model
        history = model.fit(
            X_train, y_train_encoded,
            epochs=epochs,
            batch_size=512,
            validation_data=(X_test, y_test_encoded),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Save the trained model
        model.save(self.model_path)
        print(f"✅ Model trained and saved to: {self.model_path}")
        
        # Plot training history
        self.plot_training_history(history)
        
        return history, label_encoder
    
    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Flow Classifier Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Flow Classifier Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.data_path / "flow_classifier_training_history.png")
        plt.show()
    
    def evaluate_classifier(self, model, X_test, y_test, label_encoder):
        """Evaluate the trained classifier"""
        print("\n" + "="*50)
        print("📊 EVALUATING FLOW CLASSIFIER")
        print("="*50)
        
        # Encode test labels
        y_test_encoded = label_encoder.transform(y_test)
        
        # Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Get predictions
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred_encoded = np.argmax(y_pred_probs, axis=1)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        self.plot_confusion_matrix(y_test, y_pred, label_encoder.classes_)
        
        return test_accuracy
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred, labels=class_names)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Confusion Matrix - Flow Classifier (Test Set)')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.data_path / "flow_classifier_confusion_matrix.png")
        plt.show()
    
    def run_training(self, epochs: int = 50):
        """Run complete training pipeline"""
        print("="*50)
        print("🤖 FLOW CLASSIFIER TRAINING PIPELINE")
        print("="*50)
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_processed_data()
        
        # Determine model parameters
        input_dim = X_train.shape[1]
        num_classes = len(y_train.unique())
        
        print(f"Input dimension: {input_dim}")
        print(f"Number of classes: {num_classes}")
        
        # Build model
        model = self.build_classifier(input_dim, num_classes)
        
        # Train model
        history, label_encoder = self.train_classifier(model, X_train, y_train, X_test, y_test, epochs)
        
        # Evaluate model
        accuracy = self.evaluate_classifier(model, X_test, y_test, label_encoder)
        
        print("\n" + "="*50)
        print(f"🎉 FLOW CLASSIFIER TRAINING COMPLETED!")
        print(f"📊 Final Test Accuracy: {accuracy:.4f}")
        print("="*50)
        
        return accuracy

# Main execution
if __name__ == "__main__":
    classifier = FlowClassifier()
    classifier.run_training(epochs=50)