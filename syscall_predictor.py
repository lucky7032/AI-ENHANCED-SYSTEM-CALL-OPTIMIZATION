import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import logging
from pathlib import Path
import json
import os
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime
import optuna
from optuna.integration import TFKerasPruningCallback
import gc

class SystemCallPredictor:
    def __init__(self, 
                 sequence_length: int = 10,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 model_dir: str = 'models',
                 log_dir: str = 'logs'):
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model_dir = Path(model_dir)
        self.log_dir = Path(log_dir)
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.history = None
        
        # Create directories
        self.model_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        
        # Enable mixed precision training
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    def _setup_logging(self):
        """Set up logging with rotation"""
        log_file = self.log_dir / 'predictor.log'
        
        # Set up file handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        
        # Set up console handler
        console_handler = logging.StreamHandler()
        
        # Set up formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Set up logger
        self.logger = logging.getLogger('SystemCallPredictor')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def build_model(self, input_shape: Tuple[int, int], num_classes: int) -> Sequential:
        """Build an enhanced LSTM model with attention and regularization"""
        model = Sequential([
            # Input layer with normalization
            LayerNormalization(input_shape=input_shape),
            
            # First bidirectional LSTM layer with dropout
            Bidirectional(LSTM(256, return_sequences=True)),
            Dropout(0.3),
            LayerNormalization(),
            
            # Second bidirectional LSTM layer with dropout
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.3),
            LayerNormalization(),
            
            # Third bidirectional LSTM layer
            Bidirectional(LSTM(64)),
            Dropout(0.3),
            LayerNormalization(),
            
            # Dense layers with regularization
            Dense(128, activation='relu'),
            Dropout(0.3),
            LayerNormalization(),
            
            Dense(64, activation='relu'),
            Dropout(0.3),
            LayerNormalization(),
            
            # Output layer
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile model with improved optimizer settings
        optimizer = Adam(
            learning_rate=self.learning_rate,
            clipnorm=1.0,
            clipvalue=0.5
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def load_data(self, data_file: str = 'processed_data.npz') -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess the data"""
        self.logger.info(f"Loading data from {data_file}")
        
        try:
            data = np.load(data_file)
            X, y = data['X'], data['y']
            
            # Convert labels to categorical
            y = to_categorical(y, num_classes=len(np.unique(y)))
            
            self.logger.info(f"Loaded {len(X)} sequences with {y.shape[1]} classes")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective for hyperparameter optimization"""
        # Define hyperparameters to optimize
        hp = {
            'lstm_units_1': trial.suggest_int('lstm_units_1', 64, 512),
            'lstm_units_2': trial.suggest_int('lstm_units_2', 32, 256),
            'lstm_units_3': trial.suggest_int('lstm_units_3', 16, 128),
            'dense_units_1': trial.suggest_int('dense_units_1', 32, 256),
            'dense_units_2': trial.suggest_int('dense_units_2', 16, 128),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        }
        
        # Build model with trial hyperparameters
        model = self.build_model(
            input_shape=(self.sequence_length, X.shape[2]),
            num_classes=y.shape[1]
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=hp['batch_size'],
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=3),
                TFKerasPruningCallback(trial, 'val_loss')
            ],
            verbose=0
        )
        
        return history.history['val_loss'][-1]
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, n_trials: int = 50):
        """Optimize model hyperparameters using Optuna"""
        self.logger.info("Starting hyperparameter optimization...")
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=n_trials)
        
        # Get best hyperparameters
        best_params = study.best_params
        self.logger.info(f"Best hyperparameters: {best_params}")
        
        return best_params
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100):
        """Train the model with improved pipeline"""
        self.logger.info("Starting model training...")
        
        try:
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Build model
            self.model = self.build_model(
                input_shape=(self.sequence_length, X.shape[2]),
                num_classes=y.shape[1]
            )
            
            # Set up callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                ModelCheckpoint(
                    self.model_dir / 'best_model.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min'
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
            
            # Train model
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            self.logger.info("Model training completed successfully!")
            
            # Save training history
            self._save_training_history()
            
            # Plot training curves
            self._plot_training_curves()
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise
    
    def _save_training_history(self):
        """Save training history to file"""
        if self.history is None:
            return
        
        history_file = self.log_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.history.history, f)
    
    def _plot_training_curves(self):
        """Plot training and validation curves"""
        if self.history is None:
            return
        
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Save plot
        plt.savefig(self.log_dir / 'training_curves.png')
        plt.close()
    
    def predict_next_syscall(self, sequence: np.ndarray) -> Tuple[str, float]:
        """Predict the next system call with confidence"""
        try:
            # Ensure sequence is properly shaped
            if len(sequence.shape) == 2:
                sequence = sequence.reshape(1, *sequence.shape)
            
            # Make prediction
            prediction = self.model.predict(sequence, verbose=0)
            
            # Get predicted class and confidence
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            # Decode prediction
            predicted_syscall = self.label_encoder.inverse_transform([predicted_class])[0]
            
            self.logger.debug(
                f"Predicted syscall: {predicted_syscall} "
                f"(confidence: {confidence:.2f})"
            )
            
            return predicted_syscall, confidence
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def save_model(self, filepath: str = 'model.h5'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No trained model available")
        
        try:
            self.model.save(filepath)
            self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath: str = 'model.h5'):
        """Load a trained model"""
        try:
            self.model = load_model(filepath)
            self.logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

def main():
    # Initialize predictor
    predictor = SystemCallPredictor()
    
    try:
        # Load data
        X, y = predictor.load_data()
        
        # Train model
        predictor.train(X, y)
        
        # Save model
        predictor.save_model()
        
        print("Model training completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 