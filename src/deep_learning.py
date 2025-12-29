"""
Deep Learning Module for Semiconductor Data Analysis
Implements neural network-based anomaly detection and classification
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .config import RANDOM_STATE


class Autoencoder(nn.Module):
    """
    Autoencoder neural network for anomaly detection.
    
    Learns to reconstruct normal equipment behavior. Anomalies are detected
    when reconstruction error exceeds a threshold.
    """
    
    def __init__(self, input_dim: int, encoding_dim: int = 32):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)


class DeepAnomalyDetector:
    """
    Deep learning-based anomaly detection for equipment monitoring.
    
    Uses an autoencoder to learn normal equipment behavior and detect
    anomalies based on reconstruction error.
    """
    
    def __init__(self, 
                 encoding_dim: int = 32,
                 epochs: int = 100,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 threshold_percentile: float = 95,
                 random_state: int = RANDOM_STATE):
        """
        Initialize deep anomaly detector.
        
        Args:
            encoding_dim: Dimension of the encoded representation
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            threshold_percentile: Percentile of reconstruction error for threshold
            random_state: Random seed for reproducibility
        """
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.threshold_percentile = threshold_percentile
        self.random_state = random_state
        
        self.model = None
        self.scaler = None
        self.threshold_ = None
        self.training_history_ = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not TORCH_AVAILABLE:
            warnings.warn("PyTorch not available. Deep learning features disabled.")
    
    def fit(self, X: pd.DataFrame, verbose: bool = True) -> 'DeepAnomalyDetector':
        """
        Train the autoencoder on normal data.
        
        Args:
            X: Feature DataFrame (should be from normal/healthy operation)
            verbose: Print training progress
        
        Returns:
            self
        """
        if not TORCH_AVAILABLE:
            print("   [WARN] PyTorch not available. Skipping deep learning training.")
            return self
        
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Scale data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split for validation
        X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=self.random_state)
        
        # Create data loaders
        train_tensor = torch.FloatTensor(X_train)
        val_tensor = torch.FloatTensor(X_val)
        
        train_loader = DataLoader(
            TensorDataset(train_tensor, train_tensor),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Initialize model
        input_dim = X.shape[1]
        self.model = Autoencoder(input_dim, self.encoding_dim).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        if verbose:
            print(f"   [DL] Training autoencoder on {len(X_train)} samples...")
            print(f"   [DL] Device: {self.device}")
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            
            for batch_x, _ in train_loader:
                batch_x = batch_x.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_x)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_tensor_device = val_tensor.to(self.device)
                val_output = self.model(val_tensor_device)
                val_loss = criterion(val_output, val_tensor_device).item()
            
            self.training_history_.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss
            })
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    if verbose:
                        print(f"   [DL] Early stopping at epoch {epoch + 1}")
                    break
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"   [DL] Epoch {epoch + 1}/{self.epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Calculate threshold based on training reconstruction error
        self.model.eval()
        with torch.no_grad():
            train_tensor_device = train_tensor.to(self.device)
            reconstructed = self.model(train_tensor_device)
            reconstruction_errors = torch.mean((train_tensor_device - reconstructed) ** 2, dim=1).cpu().numpy()
        
        self.threshold_ = np.percentile(reconstruction_errors, self.threshold_percentile)
        
        if verbose:
            print(f"   [DL] Training complete. Threshold: {self.threshold_:.6f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies based on reconstruction error.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Array of predictions (1 = anomaly, 0 = normal)
        """
        if not TORCH_AVAILABLE or self.model is None:
            return np.zeros(len(X))
        
        reconstruction_errors = self.get_reconstruction_error(X)
        return (reconstruction_errors > self.threshold_).astype(int)
    
    def get_reconstruction_error(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get reconstruction error for each sample.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Array of reconstruction errors
        """
        if not TORCH_AVAILABLE or self.model is None:
            return np.zeros(len(X))
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
        
        return errors
    
    def get_encoded_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get encoded representation of the data.
        
        Useful for visualization and downstream tasks.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Array of encoded features
        """
        if not TORCH_AVAILABLE or self.model is None:
            return np.zeros((len(X), self.encoding_dim))
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            encoded = self.model.encode(X_tensor).cpu().numpy()
        
        return encoded


class DeepClassifier:
    """
    Deep neural network classifier for yield prediction.
    """
    
    def __init__(self,
                 hidden_dims: Tuple[int, ...] = (128, 64, 32),
                 epochs: int = 100,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 random_state: int = RANDOM_STATE):
        
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        self.model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _build_model(self, input_dim: int) -> nn.Module:
        """Build the classifier network."""
        layers = []
        prev_dim = input_dim
        
        for dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(0.3)
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: bool = True) -> 'DeepClassifier':
        """Train the classifier."""
        if not TORCH_AVAILABLE:
            print("   [WARN] PyTorch not available. Skipping deep learning training.")
            return self
        
        torch.manual_seed(self.random_state)
        
        # Prepare data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        y_binary = (y == 1).astype(int).values
        
        # Handle class imbalance with weighted loss
        pos_weight = torch.tensor([(y_binary == 0).sum() / max((y_binary == 1).sum(), 1)])
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_binary, test_size=0.2, 
            random_state=self.random_state, stratify=y_binary
        )
        
        train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train).unsqueeze(1)
            ),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        self.model = self._build_model(X.shape[1]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        if verbose:
            print(f"   [DL] Training deep classifier on {len(X_train)} samples...")
        
        for epoch in range(self.epochs):
            self.model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"   [DL] Epoch {epoch + 1}/{self.epochs}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        if not TORCH_AVAILABLE or self.model is None:
            return np.zeros(len(X))
        
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not TORCH_AVAILABLE or self.model is None:
            return np.zeros(len(X))
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            proba = self.model(X_tensor).cpu().numpy().flatten()
        
        return proba


def run_deep_learning_analysis(X_train: pd.DataFrame, 
                               y_train: pd.Series,
                               X_test: pd.DataFrame,
                               y_test: pd.Series) -> Dict:
    """
    Run complete deep learning analysis.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
    
    Returns:
        Dictionary with results
    """
    results = {
        'torch_available': TORCH_AVAILABLE,
        'autoencoder': None,
        'classifier': None
    }
    
    if not TORCH_AVAILABLE:
        print("   [WARN] PyTorch not installed. Install with: pip install torch")
        return results
    
    print("\n[DL] Running Deep Learning Analysis")
    print("-" * 50)
    
    # Train autoencoder on normal samples
    X_normal = X_train[y_train == -1]
    
    autoencoder = DeepAnomalyDetector(encoding_dim=32, epochs=50)
    autoencoder.fit(X_normal)
    
    # Get predictions
    train_errors = autoencoder.get_reconstruction_error(X_train)
    test_errors = autoencoder.get_reconstruction_error(X_test)
    train_preds = autoencoder.predict(X_train)
    test_preds = autoencoder.predict(X_test)
    
    # Calculate metrics
    y_train_binary = (y_train == 1).astype(int)
    y_test_binary = (y_test == 1).astype(int)
    
    from sklearn.metrics import balanced_accuracy_score, f1_score
    
    train_acc = balanced_accuracy_score(y_train_binary, train_preds)
    test_acc = balanced_accuracy_score(y_test_binary, test_preds)
    
    results['autoencoder'] = {
        'train_balanced_accuracy': train_acc,
        'test_balanced_accuracy': test_acc,
        'threshold': autoencoder.threshold_,
        'detector': autoencoder
    }
    
    print(f"   [DL] Autoencoder - Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")
    
    return results
