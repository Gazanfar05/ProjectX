import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class LSTMModel(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(n_features, hidden_size, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(hidden_size * 2, 32, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32 * 2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = x[:, -1, :]  # Take last output
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class HypoglycemiaPredictorModel:
    def __init__(self, sequence_length: int = 12, n_features: int = 9):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"   Using device: {self.device}")
        self.model = None
        
    def build_model(self):
        """Build LSTM model for hypoglycemia prediction"""
        self.model = LSTMModel(self.n_features).to(self.device)
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs: int = 50, batch_size: int = 32):
        """Train the model"""
        if self.model is None:
            self.build_model()
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            
            print(f'   Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model('models/best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'   Early stopping at epoch {epoch+1}')
                    break
        
        return {'best_val_loss': best_val_loss}
    
    def predict_risk(self, sequence: np.ndarray) -> float:
        """Predict hypoglycemia risk for a sequence"""
        if self.model is None:
            raise ValueError("Model not initialized")
        
        self.model.eval()
        
        if len(sequence.shape) == 2:
            sequence = sequence.reshape(1, self.sequence_length, self.n_features)
        
        with torch.no_grad():
            sequence_tensor = torch.FloatTensor(sequence).to(self.device)
            prediction = self.model(sequence_tensor)
            return prediction.cpu().numpy()[0][0]
    
    def save_model(self, path: str):
        """Save model to disk"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'sequence_length': self.sequence_length,
            'n_features': self.n_features
        }, path)
    
    def load_model(self, path: str):
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=self.device)
        self.sequence_length = checkpoint['sequence_length']
        self.n_features = checkpoint['n_features']
        self.build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
