"""Deep Learning Wrapper class"""
import torch
import numpy as np

from scripts.cnn import CNNModel
from scripts.lstm import LSTMModel
from scripts.rnn import RNNModel

class ModelWrapper:
    def __init__(self, model):
        self.model = model
        self.model.eval()  # Ensure model is in evaluation mode

    def predict_proba(self, X):
        """
        Computes probability scores for input samples.
        Ensures the model output is in (n_samples, 2) format for LIME.
        """
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)

            # Ensure correct shape for CNN (Conv1D expects (batch, channels, sequence_length))
            if isinstance(self.model, CNNModel) and len(X_tensor.shape) == 2:
                X_tensor = X_tensor.unsqueeze(1)  # Add channel dimension

            # Ensure correct shape for LSTM/RNN (expects 3D shape: (batch, sequence, features))
            elif isinstance(self.model, (LSTMModel, RNNModel)) and len(X_tensor.shape) == 2:
                X_tensor = X_tensor.unsqueeze(1)  # Add sequence dimension

            outputs = self.model(X_tensor)  # Forward pass
            print(f"Model output shape: {outputs.shape}")  # Debugging step

            # Ensure output is in probability format
            if outputs.shape[-1] == 1:  # Binary classification case
                outputs = outputs.squeeze(-1)  # Remove single dim
                outputs = torch.sigmoid(outputs)  # Apply sigmoid

                # Ensure outputs is at least 2D for concatenation
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(1)  # Convert to (n_samples, 1)

                probs = torch.cat([1 - outputs, outputs], dim=1)  # Convert to (n_samples, 2)
            else:
                probs = torch.nn.functional.softmax(outputs, dim=1)  # Multi-class case

            return probs.numpy()  # Convert to NumPy for compatibility

