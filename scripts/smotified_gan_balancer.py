"""smotified_gan_balancer.py` (Standalone Module for Balancing)"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


class SMOTifiedGANBalancer:
    """
    A class to balance imbalanced fraud datasets using SMOTE followed by a GAN.
    """

    def __init__(self, latent_dim=16, num_epochs=100, batch_size=64, lr=0.0002):
        """
        Initialize the balancer with hyperparameters.

        :param latent_dim: Size of the noise vector for the GAN generator.
        :param num_epochs: Number of epochs for training the GAN.
        :param batch_size: Batch size for GAN training.
        :param lr: Learning rate for both the generator and discriminator.
        """
        self.latent_dim = latent_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

    class Generator(nn.Module):
        """Simple GAN Generator."""
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim),
            )

        def forward(self, z):
            return self.model(z)

    class Discriminator(nn.Module):
        """Simple GAN Discriminator."""
        def __init__(self, input_dim):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.model(x)
