"""smotified_gan_balancer.py` (Standalone Module for Balancing)"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


import matplotlib.pyplot as plt


import random

# Set random seed for reproducibility
def set_seed(seed_value=42):
    """Set seed for reproducibility across all necessary libraries."""
    random.seed(seed_value)  # Python random
    np.random.seed(seed_value)  # NumPy
    torch.manual_seed(seed_value)  # PyTorch (CPU)

    # If CUDA is used
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)  # PyTorch CUDA
        torch.cuda.manual_seed_all(seed_value)  # Multi-GPU
        torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
        torch.backends.cudnn.benchmark = False  # Turn off optimizations that introduce randomness

    print(f"Random seed set to: {seed_value}")

# Call this before generating random samples
set_seed(42)


class SMOTifiedGANBalancer:
    """
    A class to balance imbalanced fraud datasets using SMOTE followed by a GAN.
    """

    def __init__(self, latent_dim=16, num_epochs=100, batch_size=64, lr=0.0001):
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
            nn.Linear(input_dim, 64),   # Expand to 64 neurons
            nn.ReLU(),
            nn.Linear(64, 128),  # Further expand to 128 neurons
            nn.ReLU(),
            nn.Linear(128, 64),  # Compress back to 64
            nn.ReLU(),
            nn.Linear(64, output_dim),  # Generate final output
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

    def balance_data(self, X, y):
        """
        Balances the dataset using SMOTE and GAN.

        :param X: Input features (numpy array).
        :param y: Target labels (numpy array).
        :return: Balanced dataset (X_balanced, y_balanced).
        """

        print("Starting SMOTified+GAN Data Balancing...")

        # Step 1: Apply SMOTE to balance the class distribution
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print(f"Data shape after SMOTE:\n Shape of X numpy array: {X_resampled.shape}, shape of y numpy array: {y_resampled.shape}")

        # Convert to PyTorch tensors
        X_resampled_tensor = torch.tensor(X_resampled, dtype=torch.float32)
        y_resampled_tensor = torch.tensor(y_resampled, dtype=torch.float32)

        # Step 2: Train a GAN for Further Balancing
        input_dim = X_resampled.shape[1]
        generator = self.Generator(self.latent_dim, input_dim)
        discriminator = self.Discriminator(input_dim)

        criterion = nn.BCELoss()
        optimizer_G = optim.Adam(generator.parameters(), lr=self.lr)
        optimizer_D = optim.Adam(discriminator.parameters(), lr=self.lr)

        dataset = TensorDataset(X_resampled_tensor, y_resampled_tensor)

        # Create a DataLoader with controlled randomness
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            worker_init_fn=lambda worker_id: np.random.seed(42),
            generator=torch.Generator().manual_seed(42)
        )


        d_losses = []
        g_losses = []

        for epoch in range(self.num_epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            batch_count = 0
            for real_samples, _ in dataloader:
                batch_size = real_samples.size(0)

                # Train Discriminator
                optimizer_D.zero_grad()
                real_labels = torch.ones(batch_size, 1)
                fake_labels = torch.zeros(batch_size, 1)

                real_preds = discriminator(real_samples)
                real_loss = criterion(real_preds, real_labels)

                noise = torch.randn(batch_size, self.latent_dim)
                fake_samples = generator(noise)
                fake_preds = discriminator(fake_samples.detach())
                fake_loss = criterion(fake_preds, fake_labels)

                d_loss = real_loss + fake_loss
                d_loss.backward()
                optimizer_D.step()

                # Train Generator
                optimizer_G.zero_grad()
                fake_preds = discriminator(fake_samples)
                g_loss = criterion(fake_preds, real_labels)

                g_loss.backward()
                optimizer_G.step()

                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                batch_count += 1

            # Store average loss for this epoch
            g_losses.append(epoch_g_loss / batch_count)
            d_losses.append(epoch_d_loss / batch_count)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")


        # After training, plot the loss curves:
        plt.figure(figsize=(8, 5))
        epochs = list(range(1, self.num_epochs + 1))
        plt.plot(epochs, g_losses, label="Generator Loss")
        plt.plot(epochs, d_losses, label="Discriminator Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("GAN Training Loss Curve")
        plt.legend()
        plt.savefig(
            "/home/am/Documents/Software Development/10_Academy Training/week_8-9/fraud-detection/notebooks/plots/class_balancing/loss_curve_GAN.png",
            dpi=300,
            bbox_inches="tight"
            )

        plt.show()


        print("GAN Training Completed. Generating Synthetic Data...")

        # Step 3: Generate Synthetic Data
        num_synthetic_samples = sum(y_resampled == 0)
        noise = torch.randn(num_synthetic_samples, self.latent_dim)
        synthetic_samples = generator(noise).detach().numpy()

        # Step 4: Merge Real and Synthetic Data
        X_final = np.vstack((X_resampled, synthetic_samples))
        y_final = np.hstack((y_resampled, np.ones(num_synthetic_samples)))  # Assign class 1 to synthetic samples

        print(f"Shape of Balanced Data (Using SMOTE + GAN): {X_final.shape}, {y_final.shape}")

        return X_final, y_final
