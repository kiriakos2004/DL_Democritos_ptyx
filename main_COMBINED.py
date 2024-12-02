import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib
import json

# Import the DataProcessor from the separate module
from read_data import DataProcessor

# Custom weight initialization function
def initialize_weights(model):
    """Function to initialize weights to make results between PINN and no_PINN more comparable."""
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            # Use Kaiming uniform initialization for the linear layers
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

class ShipSpeedPredictorModel:
    def __init__(self, input_size, lr=0.001, epochs=100, batch_size=32,
                 optimizer_choice='Adam', loss_function_choice='MSE', alpha=1.0,
                 beta=1e-4, gamma=0.1, debug_mode=False):
        self.lr = lr  # Part of hyperparameter search
        self.epochs = epochs  # Manually specified
        self.batch_size = batch_size  # Part of hyperparameter search
        self.optimizer_choice = optimizer_choice  # Manually specified
        self.loss_function_choice = loss_function_choice  # Manually specified
        self.alpha = alpha  # Weight for data loss, hyperparameter
        self.beta = beta    # Weight for physics and PDE loss, hyperparameter
        self.gamma = gamma  # Weight for boundary condition loss, hyperparameter
        self.device = self.get_device()
        self.debug_mode = debug_mode  # Enable or disable debug mode

        # Constants for physics-based loss
        self.rho = 1025.0      # Water density (kg/m³)
        self.nu = 1e-6         # Kinematic viscosity of water (m²/s)
        self.g = 9.81          # Gravitational acceleration (m/s²)

        # Ship dimensions (adjust as per your ship's specifications)
        self.S = 9950.0        # Wetted surface area in m²
        self.L = 229.0         # Ship length in meters
        self.B = 32.0          # Beam (width) of the ship in meters
        self.A_front = 850.0   # Frontal area exposed to air (m²)
        self.S_APP = 150.0     # Wetted surface area of appendages in m²

        # Set random seed for reproducibility
        torch.manual_seed(42)

        # Initialize the model
        self.model = self.ShipSpeedPredictor(input_size).to(self.device)
        initialize_weights(self.model)  # Apply custom initialization

    class ShipSpeedPredictor(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            # Define layers with dropout for regularization
            self.fc1 = nn.Linear(input_size, 512)
            self.dropout1 = nn.Dropout(p=0.1)
            self.fc2 = nn.Linear(512, 256)
            self.dropout2 = nn.Dropout(p=0.1)
            self.fc3 = nn.Linear(256, 128)
            self.dropout3 = nn.Dropout(p=0.1)
            self.fc4 = nn.Linear(128, 64)
            self.dropout4 = nn.Dropout(p=0.1)
            self.fc5 = nn.Linear(64, 32)
            self.dropout5 = nn.Dropout(p=0.1)
            self.fc6 = nn.Linear(32, 16)
            self.dropout6 = nn.Dropout(p=0.1)
            self.fc7 = nn.Linear(16, 1)

            # Define trainable physics parameters with proper initialization
            self.k_wave = nn.Parameter(torch.tensor(np.random.uniform(0.1, 0.4), dtype=torch.float32))
            self.k_aw = nn.Parameter(torch.tensor(np.random.uniform(0.5, 2.0), dtype=torch.float32))
            self.k_appendage = nn.Parameter(torch.tensor(np.random.uniform(0.05, 0.1), dtype=torch.float32))
            self.eta_D = nn.Parameter(torch.tensor(np.random.uniform(0.5, 0.7), dtype=torch.float32))
            self.C_d = nn.Parameter(torch.tensor(np.random.uniform(0.8, 1.0), dtype=torch.float32))

        def forward(self, x):
            x = torch.tanh(self.fc1(x))
            x = self.dropout1(x)
            x = torch.tanh(self.fc2(x))
            x = self.dropout2(x)
            x = torch.tanh(self.fc3(x))
            x = self.dropout3(x)
            x = torch.tanh(self.fc4(x))
            x = self.dropout4(x)
            x = torch.tanh(self.fc5(x))
            x = self.dropout5(x)
            x = torch.tanh(self.fc6(x))
            x = self.dropout6(x)
            x = self.fc7(x)
            return x

    def get_device(self):
        """Function to check if a GPU is available and return the appropriate device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using NVIDIA GPU with CUDA")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple Silicon GPU with MPS")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        return device

    def get_optimizer(self):
        """Function to get the optimizer based on user choice."""
        if self.optimizer_choice == 'Adam':
            return optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer_choice == 'SGD':
            return optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        elif self.optimizer_choice == 'RMSprop':
            return optim.RMSprop(self.model.parameters(), lr=self.lr)
        elif self.optimizer_choice == 'LBFGS':
            return optim.LBFGS(self.model.parameters(), lr=self.lr, max_iter=20,
                               history_size=10, line_search_fn="strong_wolfe")
        else:
            raise ValueError(f"Optimizer {self.optimizer_choice} not recognized.")

    def get_loss_function(self):
        """Function to get the loss function based on user choice."""
        if self.loss_function_choice == 'MSE':
            return nn.MSELoss()
        elif self.loss_function_choice == 'MAE':
            return nn.L1Loss()
        else:
            raise ValueError(f"Loss function {self.loss_function_choice} not recognized.")

    def prepare_dataloader(self, X, y):
        """Function to prepare the DataLoader from data and move tensors to the device."""
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1).to(self.device)

        # Adjust batch size for LBFGS optimizer
        if self.optimizer_choice == 'LBFGS':
            batch_size = len(X)
        else:
            batch_size = self.batch_size

        # Create DataLoader for batching
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        return loader

    def prepare_unscaled_dataloader(self, X_unscaled):
        """Function to prepare the DataLoader for unscaled data."""
        X_unscaled_tensor = torch.tensor(X_unscaled.values, dtype=torch.float32).to(self.device)

        # Adjust batch size for LBFGS optimizer
        if self.optimizer_choice == 'LBFGS':
            batch_size = len(X_unscaled)
        else:
            batch_size = self.batch_size

        unscaled_dataset = TensorDataset(X_unscaled_tensor)
        unscaled_loader = DataLoader(unscaled_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        return unscaled_loader

    def sample_collocation_points(self, num_points, X_train_unscaled, data_processor):
        """Function to sample collocation points within the domain of input features."""
        # Get min and max values from unscaled data
        x_min = X_train_unscaled.min()
        x_max = X_train_unscaled.max()

        # Generate random samples within the range of each feature
        x_collocation_unscaled = pd.DataFrame({
            col: np.random.uniform(low=x_min[col], high=x_max[col], size=num_points)
            for col in X_train_unscaled.columns
        })

        # Scale the collocation points using the same scaler as the training data
        x_collocation_scaled = data_processor.scaler_X.transform(x_collocation_unscaled)
        x_collocation = torch.tensor(x_collocation_scaled, dtype=torch.float32,
                                     device=self.device)
        x_collocation.requires_grad = True
        return x_collocation

    def sample_boundary_points(self, num_points, X_train_unscaled, feature_indices,
                               data_processor):
        """Function to sample boundary points for enforcing boundary conditions."""
        # Get min and max values from unscaled data
        x_min = X_train_unscaled.min()
        x_max = X_train_unscaled.max()

        # Initialize a DataFrame to hold unscaled boundary points
        x_boundary_unscaled = pd.DataFrame(columns=X_train_unscaled.columns)

        # Set 'Speed-Through-Water' to zero
        V_col = X_train_unscaled.columns[feature_indices['Speed-Through-Water']]
        x_boundary_unscaled[V_col] = np.zeros(num_points)

        # For other features, sample within their min and max range
        for col in X_train_unscaled.columns:
            if col != V_col:
                x_boundary_unscaled[col] = np.random.uniform(low=x_min[col], high=x_max[col],
                                                             size=num_points)

        # Scale the boundary points
        x_boundary_scaled = data_processor.scaler_X.transform(x_boundary_unscaled)
        x_boundary = torch.tensor(x_boundary_scaled, dtype=torch.float32, device=self.device)
        x_boundary.requires_grad = True
        return x_boundary

    def compute_pde_residual(self, x_collocation, feature_indices, data_processor):
        """Compute PDE residuals at collocation points with scaling adjustments."""
        x_collocation.requires_grad = True
        outputs = self.model(x_collocation)

        # Compute gradient of outputs with respect to inputs
        outputs_x = torch.autograd.grad(
            outputs=outputs,
            inputs=x_collocation,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True
        )[0]

        # Get index and values of 'Speed-Through-Water' (V)
        V_idx = feature_indices['Speed-Through-Water']
        V = x_collocation[:, V_idx].view(-1, 1)

        # Obtain scaling parameters
        mean_P = torch.tensor(data_processor.scaler_y.mean_, dtype=torch.float32,
                              device=self.device)
        std_P = torch.tensor(data_processor.scaler_y.scale_, dtype=torch.float32,
                             device=self.device)
        mean_V = torch.tensor(data_processor.scaler_X.mean_[V_idx], dtype=torch.float32,
                              device=self.device)
        std_V = torch.tensor(data_processor.scaler_X.scale_[V_idx], dtype=torch.float32,
                             device=self.device)

        # Unscale outputs and V
        outputs_unscaled = outputs * std_P + mean_P
        V_unscaled = V * std_V + mean_V

        # Avoid division by zero or very small values
        V_unscaled_safe = torch.clamp(V_unscaled, min=1e-2)

        # Adjust derivative for scaling
        outputs_V = outputs_x[:, V_idx].view(-1, 1)
        outputs_V_unscaled = (outputs_V * std_P) / std_V

        # Compute residual based on the PDE
        residual = outputs_V_unscaled - (3 * outputs_unscaled) / V_unscaled_safe

        # Adjust scaling factor to balance the PDE loss
        scaling_factor = torch.tensor(1e4, dtype=torch.float32, device=self.device)
        residual_normalized = residual / scaling_factor

        return residual_normalized

    def compute_boundary_loss(self, x_boundary):
        """Function to compute boundary condition loss."""
        outputs_boundary = self.model(x_boundary)
        boundary_loss = torch.mean(outputs_boundary**2)  # Enforce P = 0 when V = 0
        return boundary_loss

    def calculate_physics_loss(self, V, predicted_power_scaled,
                               H_s, theta_ship, theta_wave, data_processor):
        """Calculate the physics-based loss using analytical relationships."""

        # Extract trainable physics parameters from the model
        k_wave = self.model.k_wave          # For wave-making resistance R_w
        k_aw = self.model.k_aw              # For added resistance due to waves R_aw
        eta_D = self.model.eta_D            # Propulsive efficiency
        k_appendage = self.model.k_appendage  # Fraction of R_f for appendage resistance
        C_d = self.model.C_d                # Air resistance drag coefficient

        # Constants
        rho_air = 1.225  # Air density (kg/m³)

        # Ensure V and H_s are not zero to avoid division by zero errors
        V = torch.clamp(V, min=1e-5)
        H_s = torch.clamp(H_s, min=0.0)

        # Calculate Reynolds number Re
        Re = V * self.L / self.nu
        Re = torch.clamp(Re, min=1e-5)

        # Calculate frictional resistance coefficient C_f using ITTC-1957 formula
        C_f = 0.075 / (torch.log10(Re) - 2) ** 2

        # Frictional Resistance (R_f)
        R_f = 0.5 * self.rho * V**2 * self.S * C_f

        # Wave-Making Resistance (R_w)
        R_w = k_wave * R_f

        # Air Resistance (R_a)
        R_a = 0.5 * rho_air * V**2 * self.A_front * C_d

        # Appendage Resistance (R_appendage)
        R_appendage = k_appendage * R_f

        # Compute relative wave direction in degrees
        theta_rel_wave = torch.abs(theta_wave - theta_ship) % 360
        theta_rel_wave = torch.where(theta_rel_wave > 180, 360 - theta_rel_wave, theta_rel_wave)
        theta_rel_wave_rad = theta_rel_wave * torch.pi / 180  # Convert to radians

        # Added Resistance due to Waves (R_aw)
        R_aw = k_aw * self.rho * self.g * H_s**2 * self.B * torch.cos(theta_rel_wave_rad)**2

        # Total Resistance (R_t)
        R_t = R_f + R_w + R_a + R_aw + R_appendage

        # Calculate shaft power (P_S)
        P_S = (V * R_t) / eta_D  # Power in watts

        # Convert power to kilowatts
        P_S = P_S / 1000  # Convert to kilowatts

        # Scaling P_S using the same scaler as the target variable
        scaler_y_mean = torch.tensor(data_processor.scaler_y.mean_, dtype=V.dtype, device=V.device)
        scaler_y_scale = torch.tensor(data_processor.scaler_y.scale_, dtype=V.dtype, device=V.device)
        P_S_scaled = (P_S - scaler_y_mean) / scaler_y_scale  # Element-wise scaling

        # Compute physics loss in scaled space to match data loss scale
        physics_loss = torch.mean((predicted_power_scaled.squeeze() - P_S_scaled) ** 2)

        # Scale the physics loss
        scaling_factor = torch.tensor(100.0, dtype=torch.float32, device=self.device)
        physics_loss_scaled = physics_loss / scaling_factor
        return physics_loss_scaled, P_S_scaled

    def train(self, train_loader, X_train_unscaled, feature_indices, data_processor,
              unscaled_data_loader, val_loader=None, val_unscaled_loader=None, live_plot=False):
        """Function to train the model, including PDE residuals, boundary conditions, and physics-based loss."""
        optimizer = self.get_optimizer()
        loss_function = self.get_loss_function()

        # Add a learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        # Lists to store loss values
        train_losses = []
        val_losses = []

        # Initialize live plotting if enabled
        if live_plot:
            plt.ion()  # Enable interactive mode
            fig, ax = plt.subplots()

        # Extract feature indices for physics loss
        speed_idx = feature_indices['Speed-Through-Water']
        h_s_idx = feature_indices.get('Significant_Wave_Height')
        theta_ship_idx = feature_indices.get('True_Heading')
        theta_wave_idx = feature_indices.get('Mean_Wave_Direction')

        # Adjust loss weights
        beta_pde = self.beta      # Weight for PDE loss
        beta_physics = self.beta  # Weight for physics loss
        gamma = self.gamma

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            running_data_loss = 0.0
            running_pde_loss = 0.0
            running_boundary_loss = 0.0
            running_physics_loss = 0.0

            total_batches = len(train_loader)

            progress_bar = tqdm(
                enumerate(zip(train_loader, unscaled_data_loader)),
                desc=f"Epoch {epoch+1}/{self.epochs}",
                leave=True,
                total=total_batches
            )

            for batch_index, ((X_batch, y_batch), (X_unscaled_batch,)) in progress_bar:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                X_unscaled_batch = X_unscaled_batch.to(self.device)
                optimizer.zero_grad()  # Zero out the gradients

                # Data-driven loss
                outputs = self.model(X_batch)
                data_loss = loss_function(outputs, y_batch)

                # Physics-based loss
                V = X_unscaled_batch[:, speed_idx] * 0.51444  # Convert knots to m/s

                # Only compute physics loss if wave data is available
                if h_s_idx is not None and theta_ship_idx is not None and theta_wave_idx is not None:
                    H_s = X_unscaled_batch[:, h_s_idx]
                    theta_ship = X_unscaled_batch[:, theta_ship_idx]
                    theta_wave = X_unscaled_batch[:, theta_wave_idx]
                else:
                    H_s = torch.zeros_like(V)
                    theta_ship = torch.zeros_like(V)
                    theta_wave = torch.zeros_like(V)

                physics_loss, _ = self.calculate_physics_loss(
                    V, outputs,
                    H_s, theta_ship, theta_wave, data_processor
                )

                # Sample collocation points for PDE residuals
                x_collocation = self.sample_collocation_points(self.batch_size,
                                                               X_train_unscaled,
                                                               data_processor)
                pde_residual = self.compute_pde_residual(x_collocation, feature_indices,
                                                         data_processor)
                pde_loss = torch.mean(pde_residual**2)

                # Sample boundary points for boundary condition loss
                x_boundary = self.sample_boundary_points(self.batch_size, X_train_unscaled,
                                                         feature_indices, data_processor)
                boundary_loss = self.compute_boundary_loss(x_boundary)

                # Combine losses
                total_loss = (
                    self.alpha * data_loss
                    + beta_physics * physics_loss
                    + beta_pde * pde_loss
                    + gamma * boundary_loss
                )

                # Backward pass and optimization
                total_loss.backward()
                optimizer.step()

                # Update running losses
                running_loss += total_loss.item()
                running_data_loss += data_loss.item()
                running_pde_loss += pde_loss.item()
                running_boundary_loss += boundary_loss.item()
                running_physics_loss += physics_loss.item()

                # Update progress bar with current losses
                progress_bar.set_postfix({
                    "Total Loss": f"{running_loss / (batch_index + 1):.8f}",
                    "Data Loss": f"{running_data_loss / (batch_index + 1):.8f}",
                    "PDE Loss": f"{running_pde_loss / (batch_index + 1):.8f}",
                    "Boundary Loss": f"{running_boundary_loss / (batch_index + 1):.8f}",
                    "Physics Loss": f"{running_physics_loss / (batch_index + 1):.8f}",
                })

            # Compute average losses for the epoch
            avg_total_loss = running_loss / total_batches
            avg_data_loss = running_data_loss / total_batches
            avg_pde_loss = running_pde_loss / total_batches
            avg_boundary_loss = running_boundary_loss / total_batches
            avg_physics_loss = running_physics_loss / total_batches

            # Store losses for plotting
            train_losses.append(avg_total_loss)

            # Compute validation loss if validation data is provided
            if val_loader is not None and val_unscaled_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    val_loss = self.evaluate_on_loader(val_loader)
                val_losses.append(val_loss)
                scheduler.step(val_loss)
                print(f"Epoch [{epoch+1}/{self.epochs}], Total Loss: {avg_total_loss:.8f}, "
                      f"Data Loss: {avg_data_loss:.8f}, PDE Loss: {avg_pde_loss:.8f}, "
                      f"Boundary Loss: {avg_boundary_loss:.8f}, Physics Loss: {avg_physics_loss:.8f}, "
                      f"Validation Loss: {val_loss:.8f}, Learning Rate: {optimizer.param_groups[0]['lr']:.1e}")
            else:
                val_losses.append(None)
                scheduler.step(avg_total_loss)
                print(f"Epoch [{epoch+1}/{self.epochs}], Total Loss: {avg_total_loss:.8f}, "
                      f"Data Loss: {avg_data_loss:.8f}, PDE Loss: {avg_pde_loss:.8f}, "
                      f"Boundary Loss: {avg_boundary_loss:.8f}, Physics Loss: {avg_physics_loss:.8f}, "
                      f"Learning Rate: {optimizer.param_groups[0]['lr']:.1e}")

            # Live plotting
            if live_plot:
                ax.clear()
                ax.plot(range(1, epoch+2), train_losses, label='Training Loss')
                if val_loader is not None:
                    ax.plot(range(1, epoch+2), val_losses, label='Validation Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('Training and Validation Loss over Epochs')
                ax.legend()
                plt.pause(0.01)  # Pause to update the plot

        # After training, finalize the plot
        if live_plot:
            plt.ioff()  # Disable interactive mode
            plt.show()
            fig.savefig('training_validation_loss_plot.png')

    def evaluate_on_loader(self, data_loader):
        """Evaluate the model on a data loader and return the average loss."""
        self.model.eval()
        loss_function = self.get_loss_function()
        running_loss = 0.0
        total_batches = len(data_loader)
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = loss_function(outputs, y_batch)
                running_loss += loss.item()
        avg_loss = running_loss / total_batches
        return avg_loss

    def evaluate(self, X_eval, y_eval, dataset_type="Validation", data_processor=None):
        """Function to evaluate the model on the given dataset (validation or test)."""
        self.model.eval()  # Set the model to evaluation mode
        X_eval_tensor = torch.tensor(X_eval.values, dtype=torch.float32).to(self.device)
        y_eval_tensor = torch.tensor(y_eval.values, dtype=torch.float32).view(-1, 1).to(self.device)

        loss_function = self.get_loss_function()
        with torch.no_grad():
            outputs = self.model(X_eval_tensor)
            loss = loss_function(outputs, y_eval_tensor)
            print(f"\n{dataset_type} Loss: {loss.item():.8f}")

            if data_processor:
                # Inverse transform outputs and y_eval to original scale
                outputs_original = data_processor.inverse_transform_y(outputs.cpu().numpy())
                y_eval_original = data_processor.inverse_transform_y(y_eval_tensor.cpu().numpy())

                # Calculate evaluation metrics (e.g., RMSE)
                rmse = np.sqrt(np.mean((outputs_original - y_eval_original) ** 2))
                print(f"{dataset_type} RMSE: {rmse:.4f}")

        return loss.item()

    def cross_validate(self, X, X_unscaled, y, feature_indices, data_processor, k_folds=5):
        """Function to perform cross-validation on the model using training and validation data."""
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            print(f"\nFold {fold+1}/{k_folds}")

            # Split the data into training and validation sets
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            X_train_unscaled_fold, X_val_unscaled_fold = X_unscaled.iloc[train_idx], X_unscaled.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            # Prepare the data loaders
            train_loader = self.prepare_dataloader(X_train_fold, y_train_fold)
            unscaled_data_loader = self.prepare_unscaled_dataloader(X_train_unscaled_fold)
            val_loader = self.prepare_dataloader(X_val_fold, y_val_fold)
            val_unscaled_loader = self.prepare_unscaled_dataloader(X_val_unscaled_fold)

            # Reset model weights for each fold
            self.model.apply(self.reset_weights)

            # Train the model on the training split without live plotting
            self.train(train_loader, X_train_unscaled_fold, feature_indices, data_processor,
                       unscaled_data_loader, val_loader=val_loader, val_unscaled_loader=val_unscaled_loader, live_plot=False)

            # Evaluate the model on the validation split
            val_loss = self.evaluate(X_val_fold, y_val_fold, dataset_type="Validation",
                                     data_processor=data_processor)
            fold_results.append(val_loss)

        # Calculate average validation loss across all folds
        avg_val_loss = np.mean(fold_results)
        print(f"\nCross-validation results: Average Validation Loss = {avg_val_loss:.8f}")
        return avg_val_loss

    @staticmethod
    def reset_weights(m):
        """Function to reset weights of the neural network for each fold."""
        if isinstance(m, nn.Linear):
            m.reset_parameters()

    @staticmethod
    def hyperparameter_search(X_train, X_train_unscaled, y_train, feature_indices,
                              param_grid, epochs_cv, optimizer, loss_function, data_processor, k_folds=5):
        """Function to perform hyperparameter search with cross-validation."""
        best_params = None
        best_loss = float('inf')

        # Generate all combinations of hyperparameters
        hyperparameter_combinations = list(product(
            param_grid['lr'],
            param_grid['batch_size'],
            param_grid['alpha'],
            param_grid['beta'],
            param_grid['gamma']
        ))

        for lr, batch_size, alpha, beta, gamma in hyperparameter_combinations:
            print(f"\nTesting combination: lr={lr}, batch_size={batch_size}, alpha={alpha}, beta={beta}, gamma={gamma}")

            # Initialize model with the current hyperparameters
            model = ShipSpeedPredictorModel(
                input_size=X_train.shape[1],
                lr=lr,
                epochs=epochs_cv,
                optimizer_choice=optimizer,
                loss_function_choice=loss_function,
                batch_size=batch_size,
                alpha=alpha,
                beta=beta,
                gamma=gamma
            )

            # Perform cross-validation
            avg_val_loss = model.cross_validate(
                X_train, X_train_unscaled, y_train, feature_indices, data_processor, k_folds=k_folds
            )

            # Update the best combination if this one is better
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_params = {'lr': lr, 'batch_size': batch_size, 'alpha': alpha, 'beta': beta, 'gamma': gamma}

        print(f"\nBest parameters: {best_params}, with average validation loss: {best_loss:.8f}")

        # Save the best hyperparameters to a text file
        with open("best_hyperparameters.txt", "w") as f:
            f.write(f"Best parameters: {best_params}\n")
            f.write(f"Best average validation loss: {best_loss:.8f}\n")

        return best_params, best_loss

if __name__ == "__main__":
    # Load data using the DataProcessor class
    data_processor = DataProcessor(
        file_path='data/Aframax/P data_20200213-20200726_Democritos.csv',
        target_column='Power',
        keep_columns_file='columns_to_keep.txt'
    )
    result = data_processor.load_and_prepare_data()
    if result is not None:
        X_train, X_test, X_train_unscaled, X_test_unscaled, y_train, y_test, \
            y_train_unscaled, y_test_unscaled = result

        # Print dataset shapes
        print(f"X_train shape: {X_train.shape}")
        print(f"X_train_unscaled shape: {X_train_unscaled.shape}")
        print(f"y_train shape: {y_train.shape}")

        # Ensure that the columns are in the same order in scaled and unscaled data
        assert list(X_train.columns) == list(X_train_unscaled.columns), \
            "Column mismatch between scaled and unscaled data"

        # Create a mapping from feature names to indices
        feature_indices = {col: idx for idx, col in enumerate(X_train_unscaled.columns)}

        # Check if necessary columns are present
        required_columns = [
            'Speed-Through-Water', 'Draft_Fore', 'Draft_Aft',
            'Significant_Wave_Height', 'True_Heading', 'Mean_Wave_Direction'
        ]
        for col in required_columns:
            if col not in feature_indices:
                raise ValueError(f"Required column '{col}' not found in data")

        # Define hyperparameter grid (search for learning rate, batch size, alpha, beta, gamma)
        param_grid = {
            'lr': [0.001],          # Learning rate values to search
            'batch_size': [256],    # Batch size values to search
            'alpha': [1.0, 0.9],         # Data loss weight
            'beta': [1e-4, 1e-5],   # Beta values to search for PDE and physics losses
            'gamma': [0.1, 0.2]     # Gamma values to search
        }

        # Manually specify other hyperparameters
        epochs_cv = 50     # Number of epochs during cross-validation
        epochs_final = 700  # Number of epochs during final training
        optimizer = 'Adam'
        loss_function = 'MSE'

        # Perform hyperparameter search with cross-validation
        best_params, best_loss = ShipSpeedPredictorModel.hyperparameter_search(
            X_train, X_train_unscaled, y_train, feature_indices,
            param_grid, epochs_cv, optimizer, loss_function, data_processor, k_folds=5
        )

        # Split X_train and y_train into training and validation sets for final training
        X_train_final, X_val_final, X_train_unscaled_final, X_val_unscaled_final, \
            y_train_final, y_val_final = train_test_split(
                X_train, X_train_unscaled, y_train, test_size=0.2, random_state=42
            )

        # Create the final model instance with adjusted weights
        final_model = ShipSpeedPredictorModel(
            input_size=X_train.shape[1],
            lr=best_params['lr'],                    # Best learning rate
            epochs=epochs_final,                     # Number of epochs for final training
            optimizer_choice=optimizer,              # Manually specified
            loss_function_choice=loss_function,      # Manually specified
            batch_size=best_params['batch_size'],    # Best batch size
            alpha=best_params['alpha'],              # Data loss weight
            beta=best_params['beta'],                # Physics and PDE loss weight
            gamma=best_params['gamma'],              # Boundary loss weight
            debug_mode=False                         # Disable debug mode for final training
        )

        # Prepare the data loaders for the final model
        final_train_loader = final_model.prepare_dataloader(X_train_final, y_train_final)
        final_unscaled_loader = final_model.prepare_unscaled_dataloader(X_train_unscaled_final)
        final_val_loader = final_model.prepare_dataloader(X_val_final, y_val_final)
        final_val_unscaled_loader = final_model.prepare_unscaled_dataloader(X_val_unscaled_final)

        # Train the final model on the training set, with validation data and live plotting enabled
        final_model.train(
            final_train_loader, X_train_unscaled_final, feature_indices, data_processor,
            unscaled_data_loader=final_unscaled_loader,
            val_loader=final_val_loader,
            val_unscaled_loader=final_val_unscaled_loader,
            live_plot=True
        )

        # Save the trained model's state_dict
        torch.save(final_model.model.state_dict(), 'final_model.pth')

        # Save the DataProcessor's scaler parameters
        joblib.dump(data_processor.scaler_X, 'scaler_X.save')
        joblib.dump(data_processor.scaler_y, 'scaler_y.save')

        # Save the best hyperparameters
        with open('best_hyperparameters.json', 'w') as f:
            json.dump(best_params, f)

        # Evaluate the final model on the test set
        final_model.evaluate(X_test, y_test, dataset_type="Test", data_processor=data_processor)
