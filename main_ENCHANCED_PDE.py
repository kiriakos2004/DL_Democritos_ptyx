import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib
import json
from read_data_fragment import DataProcessor
import copy
import math

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
    def __init__(self, 
                 input_size, 
                 lr=0.001, 
                 epochs=100, 
                 batch_size=32,
                 optimizer_choice='Adam', 
                 loss_function_choice='MSE',
                 debug_mode=False,
                 early_stopping=False,
                 patience=10,
                 min_delta=0.0001
                 ):
        
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_choice = optimizer_choice
        self.loss_function_choice = loss_function_choice
        self.device = self.get_device()
        self.debug_mode = debug_mode

        # Early Stopping settings
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta

        # Set random seed for reproducibility
        torch.manual_seed(42)

        # Initialize the model
        self.model = self.ShipSpeedPredictor(input_size).to(self.device)
        initialize_weights(self.model)  # Apply custom initialization

    class ShipSpeedPredictor(nn.Module):
        def __init__(self, input_size, dropout_rate=0.2):
            super().__init__()
            self.fc1 = nn.Linear(input_size, 1024)
            self.fc2 = nn.Linear(1024, 512)
            self.fc3 = nn.Linear(512, 256)
            self.fc4 = nn.Linear(256, 128)
            self.fc5 = nn.Linear(128, 64)
            self.fc6 = nn.Linear(64, 32)
            self.fc7 = nn.Linear(32, 16)
            self.fc8 = nn.Linear(16, 1)

            # Dropout layers
            self.dropout = nn.Dropout(p=dropout_rate)

            # PDE coefficients for wave height, trim, etc.
            self.gamma = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            self.delta = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

            # Trainable exponent parameter (raw_beta) 
            # We'll enforce beta >= 2.0 using beta = 2.0 + exp(raw_beta)
            self.raw_beta = nn.Parameter(torch.tensor(math.log(0.5), dtype=torch.float32))

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = torch.relu(self.fc3(x))
            x = self.dropout(x)
            x = torch.relu(self.fc4(x))
            x = self.dropout(x)
            x = torch.relu(self.fc5(x))
            x = self.dropout(x)
            x = torch.relu(self.fc6(x))
            x = self.dropout(x)
            x = torch.relu(self.fc7(x))
            x = self.fc8(x)
            return x

    def get_device(self):
        """Check if a GPU is available and return the appropriate device."""
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
        """Get the optimizer based on user choice."""
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
        """Get the loss function based on user choice."""
        if self.loss_function_choice == 'MSE':
            return nn.MSELoss()
        elif self.loss_function_choice == 'MAE':
            return nn.L1Loss()
        else:
            raise ValueError(f"Loss function {self.loss_function_choice} not recognized.")

    def prepare_dataloader(self, X, y):
        """Prepare the DataLoader from data and move tensors to the device."""
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1).to(self.device)

        # Adjust batch size for LBFGS optimizer
        if self.optimizer_choice == 'LBFGS':
            batch_size = len(X)
        else:
            batch_size = self.batch_size

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        return loader

    def prepare_unscaled_dataloader(self, X_unscaled):
        """Prepare the DataLoader for unscaled data."""
        X_unscaled_tensor = torch.tensor(X_unscaled.values, dtype=torch.float32).to(self.device)

        if self.optimizer_choice == 'LBFGS':
            batch_size = len(X_unscaled)
        else:
            batch_size = self.batch_size

        unscaled_dataset = TensorDataset(X_unscaled_tensor)
        unscaled_loader = DataLoader(unscaled_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        return unscaled_loader

    def sample_collocation_points(self, num_points, X_unscaled, data_processor):
        """Sample collocation points within the domain of the given unscaled data."""
        x_min = X_unscaled.min()
        x_max = X_unscaled.max()

        x_collocation_unscaled = pd.DataFrame({
            col: np.random.uniform(low=x_min[col], high=x_max[col], size=num_points)
            for col in X_unscaled.columns
        })

        x_collocation_scaled = data_processor.scaler_X.transform(x_collocation_unscaled)
        x_collocation = torch.tensor(x_collocation_scaled, dtype=torch.float32, device=self.device)
        x_collocation.requires_grad = True
        return x_collocation

    def compute_pde_residual(self, x_collocation, feature_indices, data_processor):

        x_collocation.requires_grad_(True)
        outputs = self.model(x_collocation)  # scaled power: shape [batch, 1]

        # Partial derivatives wrt each input feature
        outputs_x = torch.autograd.grad(
            outputs=outputs,
            inputs=x_collocation,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True
        )[0]  # shape [batch, num_features]

        # Identify columns
        V_idx = feature_indices['Speed-Through-Water']
        Hs_idx = feature_indices.get('Significant_Wave_Height', None)
        fore_idx = feature_indices.get('Draft_Fore', None)
        aft_idx = feature_indices.get('Draft_Aft', None)

        # Extract partial derivatives (scaled)
        dP_dV_scaled = outputs_x[:, V_idx].view(-1, 1)

        if Hs_idx is not None:
            dP_dHs_scaled = outputs_x[:, Hs_idx].view(-1, 1)
        else:
            dP_dHs_scaled = torch.zeros_like(dP_dV_scaled)

        if fore_idx is not None and aft_idx is not None:
            dP_dFore_scaled = outputs_x[:, fore_idx].view(-1, 1)
            dP_dAft_scaled  = outputs_x[:, aft_idx].view(-1, 1)
            # dP/dTrim = dP/dFore - dP/dAft
            dP_dTrim_scaled = dP_dFore_scaled - dP_dAft_scaled
        else:
            dP_dTrim_scaled = torch.zeros_like(dP_dV_scaled)

        # Unscale P
        mean_P = torch.tensor(data_processor.scaler_y.mean_, dtype=torch.float32, device=self.device)
        std_P = torch.tensor(data_processor.scaler_y.scale_, dtype=torch.float32, device=self.device)
        P_unscaled = outputs * std_P + mean_P  # shape [batch, 1]

        # Unscale partial derivatives: (dP_scaled/dX_scaled) * (std_P / std_X)
        mean_V = torch.tensor(data_processor.scaler_X.mean_[V_idx], dtype=torch.float32, device=self.device)
        std_V  = torch.tensor(data_processor.scaler_X.scale_[V_idx], dtype=torch.float32, device=self.device)

        dP_dV_unscaled = (dP_dV_scaled * std_P) / std_V

        if Hs_idx is not None:
            mean_Hs = torch.tensor(data_processor.scaler_X.mean_[Hs_idx], dtype=torch.float32, device=self.device)
            std_Hs  = torch.tensor(data_processor.scaler_X.scale_[Hs_idx], dtype=torch.float32, device=self.device)
            dP_dHs_unscaled = (dP_dHs_scaled * std_P) / std_Hs
        else:
            dP_dHs_unscaled = torch.zeros_like(dP_dV_unscaled)

        if fore_idx is not None and aft_idx is not None:
            mean_Fore = torch.tensor(data_processor.scaler_X.mean_[fore_idx], dtype=torch.float32, device=self.device)
            std_Fore  = torch.tensor(data_processor.scaler_X.scale_[fore_idx], dtype=torch.float32, device=self.device)
            mean_Aft = torch.tensor(data_processor.scaler_X.mean_[aft_idx], dtype=torch.float32, device=self.device)
            std_Aft  = torch.tensor(data_processor.scaler_X.scale_[aft_idx], dtype=torch.float32, device=self.device)

            dP_dFore_unscaled = (dP_dFore_scaled * std_P) / std_Fore
            dP_dAft_unscaled  = (dP_dAft_scaled  * std_P) / std_Aft
            dP_dTrim_unscaled = dP_dFore_unscaled - dP_dAft_unscaled
        else:
            dP_dTrim_unscaled = torch.zeros_like(dP_dV_unscaled)

        # Unscale speed
        V_scaled = x_collocation[:, V_idx].view(-1, 1)
        V_unscaled = V_scaled * std_V + mean_V
        V_unscaled_safe = torch.clamp(V_unscaled, min=1e-3)

        # PDE coefficients
        gamma = self.model.gamma
        delta = self.model.delta

        # Enforce beta >= 2 with beta = 2.0 + exp(raw_beta)
        beta = 2.0 + torch.exp(self.model.raw_beta)

        # PDE:
        # dP/dV + gamma*dP/dH_s + delta*dP/dTrim - (beta*P / V) = 0
        residual = (
            dP_dV_unscaled
            + gamma * dP_dHs_unscaled
            + delta * dP_dTrim_unscaled
            - (beta * P_unscaled) / V_unscaled_safe
        )

        # Scale down the residual for stability
        scaling_factor = torch.tensor(1e5, dtype=torch.float32, device=self.device)
        residual_normalized = residual / scaling_factor

        return residual_normalized

    def compute_boundary_loss(self, feature_indices, data_processor, X_unscaled, scale=1e8):
        """
        Boundary conditions (example):
         - P ~ 0 for speeds in [0,1)
         - P >= 9000 kW at speed = 13 knots
        """
        V_idx = feature_indices['Speed-Through-Water']
        x_min = X_unscaled.min()
        x_max = X_unscaled.max()

        num_points = self.batch_size

        mean_P = torch.tensor(data_processor.scaler_y.mean_, dtype=torch.float32, device=self.device)
        std_P = torch.tensor(data_processor.scaler_y.scale_, dtype=torch.float32, device=self.device)

        # Condition 1: P ~ 0 for speeds in [0,1)
        V_col = X_unscaled.columns[V_idx]
        x_boundary_unscaled_low_speed = pd.DataFrame(columns=X_unscaled.columns)

        x_boundary_unscaled_low_speed[V_col] = np.random.uniform(0.0, 1.0, size=num_points)
        for col in X_unscaled.columns:
            if col != V_col:
                x_boundary_unscaled_low_speed[col] = np.random.uniform(
                    low=x_min[col], high=x_max[col], size=num_points
                )

        x_boundary_low_speed_scaled = data_processor.scaler_X.transform(x_boundary_unscaled_low_speed)
        x_boundary_low_speed = torch.tensor(x_boundary_low_speed_scaled, dtype=torch.float32, device=self.device)
        outputs_low_speed = self.model(x_boundary_low_speed)

        boundary_loss_low_speed = torch.mean(outputs_low_speed**2) / scale

        # Condition 2: P >= 9000 kW at speed=13 knots
        V_ineq_knots = 13.0
        P_min = 9000.0
        x_boundary_unscaled_ineq = pd.DataFrame(columns=X_unscaled.columns)
        x_boundary_unscaled_ineq[V_col] = np.full(num_points, V_ineq_knots)

        for col in X_unscaled.columns:
            if col != V_col:
                x_boundary_unscaled_ineq[col] = np.random.uniform(
                    low=x_min[col], high=x_max[col], size=num_points
                )

        x_boundary_ineq_scaled = data_processor.scaler_X.transform(x_boundary_unscaled_ineq)
        x_boundary_ineq = torch.tensor(x_boundary_ineq_scaled, dtype=torch.float32, device=self.device)
        outputs_ineq = self.model(x_boundary_ineq)

        outputs_ineq_unscaled = outputs_ineq * std_P + mean_P

        violation = torch.relu(P_min - outputs_ineq_unscaled)
        boundary_loss_ineq = torch.mean(violation**2) / scale

        boundary_loss = boundary_loss_low_speed + boundary_loss_ineq
        return boundary_loss

    @staticmethod
    def compute_total_loss(data_loss, pde_loss, boundary_loss,
                           data_loss_coeff=1.0, pde_loss_coeff=1.0,
                           boundary_loss_coeff=1.0):

        return (data_loss_coeff * data_loss) \
             + (pde_loss_coeff * pde_loss) \
             + (boundary_loss_coeff * boundary_loss)

    def train(self, 
            train_loader, 
            X_train_unscaled, 
            feature_indices, 
            data_processor,
            unscaled_data_loader, 
            val_loader=None, 
            val_unscaled_loader=None, 
            X_val_unscaled=None,
            live_plot=False):
        """
        Train method with optional early stopping based on validation total loss.
        """
        optimizer = self.get_optimizer()
        loss_function = self.get_loss_function()

        train_losses = []
        val_losses = []

        # Track betas across epochs (optional)
        betas = []

        # For live plot (if you want to visualize during training)
        if live_plot:
            plt.ion()
            fig, ax = plt.subplots()

        # Early stopping tracking
        best_val_loss = float('inf')
        best_model_state = None
        epochs_no_improvement = 0

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            running_data_loss = 0.0
            running_pde_loss = 0.0
            running_boundary_loss = 0.0

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

                optimizer.zero_grad()

                # Forward pass => data loss
                outputs = self.model(X_batch)
                data_loss = loss_function(outputs, y_batch)

                # Compute PDE residual
                x_collocation = self.sample_collocation_points(self.batch_size, 
                                                            X_train_unscaled, 
                                                            data_processor)
                pde_residual = self.compute_pde_residual(x_collocation, 
                                                        feature_indices, 
                                                        data_processor)
                pde_loss = torch.mean(pde_residual**2)

                # Boundary loss
                boundary_loss = self.compute_boundary_loss(feature_indices, 
                                                        data_processor, 
                                                        X_train_unscaled)

                # For demonstration, let's do a smaller PDE weight if desired:
                total_loss = self.compute_total_loss(
                    data_loss=data_loss,
                    pde_loss=pde_loss,
                    boundary_loss=boundary_loss,
                    data_loss_coeff=1.0, 
                    pde_loss_coeff=1.0,  # or 1.0, depending on preference
                    boundary_loss_coeff=1.0
                )

                total_loss.backward()
                optimizer.step()

                # Accumulate
                running_loss += total_loss.item()
                running_data_loss += data_loss.item()
                running_pde_loss += pde_loss.item()
                running_boundary_loss += boundary_loss.item()

                # Log the current beta (now forced >= 2)
                with torch.no_grad():
                    current_beta = 2.0 + torch.exp(self.model.raw_beta)

                progress_bar.set_postfix({
                    "Total Loss": f"{running_loss / (batch_index + 1):.6f}",
                    "Data Loss": f"{running_data_loss / (batch_index + 1):.6f}",
                    "PDE Loss": f"{running_pde_loss / (batch_index + 1):.6f}",
                    "Boundary Loss": f"{running_boundary_loss / (batch_index + 1):.6f}",
                    "beta": f"{current_beta.item():.3f}"
                })

            avg_total_loss = running_loss / total_batches
            avg_data_loss = running_data_loss / total_batches
            avg_pde_loss = running_pde_loss / total_batches
            avg_boundary_loss = running_boundary_loss / total_batches

            train_losses.append(avg_total_loss)

            # Store beta for each epoch (optional)
            betas.append(current_beta.item())

            # Validation
            val_total_loss = None
            if val_loader is not None and val_unscaled_loader is not None and X_val_unscaled is not None:
                val_total_loss, val_data_loss, val_pde_loss, val_boundary_loss = \
                    self.evaluate_on_loader(val_loader, val_unscaled_loader, 
                                            feature_indices, data_processor, 
                                            X_val_unscaled)
                val_losses.append(val_total_loss)

                print(f"Epoch [{epoch+1}/{self.epochs}], "
                    f"Total Loss: {avg_total_loss:.6f}, Data Loss: {avg_data_loss:.6f}, "
                    f"PDE Loss: {avg_pde_loss:.6f}, Boundary Loss: {avg_boundary_loss:.6f}, "
                    f"Validation Total Loss: {val_total_loss:.6f}, "
                    f"Validation Data Loss: {val_data_loss:.6f}, "
                    f"Validation PDE Loss: {val_pde_loss:.6f}, "
                    f"Validation Boundary Loss: {val_boundary_loss:.6f}, "
                    f"LR: {optimizer.param_groups[0]['lr']:.1e}, "
                    f"beta: {current_beta.item():.3f}")
            else:
                val_losses.append(None)
                print(f"Epoch [{epoch+1}/{self.epochs}], "
                    f"Total Loss: {avg_total_loss:.6f}, Data Loss: {avg_data_loss:.6f}, "
                    f"PDE Loss: {avg_pde_loss:.6f}, Boundary Loss: {avg_boundary_loss:.6f}, "
                    f"LR: {optimizer.param_groups[0]['lr']:.1e}, "
                    f"beta: {current_beta.item():.3f}")


            # EARLY STOPPING LOGIC
            if self.early_stopping and val_total_loss is not None:
                if val_total_loss < (best_val_loss - self.min_delta):
                    best_val_loss = val_total_loss
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    epochs_no_improvement = 0
                else:
                    epochs_no_improvement += 1
                    if epochs_no_improvement >= self.patience:
                        print(f"Early stopping triggered at epoch {epoch+1}.")
                        break

            # Live plot, if desired
            if live_plot:
                ax.clear()
                ax.plot(range(1, epoch+2), train_losses, label='Training Total Loss')
                if val_loader is not None:
                    valid_val_epochs = [i+1 for i, v in enumerate(val_losses) if v is not None]
                    valid_val_losses = [v for v in val_losses if v is not None]
                    ax.plot(valid_val_epochs, valid_val_losses, label='Validation Total Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('Training and Validation Total Loss over Epochs')
                ax.legend()
                plt.pause(0.01)

        if self.early_stopping and best_model_state is not None:
            print("Restoring best model state from early-stopping.")
            self.model.load_state_dict(best_model_state)

        if live_plot:
            plt.ioff()
            plt.show()
            fig.savefig('training_validation_loss_plot.png')

    def evaluate_on_loader(self, data_loader, unscaled_data_loader, feature_indices, data_processor, X_val_unscaled):

        self.model.eval()
        loss_function = self.get_loss_function()
        running_total_loss = 0.0
        running_data_loss = 0.0
        running_pde_loss = 0.0
        running_boundary_loss = 0.0
        total_batches = len(data_loader)

        with torch.no_grad():
            for (X_batch, y_batch), (X_unscaled_batch,) in zip(data_loader, unscaled_data_loader):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                # Data loss
                outputs = self.model(X_batch)
                data_loss = loss_function(outputs, y_batch)

                # PDE computations: must briefly enable grad
                with torch.enable_grad():
                    x_collocation = self.sample_collocation_points(len(X_batch), X_val_unscaled, data_processor)
                    x_collocation.requires_grad_(True)
                    pde_residual = self.compute_pde_residual(x_collocation, feature_indices, data_processor)
                    pde_loss = torch.mean(pde_residual**2)

                # Boundary loss
                boundary_loss = self.compute_boundary_loss(feature_indices, data_processor, X_val_unscaled)

                # Total loss
                total_loss = self.compute_total_loss(data_loss, pde_loss, boundary_loss)

                running_total_loss += total_loss.item()
                running_data_loss += data_loss.item()
                running_pde_loss += pde_loss.item()
                running_boundary_loss += boundary_loss.item()

        avg_total_loss = running_total_loss / total_batches
        avg_data_loss = running_data_loss / total_batches
        avg_pde_loss = running_pde_loss / total_batches
        avg_boundary_loss = running_boundary_loss / total_batches

        print(f"Validation Loss Breakdown: "
              f"Total: {avg_total_loss:.6f}, "
              f"Data: {avg_data_loss:.6f}, "
              f"PDE: {avg_pde_loss:.6f}, "
              f"Boundary: {avg_boundary_loss:.6f}")

        return avg_total_loss, avg_data_loss, avg_pde_loss, avg_boundary_loss

    def evaluate(self, X_eval, y_eval, dataset_type="Validation", data_processor=None):
        """
        Evaluate data loss only (no PDE or boundary) on a held-out set.
        """
        self.model.eval()
        X_eval_tensor = torch.tensor(X_eval.values, dtype=torch.float32).to(self.device)
        y_eval_tensor = torch.tensor(y_eval.values, dtype=torch.float32).view(-1, 1).to(self.device)

        loss_function = self.get_loss_function()
        with torch.no_grad():
            outputs = self.model(X_eval_tensor)
            loss = loss_function(outputs, y_eval_tensor)
            print(f"\n{dataset_type} Loss: {loss.item():.6f}")

            if data_processor:
                outputs_original = data_processor.inverse_transform_y(outputs.cpu().numpy())
                y_eval_original = data_processor.inverse_transform_y(y_eval_tensor.cpu().numpy())
                rmse = np.sqrt(np.mean((outputs_original - y_eval_original) ** 2))
                print(f"{dataset_type} RMSE: {rmse:.4f}")

        return loss.item()

    def cross_validate(self, X, X_unscaled, y, feature_indices, data_processor, k_folds=5):
        """
        k-fold cross-validation using data, PDE, and boundary losses.
        """
        kfold = KFold(n_splits=k_folds)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            print(f"\nFold {fold+1}/{k_folds}")
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            X_train_unscaled_fold, X_val_unscaled_fold = X_unscaled.iloc[train_idx], X_unscaled.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            train_loader = self.prepare_dataloader(X_train_fold, y_train_fold)
            unscaled_data_loader = self.prepare_unscaled_dataloader(X_train_unscaled_fold)
            val_loader = self.prepare_dataloader(X_val_fold, y_val_fold)
            val_unscaled_loader = self.prepare_unscaled_dataloader(X_val_unscaled_fold)

            self.model.apply(self.reset_weights)
            self.train(train_loader, X_train_unscaled_fold, feature_indices, data_processor,
                       unscaled_data_loader,
                       val_loader=val_loader,
                       val_unscaled_loader=val_unscaled_loader,
                       X_val_unscaled=X_val_unscaled_fold,
                       live_plot=False)
            val_loss = self.evaluate(X_val_fold, y_val_fold, dataset_type="Validation", data_processor=data_processor)
            fold_results.append(val_loss)

        avg_val_loss = np.mean(fold_results)
        print(f"\nCross-validation results: Average Validation Loss = {avg_val_loss:.6f}")
        return avg_val_loss

    @staticmethod
    def reset_weights(m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()

    @staticmethod
    def hyperparameter_search(X_train, X_train_unscaled, y_train, feature_indices,
                              param_grid, epochs_cv, optimizer, loss_function,
                              data_processor, k_folds=5):

        best_params = None
        best_loss = float('inf')
        hyperparameter_combinations = list(product(
            param_grid['lr'],
            param_grid['batch_size']
        ))

        for lr, batch_size in hyperparameter_combinations:
            print(f"\nTesting combination: lr={lr}, batch_size={batch_size}")
            model = ShipSpeedPredictorModel(
                input_size=X_train.shape[1],
                lr=lr,
                epochs=epochs_cv,
                optimizer_choice=optimizer,
                loss_function_choice=loss_function,
                batch_size=batch_size,
                debug_mode=False
            )

            avg_val_loss = model.cross_validate(
                X_train, X_train_unscaled, y_train, feature_indices, data_processor,
                k_folds=k_folds
            )

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_params = {'lr': lr, 'batch_size': batch_size}

        print(f"\nBest parameters: {best_params}, with average validation loss: {best_loss:.6f}")
        with open("best_hyperparameters.txt", "w") as f:
            f.write(f"Best parameters: {best_params}\n")
            f.write(f"Best average validation loss: {best_loss:.6f}\n")

        return best_params, best_loss


if __name__ == "__main__":
    # Example usage:
    data_processor = DataProcessor(
        file_path='data/Dan/P data_20210428-20211111_Democritos.csv',
        target_column='Power',
        keep_columns_file='columns_to_keep.txt'
    )
    result = data_processor.load_and_prepare_data()
    if result is not None:
        X_train, X_test, X_train_unscaled, X_test_unscaled, y_train, y_test, \
            y_train_unscaled, y_test_unscaled = result

        feature_indices = {col: idx for idx, col in enumerate(X_train_unscaled.columns)}

        # Check required columns
        required_cols = ['Speed-Through-Water', 'Draft_Fore', 'Draft_Aft']
        for rc in required_cols:
            if rc not in feature_indices:
                raise ValueError(f"Required column '{rc}' not found in data")

        # Hyperparameter grid
        param_grid = {
            'lr': [0.0001],
            'batch_size': [1024]
        }
        epochs_cv = 1
        epochs_final = 100
        optimizer = 'Adam'
        loss_function = 'MSE'

        best_params, best_loss = ShipSpeedPredictorModel.hyperparameter_search(
            X_train, X_train_unscaled, y_train, feature_indices,
            param_grid, epochs_cv, optimizer, loss_function, data_processor, k_folds=5
        )

        # Final training
        X_train_final, X_val_final, X_train_unscaled_final, X_val_unscaled_final, \
            y_train_final, y_val_final = train_test_split(
                X_train, X_train_unscaled, y_train, test_size=0.1
            )

        final_model = ShipSpeedPredictorModel(
            input_size=X_train.shape[1],
            lr=best_params['lr'],
            epochs=epochs_final,
            optimizer_choice=optimizer,
            loss_function_choice=loss_function,
            batch_size=best_params['batch_size'],
            debug_mode=False,
            early_stopping=False,  # or True, if you wish
            patience=40,
            min_delta=0.0001
        )

        train_loader = final_model.prepare_dataloader(X_train_final, y_train_final)
        unscaled_loader = final_model.prepare_unscaled_dataloader(X_train_unscaled_final)
        val_loader = final_model.prepare_dataloader(X_val_final, y_val_final)
        val_unscaled_loader = final_model.prepare_unscaled_dataloader(X_val_unscaled_final)

        final_model.train(train_loader,
                          X_train_unscaled_final,
                          feature_indices,
                          data_processor,
                          unscaled_data_loader=unscaled_loader,
                          val_loader=val_loader,
                          val_unscaled_loader=val_unscaled_loader,
                          X_val_unscaled=X_val_unscaled_final,
                          live_plot=True)

        with open('best_hyperparameters.json', 'w') as f:
            json.dump(best_params, f)

        # Evaluate on test set
        final_model.evaluate(X_test, y_test, dataset_type="Test", data_processor=data_processor)

        # Convert for inference
        X_test_unscaled = X_test_unscaled[X_train_unscaled.columns]
        X_test_scaled = data_processor.scaler_X.transform(X_test_unscaled)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(final_model.device)

        final_model.model.eval()
        with torch.no_grad():
            y_pred_scaled = final_model.model(X_test_tensor)
            y_pred_scaled_np = y_pred_scaled.cpu().numpy()
            y_pred = data_processor.inverse_transform_y(y_pred_scaled_np).flatten()

        y_actual = y_test_unscaled.values.flatten()

        speed_unscaled = X_test_unscaled['Speed-Through-Water'].values
        draft_fore_unscaled = X_test_unscaled['Draft_Fore'].values
        draft_aft_unscaled = X_test_unscaled['Draft_Aft'].values

        if 'Significant_Wave_Height' in X_test_unscaled.columns:
            hs_unscaled = X_test_unscaled['Significant_Wave_Height'].values
        else:
            hs_unscaled = np.zeros(len(X_test_unscaled))

        results_df = pd.DataFrame({
            'Speed-Through-Water': speed_unscaled,
            'Draft_Fore': draft_fore_unscaled,
            'Draft_Aft': draft_aft_unscaled,
            'Significant_Wave_Height': hs_unscaled,
            'Actual Power': y_actual,
            'Predicted Power': y_pred
        })

        # Save predictions
        output_csv_file = 'power_predictions.csv'
        results_df.to_csv(output_csv_file, index=False)
        print(f"Predictions saved to {output_csv_file}")

        rmse = np.sqrt(np.mean((y_actual - y_pred) ** 2))
        mae = np.mean(np.abs(y_actual - y_pred))
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test MAE: {mae:.4f}")
    else:
        print("Error in loading and preparing data.")