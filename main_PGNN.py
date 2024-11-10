import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from read_data import DataProcessor

# Custom weight initialization function
def initialize_weights(model):
    """Function to customly initialize weights in order to make results between PINN and no_PINN more comparable."""
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            # Use Kaiming uniform initialization for the linear layers
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

class ShipSpeedPredictorModel:
    def __init__(self, input_size, lr=0.001, epochs=100, batch_size=32,
                 optimizer_choice='Adam', loss_function_choice='MSE', alpha=1.0, beta=0.1):
        self.lr = lr  # Part of hyperparameter search
        self.epochs = epochs  # Manually specified
        self.batch_size = batch_size  # Part of hyperparameter search
        self.optimizer_choice = optimizer_choice  # Manually specified
        self.loss_function_choice = loss_function_choice  # Manually specified
        self.alpha = alpha  # Manually specified
        self.beta = beta    # Manually specified
        self.device = self.get_device()

        # Set random seed for reproducibility
        torch.manual_seed(42)

        # Initialize the model
        self.model = self.ShipSpeedPredictor(input_size).to(self.device)
        initialize_weights(self.model)  # Apply custom initialization

    class ShipSpeedPredictor(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 32)
            self.fc4 = nn.Linear(32, 16)
            self.fc5 = nn.Linear(16, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = torch.relu(self.fc4(x))
            x = self.fc5(x)
            return x

    def get_device(self):
        """Function to check if a GPU is available (MPS for Apple Silicon or CUDA for NVIDIA) and return the appropriate device."""
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
            return optim.LBFGS(self.model.parameters(), lr=self.lr, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
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
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

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
        unscaled_loader = DataLoader(unscaled_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        return unscaled_loader

    def calculate_physics_loss(self, V, trim, predicted_power_scaled, rho, S, S_APP, A_t,
                               C_a, k, STWAVE1, alpha_trim, eta_D, L, nu, g, L_t, data_processor):
        """Use unscaled values for physics calculations, with speed-dependent C_f and F_nt."""
        # Physics constants
        g = torch.tensor(g, device=V.device, dtype=V.dtype)
        L_t = torch.tensor(L_t, device=V.device, dtype=V.dtype)
        L = torch.tensor(L, device=V.device, dtype=V.dtype)
        nu = torch.tensor(nu, device=V.device, dtype=V.dtype)

        # Ensure V is not zero to avoid division by zero errors
        V = torch.clamp(V, min=1e-5)

        # Calculate Reynolds number Re
        Re = V * L / nu

        # Avoid log of zero or negative numbers
        Re = torch.clamp(Re, min=1e-5)

        # Calculate frictional resistance coefficient C_f using ITTC-1957 formula
        C_f = 0.075 / (torch.log10(Re) - 2) ** 2

        # Frictional Resistance (R_F)
        R_F = 0.5 * rho * V**2 * S * C_f

        # Wave-Making Resistance (R_W)
        STWAVE2 = 1 + alpha_trim * trim  # Dynamic correction factor involving trim
        C_W = STWAVE1 * STWAVE2
        R_W = 0.5 * rho * V**2 * S * C_W

        # Appendage Resistance (R_APP)
        R_APP = 0.5 * rho * V**2 * S_APP * C_f

        # Calculate F_nt (Transom Froude Number)
        F_nt = V / torch.sqrt(g * L_t)

        # Transom Stern Resistance (R_TR)
        R_TR = 0.5 * rho * V**2 * A_t * (1 - F_nt)

        # Correlation Allowance Resistance (R_C)
        R_C = 0.5 * rho * V**2 * S * C_a

        # Total Resistance (R_T)
        R_T = R_F * (1 + k) + R_W + R_APP + R_TR + R_C

        # Calculate shaft power (P_S)
        P_S = ((V * R_T) / eta_D) / 1000  # Convert to kilowatts if necessary

        # Convert P_S to a DataFrame with the same column name as the target
        P_S_df = pd.DataFrame(P_S.cpu().detach().numpy(), columns=[data_processor.target_column])

        # Scale P_S using the same scaler as the target variable
        P_S_scaled = data_processor.scaler_y.transform(P_S_df).flatten()
        P_S_scaled = torch.tensor(P_S_scaled, dtype=V.dtype, device=V.device)

        # Compute physics loss in scaled space to match data loss scale
        physics_loss = (predicted_power_scaled.squeeze() - P_S_scaled) ** 2

        return physics_loss, P_S_scaled

    def train(self, train_loader, unscaled_data_loader, feature_indices, data_processor):
        """Function to train the model, now including the physics-based loss."""
        optimizer = self.get_optimizer()
        loss_function = self.get_loss_function()

        # Constants for physics-based loss
        rho = 1025.0      # Water density (kg/m³)
        S = 9950.0        # Wetted surface area in m²
        S_APP = 150.0     # Wetted surface area of appendages in m²
        A_t = 50.0        # Transom area in m²
        C_a = 0.00045     # Correlation allowance coefficient
        k = 0.15          # Form factor (dimensionless)
        STWAVE1 = 0.001   # Base wave resistance coefficient
        alpha_trim = 0.1  # Effect of trim on wave resistance
        eta_D = 0.93      # Propulsive efficiency
        L = 230.0         # Ship length in meters
        nu = 1e-6         # Kinematic viscosity of water (m²/s)
        g = 9.81          # Gravitational acceleration (m/s²)
        L_t = 20.0        # Transom length in meters

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            running_data_loss = 0.0
            running_physics_loss = 0.0

            if self.optimizer_choice == 'LBFGS':
                # For LBFGS, process the entire dataset as a single batch
                X_batch, y_batch = train_loader.dataset.tensors
                X_unscaled_batch = unscaled_data_loader.dataset.tensors[0]

                def closure():
                    optimizer.zero_grad()
                    outputs = self.model(X_batch)  # Forward pass

                    # Data-driven loss
                    data_loss = loss_function(outputs, y_batch)

                    # Extract speed and trim from unscaled features for physics-based loss
                    speed_idx = feature_indices['Speed-Through-Water']  # Adjust as necessary
                    fore_draft_idx = feature_indices['Draft_Fore']      # Adjust as necessary
                    aft_draft_idx = feature_indices['Draft_Aft']        # Adjust as necessary

                    V = X_unscaled_batch[:, speed_idx]  # Speed in knots (assuming the data is in knots)
                    trim = X_unscaled_batch[:, fore_draft_idx] - X_unscaled_batch[:, aft_draft_idx]  # Trim in meters

                    # Convert V to m/s if necessary (e.g., if V is in knots)
                    V = V * 0.51444  # Convert knots to m/s

                    # Physics-based loss
                    physics_loss, P_S_scaled = self.calculate_physics_loss(
                        V, trim, outputs, rho, S, S_APP, A_t, C_a, k,
                        STWAVE1, alpha_trim, eta_D, L, nu, g, L_t, data_processor
                    )

                    # Combine the losses using hyperparameters alpha and beta
                    total_loss = self.alpha * data_loss + self.beta * torch.mean(physics_loss)

                    # Backward pass
                    total_loss.backward()
                    return total_loss

                optimizer.step(closure)
                total_loss = closure()
                data_loss_value = (self.alpha * loss_function(self.model(X_batch), y_batch)).item()
                physics_loss_value = (self.beta * torch.mean(self.calculate_physics_loss(
                    X_unscaled_batch[:, feature_indices['Speed-Through-Water']] * 0.51444,
                    X_unscaled_batch[:, feature_indices['Draft_Fore']] - X_unscaled_batch[:, feature_indices['Draft_Aft']],
                    self.model(X_batch),
                    rho, S, S_APP, A_t, C_a, k, STWAVE1, alpha_trim, eta_D, L, nu, g, L_t, data_processor
                )[0])).item()

                running_loss = total_loss.item()
                running_data_loss = data_loss_value
                running_physics_loss = physics_loss_value

                # Update progress bar
                progress_bar = tqdm(total=1, desc=f"Epoch {epoch+1}/{self.epochs}", leave=True)
                progress_bar.set_postfix({
                    "Total Loss": f"{running_loss:.8f}",
                    "Data Loss": f"{running_data_loss:.8f}",
                    "Physics Loss": f"{running_physics_loss:.8f}"
                })
                progress_bar.update(1)
                progress_bar.close()

            else:
                # Determine the total number of batches
                total_batches = len(train_loader)

                # Progress bar for each epoch
                progress_bar = tqdm(
                    zip(train_loader, unscaled_data_loader),
                    desc=f"Epoch {epoch+1}/{self.epochs}",
                    leave=True,
                    total=total_batches
                )

                for batch_index, ((X_batch, y_batch), (X_unscaled_batch,)) in enumerate(progress_bar):
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    X_unscaled_batch = X_unscaled_batch.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(X_batch)  # Forward pass

                    # Data-driven loss
                    data_loss = loss_function(outputs, y_batch)

                    # Extract speed and trim from unscaled features for physics-based loss
                    speed_idx = feature_indices['Speed-Through-Water']  # Adjust as necessary
                    fore_draft_idx = feature_indices['Draft_Fore']      # Adjust as necessary
                    aft_draft_idx = feature_indices['Draft_Aft']        # Adjust as necessary

                    V = X_unscaled_batch[:, speed_idx]  # Speed in knots (assuming the data is in knots)
                    trim = X_unscaled_batch[:, fore_draft_idx] - X_unscaled_batch[:, aft_draft_idx]  # Trim in meters

                    # Convert V to m/s if necessary (e.g., if V is in knots)
                    V = V * 0.51444  # Convert knots to m/s

                    # Physics-based loss
                    physics_loss, P_S_scaled = self.calculate_physics_loss(
                        V, trim, outputs, rho, S, S_APP, A_t, C_a, k,
                        STWAVE1, alpha_trim, eta_D, L, nu, g, L_t, data_processor
                    )

                    # Combine the losses using hyperparameters alpha and beta
                    total_loss = self.alpha * data_loss + self.beta * torch.mean(physics_loss)

                    # Backward pass and optimization
                    total_loss.backward()
                    optimizer.step()

                    # Update running losses
                    running_loss += total_loss.item()
                    running_data_loss += data_loss.item()
                    running_physics_loss += physics_loss.mean().item()

                    # Update progress bar with current losses
                    progress_bar.set_postfix({
                        "Total Loss": f"{running_loss / (batch_index + 1):.8f}",
                        "Data Loss": f"{running_data_loss / (batch_index + 1):.8f}",
                        "Physics Loss": f"{running_physics_loss / (batch_index + 1):.8f}"
                    })

                print(f"Epoch [{epoch+1}/{self.epochs}], Total Loss: {running_loss / total_batches:.8f}, "
                      f"Data Loss: {running_data_loss / total_batches:.8f}, "
                      f"Physics Loss: {running_physics_loss / total_batches:.8f}")

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

            # Reset model weights for each fold
            self.model.apply(self.reset_weights)

            # Train the model on the training split
            self.train(train_loader, unscaled_data_loader, feature_indices, data_processor)

            # Evaluate the model on the validation split
            val_loss = self.evaluate(X_val_fold, y_val_fold, dataset_type="Validation", data_processor=data_processor)
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
                              param_grid, epochs, optimizer, loss_function, alpha, beta, data_processor, k_folds=5):
        """Function to perform hyperparameter search with cross-validation."""
        best_params = None
        best_loss = float('inf')

        # Generate all combinations of hyperparameters
        hyperparameter_combinations = list(product(
            param_grid['lr'],
            param_grid['batch_size']
        ))

        for lr, batch_size in hyperparameter_combinations:
            print(f"\nTesting combination: lr={lr}, batch_size={batch_size}")

            # Initialize model with the current hyperparameters
            model = ShipSpeedPredictorModel(
                input_size=X_train.shape[1],
                lr=lr,
                epochs=epochs,
                optimizer_choice=optimizer,
                loss_function_choice=loss_function,
                batch_size=batch_size,
                alpha=alpha,
                beta=beta
            )

            # Perform cross-validation
            avg_val_loss = model.cross_validate(
                X_train, X_train_unscaled, y_train, feature_indices, data_processor, k_folds=k_folds
            )

            # Update the best combination if this one is better
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_params = {'lr': lr, 'batch_size': batch_size}

        print(f"\nBest parameters: {best_params}, with average validation loss: {best_loss:.8f}")

        # Save the best hyperparameters to a text file
        with open("best_hyperparameters_PGNN.txt", "w") as f:
            f.write(f"Best parameters: {best_params}\n")
            f.write(f"Best average validation loss: {best_loss:.8f}\n")

        return best_params, best_loss

if __name__ == "__main__":
    # Load data using the DataProcessor class
    data_processor = DataProcessor(
        file_path='data/Aframax/P data_20200213-20200726_Democritos.csv',
        target_column='Power',
        drop_columns=['TIME']
    )
    result = data_processor.load_and_prepare_data()
    if result is not None:
        X_train, X_test, X_train_unscaled, X_test_unscaled, y_train, y_test, y_train_unscaled, y_test_unscaled = result

        # Print dataset shapes
        print(f"X_train shape: {X_train.shape}")
        print(f"X_train_unscaled shape: {X_train_unscaled.shape}")
        print(f"y_train shape: {y_train.shape}")

        # Ensure that the columns are in the same order in scaled and unscaled data
        assert list(X_train.columns) == list(X_train_unscaled.columns), "Column mismatch between scaled and unscaled data"

        # Create a mapping from feature names to indices
        feature_indices = {col: idx for idx, col in enumerate(X_train_unscaled.columns)}

        # Check if necessary columns are present
        required_columns = ['Speed-Through-Water', 'Draft_Fore', 'Draft_Aft']  # Replace with your actual column names
        for col in required_columns:
            if col not in feature_indices:
                raise ValueError(f"Required column '{col}' not found in data")

        # Define hyperparameter grid (search for learning rate and batch size only)
        param_grid = {
            'lr': [0.001, 0.01],        # Learning rate values to search
            'batch_size': [64, 128]      # Batch size values to search
        }

        # Manually specify other hyperparameters
        epochs = 8
        optimizer = 'Adam'
        loss_function = 'MSE'
        alpha = 0.8   # Manually specified
        beta = 0.2    # Manually specified

        # Perform hyperparameter search with cross-validation
        best_params, best_loss = ShipSpeedPredictorModel.hyperparameter_search(
            X_train, X_train_unscaled, y_train, feature_indices,
            param_grid, epochs, optimizer, loss_function, alpha, beta, data_processor, k_folds=5
        )

        # Train the final model with the best hyperparameters
        final_model = ShipSpeedPredictorModel(
            input_size=X_train.shape[1],
            lr=best_params['lr'],                    # Best learning rate
            epochs=epochs,                           # Manually specified
            optimizer_choice=optimizer,              # Manually specified
            loss_function_choice=loss_function,      # Manually specified
            batch_size=best_params['batch_size'],    # Best batch size
            alpha=alpha,                             # Manually specified alpha
            beta=beta                                # Manually specified beta
        )

        # Prepare the data loaders for the final model
        final_train_loader = final_model.prepare_dataloader(X_train, y_train)
        final_unscaled_loader = final_model.prepare_unscaled_dataloader(X_train_unscaled)

        # Train the final model on the entire training set
        final_model.train(final_train_loader, final_unscaled_loader, feature_indices, data_processor)

        # Evaluate the final model on the test set (after hyperparameter tuning)
        final_model.evaluate(X_test, y_test, dataset_type="Test", data_processor=data_processor)