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
from read_data import DataProcessor
import matplotlib.pyplot as plt

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
    def __init__(self, input_size, lr=0.001, epochs=100, batch_size=256,
                 optimizer_choice='Adam', loss_function_choice='MSE', alpha=1.0, beta=0.1, k_wave=0.005):
        self.lr = lr  # Part of hyperparameter search
        self.epochs = epochs  # Manually specified or part of hyperparameter search
        self.batch_size = batch_size  # Part of hyperparameter search
        self.optimizer_choice = optimizer_choice  # Manually specified
        self.loss_function_choice = loss_function_choice  # Manually specified
        self.alpha = alpha  # Weight for data loss, part of hyperparameter search
        self.beta = beta    # Weight for physics loss, part of hyperparameter search
        self.k_wave = k_wave  # Wave resistance coefficient, part of hyperparameter search
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
                               C_a, k, STWAVE1, alpha_trim, eta_D, L, nu, g, L_t,
                               H_s, theta_ship, theta_wave, data_processor):
        """Calculate the physics-based loss, including weather impact."""

        # Physics constants
        g = torch.tensor(g, device=V.device, dtype=V.dtype)
        L_t = torch.tensor(L_t, device=V.device, dtype=V.dtype)
        L = torch.tensor(L, device=V.device, dtype=V.dtype)
        nu = torch.tensor(nu, device=V.device, dtype=V.dtype)

        # Ensure V and H_s are not zero to avoid division by zero errors
        V = torch.clamp(V, min=1e-5)
        H_s = torch.clamp(H_s, min=0.0)

        # Calculate Reynolds number Re
        Re = V * L / nu
        Re = torch.clamp(Re, min=1e-5)

        # Calculate frictional resistance coefficient C_f using ITTC-1957 formula
        C_f = 0.075 / (torch.log10(Re) - 2) ** 2

        # Frictional Resistance (R_F)
        R_F = 0.5 * rho * V**2 * S * C_f

        # Wave-Making Resistance (R_W)
        STWAVE2 = 1 + alpha_trim * trim
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

        # Compute relative wave direction in degrees
        theta_rel_wave = torch.abs(theta_wave - theta_ship) % 360
        theta_rel_wave = torch.where(theta_rel_wave > 180, 360 - theta_rel_wave, theta_rel_wave)
        theta_rel_wave_rad = theta_rel_wave * np.pi / 180  # Convert to radians

        # Use the wave resistance coefficient from the model's parameter
        k_wave = self.k_wave  # Tunable constant based on empirical data

        # Compute added resistance due to waves
        R_AW = 0.5 * rho * V**2 * S * k_wave * H_s**2 * (1 + torch.cos(theta_rel_wave_rad))

        # Total Resistance (R_T)
        R_T = R_F * (1 + k) + R_W + R_APP + R_TR + R_C + R_AW

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

    def train(self, train_loader, unscaled_data_loader, feature_indices, data_processor, val_loader=None, live_plot=False):
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

        # Lists to store loss values
        train_losses = []
        val_losses = []

        # Initialize live plotting if enabled
        if live_plot:
            plt.ion()  # Enable interactive mode
            fig, ax = plt.subplots()

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
                    speed_idx = feature_indices['Speed-Through-Water']
                    fore_draft_idx = feature_indices['Draft_Fore']
                    aft_draft_idx = feature_indices['Draft_Aft']
                    V = X_unscaled_batch[:, speed_idx]  # Speed in knots
                    trim = X_unscaled_batch[:, fore_draft_idx] - X_unscaled_batch[:, aft_draft_idx]

                    # Convert V to m/s
                    V = V * 0.51444  # Convert knots to m/s

                    # Extract weather data
                    H_s = X_unscaled_batch[:, feature_indices['Significant_Wave_Height']]
                    theta_ship = X_unscaled_batch[:, feature_indices['True_Heading']]
                    theta_wave = X_unscaled_batch[:, feature_indices['Mean_Wave_Direction']]

                    # Physics-based loss
                    physics_loss, P_S_scaled = self.calculate_physics_loss(
                        V, trim, outputs, rho, S, S_APP, A_t, C_a, k,
                        STWAVE1, alpha_trim, eta_D, L, nu, g, L_t,
                        H_s, theta_ship, theta_wave, data_processor
                    )

                    # Combine the losses
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
                    rho, S, S_APP, A_t, C_a, k, STWAVE1, alpha_trim, eta_D, L, nu, g, L_t,
                    X_unscaled_batch[:, feature_indices['Significant_Wave_Height']],
                    X_unscaled_batch[:, feature_indices['True_Heading']],
                    X_unscaled_batch[:, feature_indices['Mean_Wave_Direction']],
                    data_processor
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
                    speed_idx = feature_indices['Speed-Through-Water']
                    fore_draft_idx = feature_indices['Draft_Fore']
                    aft_draft_idx = feature_indices['Draft_Aft']
                    V = X_unscaled_batch[:, speed_idx]  # Speed in knots
                    trim = X_unscaled_batch[:, fore_draft_idx] - X_unscaled_batch[:, aft_draft_idx]

                    # Convert V to m/s
                    V = V * 0.51444  # Convert knots to m/s

                    # Extract weather data
                    H_s = X_unscaled_batch[:, feature_indices['Significant_Wave_Height']]
                    theta_ship = X_unscaled_batch[:, feature_indices['True_Heading']]
                    theta_wave = X_unscaled_batch[:, feature_indices['Mean_Wave_Direction']]

                    # Physics-based loss
                    physics_loss, P_S_scaled = self.calculate_physics_loss(
                        V, trim, outputs, rho, S, S_APP, A_t, C_a, k,
                        STWAVE1, alpha_trim, eta_D, L, nu, g, L_t,
                        H_s, theta_ship, theta_wave, data_processor
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

                # Compute average losses for the epoch
                avg_total_loss = running_loss / total_batches
                avg_data_loss = running_data_loss / total_batches
                avg_physics_loss = running_physics_loss / total_batches

                # Store losses for plotting
                train_losses.append(avg_total_loss)

                # Compute validation loss if validation data is provided
                if val_loader is not None:
                    self.model.eval()
                    val_running_loss = 0.0
                    with torch.no_grad():
                        for X_val_batch, y_val_batch in val_loader:
                            X_val_batch, y_val_batch = X_val_batch.to(self.device), y_val_batch.to(self.device)
                            val_outputs = self.model(X_val_batch)
                            val_loss = loss_function(val_outputs, y_val_batch)
                            val_running_loss += val_loss.item()
                    avg_val_loss = val_running_loss / len(val_loader)
                    val_losses.append(avg_val_loss)
                    print(f"Epoch [{epoch+1}/{self.epochs}], Total Loss: {avg_total_loss:.8f}, "
                          f"Validation Loss: {avg_val_loss:.8f}")
                else:
                    val_losses.append(None)
                    print(f"Epoch [{epoch+1}/{self.epochs}], Total Loss: {avg_total_loss:.8f}")

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
            # Optionally, save the plot
            fig.savefig('training_validation_loss_plot.png')

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

            # Reset model weights for each fold
            self.model.apply(self.reset_weights)

            # Train the model on the training split without live plotting
            self.train(train_loader, unscaled_data_loader, feature_indices, data_processor, val_loader=val_loader, live_plot=False)

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
            param_grid['k_wave']  # Include k_wave in combinations
        ))

        for lr, batch_size, alpha, beta, k_wave in hyperparameter_combinations:
            print(f"\nTesting combination: lr={lr}, batch_size={batch_size}, alpha={alpha}, beta={beta}, k_wave={k_wave}")

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
                k_wave=k_wave
            )

            # Perform cross-validation
            avg_val_loss = model.cross_validate(
                X_train, X_train_unscaled, y_train, feature_indices, data_processor, k_folds=k_folds
            )

            # Update the best combination if this one is better
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_params = {'lr': lr, 'batch_size': batch_size, 'alpha': alpha, 'beta': beta, 'k_wave': k_wave}

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
        required_columns = [
            'Speed-Through-Water', 'Draft_Fore', 'Draft_Aft',
            'Significant_Wave_Height', 'True_Heading', 'Mean_Wave_Direction'
        ]
        for col in required_columns:
            if col not in feature_indices:
                raise ValueError(f"Required column '{col}' not found in data")

        # Define hyperparameter grid (search for learning rate, batch size, alpha, beta, k_wave)
        param_grid = {
            'lr': [0.001, 0.01],        # Learning rate values to search
            'batch_size': [126, 256],    # Batch size values to search
            'alpha': [0.8, 1.0],        # Alpha values to search
            'beta': [0.1, 0.05],         # Beta values to search
            'k_wave': [1e-8, 1e-7, 1e-6]  # k_wave values to search
        }

        # Manually specify other hyperparameters
        epochs_cv = 50     # Number of epochs during cross-validation
        epochs_final = 200  # Number of epochs during final training
        optimizer = 'Adam'
        loss_function = 'MSE'

        # Perform hyperparameter search with cross-validation
        best_params, best_loss = ShipSpeedPredictorModel.hyperparameter_search(
            X_train, X_train_unscaled, y_train, feature_indices,
            param_grid, epochs_cv, optimizer, loss_function, data_processor, k_folds=5
        )

        # Split X_train and y_train into training and validation sets for final training
        X_train_final, X_val_final, X_train_unscaled_final, X_val_unscaled_final, y_train_final, y_val_final = train_test_split(
            X_train, X_train_unscaled, y_train, test_size=0.2, random_state=42
        )

        # Create the final model instance first
        final_model = ShipSpeedPredictorModel(
            input_size=X_train.shape[1],
            lr=best_params['lr'],                    # Best learning rate
            epochs=epochs_final,                     # Number of epochs for final training
            optimizer_choice=optimizer,              # Manually specified
            loss_function_choice=loss_function,      # Manually specified
            batch_size=best_params['batch_size'],    # Best batch size
            alpha=best_params['alpha'],              # Best alpha
            beta=best_params['beta'],                # Best beta
            k_wave=best_params['k_wave']             # Best k_wave
        )

        # Prepare the data loaders using the 'final_model' instance
        final_train_loader = final_model.prepare_dataloader(X_train_final, y_train_final)
        final_unscaled_loader = final_model.prepare_unscaled_dataloader(X_train_unscaled_final)
        final_val_loader = final_model.prepare_dataloader(X_val_final, y_val_final)
        # No need for final_val_unscaled_loader unless required

        # Train the final model on the training set, with validation data and live plotting enabled
        final_model.train(
            final_train_loader, final_unscaled_loader, feature_indices, data_processor,
            val_loader=final_val_loader, live_plot=True
        )

        # Evaluate the final model on the test set (after hyperparameter tuning)
        final_model.evaluate(X_test, y_test, dataset_type="Test", data_processor=data_processor)
