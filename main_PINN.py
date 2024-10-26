import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd  # Ensure pandas is imported
from itertools import product
from tqdm import tqdm
from read_data import DataProcessor

class ShipSpeedPredictorModel:
    def __init__(self, input_size, lr=0.001, epochs=100, batch_size=32, optimizer_choice='Adam', loss_function_choice='MSE'):
        self.lr = lr  # Part of hyperparameter search
        self.epochs = epochs  # Manually specified
        self.batch_size = batch_size  # Part of hyperparameter search
        self.optimizer_choice = optimizer_choice  # Manually specified
        self.loss_function_choice = loss_function_choice  # Manually specified
        self.device = self.get_device()

        # Initialize the model
        self.model = self.ShipSpeedPredictor(input_size).to(self.device)

    class ShipSpeedPredictor(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 32)
            self.fc4 = nn.Linear(32, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
            return x

    def get_device(self):
        """Function to check if GPU is available and return the appropriate device."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        return device

    def get_optimizer(self):
        """Function to get the optimizer based on user choice."""
        if self.optimizer_choice == 'Adam':
            return optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer_choice == 'SGD':
            return optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        elif self.optimizer_choice == 'RMSprop':
            return optim.RMSprop(self.model.parameters(), lr=self.lr)
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

    def prepare_dataloader(self, X_train, y_train):
        """Function to prepare the DataLoader from training data and move tensors to the device."""
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(self.device)

        # Create DataLoader for batching using maximum CPU cores for parallel data loading
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count())  # Use all cores

        return train_loader

    def prepare_unscaled_dataloader(self, X_train_unscaled):
        """Function to prepare the DataLoader for unscaled data."""
        X_train_unscaled_tensor = torch.tensor(X_train_unscaled.values, dtype=torch.float32).to(self.device)
        unscaled_dataset = TensorDataset(X_train_unscaled_tensor)
        unscaled_loader = DataLoader(unscaled_dataset, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count())

        return unscaled_loader

    def calculate_physics_loss(self, V, trim, predicted_power, rho, S, S_APP, A_t, F_nt, C_f, C_a, k, STWAVE1, alpha, eta_D):
        """Use unscaled values for physics calculations"""
        # Frictional Resistance (R_F)
        R_F = 0.5 * rho * V**2 * S * C_f

        # Wave-Making Resistance (R_W)
        STWAVE2 = 1 + alpha * trim  # Dynamic correction factor involving trim
        C_W = STWAVE1 * STWAVE2
        R_W = 0.5 * rho * V**2 * S * C_W

        # Appendage Resistance (R_APP)
        R_APP = 0.5 * rho * V**2 * S_APP * C_f

        # Transom Stern Resistance (R_TR)
        R_TR = 0.5 * rho * V**2 * A_t * (1 - F_nt)

        # Correlation Allowance Resistance (R_C)
        R_C = 0.5 * rho * V**2 * S * C_a

        # Total Resistance (R_T)
        R_T = R_F * (1 + k) + R_W + R_APP + R_TR + R_C

        # Calculate shaft power (P_S)
        P_S = (V * R_T) / eta_D

        # Calculate physics-based loss as the squared difference
        physics_loss = (predicted_power - P_S) ** 2

        return physics_loss

    def train(self, train_loader, unscaled_data_loader, alpha=1.0, beta=0.1):
        """Function to train the model, now including the physics-based loss."""
        optimizer = self.get_optimizer()
        loss_function = self.get_loss_function()

        # Constants for physics-based loss
        rho = 1025.0  # Water density (kg/m³)
        S = 3000.0  # Wetted surface area in m² (example value)
        S_APP = 150.0  # Wetted surface area of appendages in m² (example value)
        A_t = 50.0  # Transom area in m² (example value)
        F_nt = 0.3  # Transom Froude number (example value)
        C_f = 0.0025  # Frictional resistance coefficient (example value)
        C_a = 0.00045  # Correlation allowance coefficient
        k = 0.15  # Form factor (dimensionless)
        STWAVE1 = 0.001  # Base wave resistance coefficient
        alpha_trim = 0.1  # Coefficient representing the effect of trim on wave resistance
        eta_D = 0.65  # Propulsive efficiency (example value)

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0

            # Progress bar for each epoch, updating on each batch
            progress_bar = tqdm(zip(train_loader, unscaled_data_loader), desc=f"Epoch {epoch+1}/{self.epochs}", leave=True)

            for (X_batch, y_batch), (X_unscaled_batch,) in progress_bar:
                optimizer.zero_grad()  # Zero out the gradients
                outputs = self.model(X_batch)  # Forward pass

                # Data-driven loss
                data_loss = loss_function(outputs, y_batch)

                # Extract speed and trim from unscaled features for physics-based loss
                V = X_unscaled_batch[:, 0]  # Assuming speed is the first feature
                trim = X_unscaled_batch[:, 1] - X_unscaled_batch[:, 2]  # Assuming fore and aft drafts are the next features

                # Physics-based loss
                physics_loss = self.calculate_physics_loss(
                    V, trim, outputs, rho, S, S_APP, A_t, F_nt, C_f, C_a, k, STWAVE1, alpha_trim, eta_D
                )

                # Combine the losses
                total_loss = alpha * data_loss + beta * torch.mean(physics_loss)

                # Backward pass and optimization
                total_loss.backward()
                optimizer.step()
                running_loss += total_loss.item()

                # Update progress bar with current loss
                progress_bar.set_postfix({"Batch Loss": f"{running_loss/len(train_loader):.4f}"})

            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {running_loss/len(train_loader):.4f}")

    def evaluate(self, X_eval, y_eval, dataset_type="Validation"):
        """Function to evaluate the model on the given dataset (validation or test)."""
        self.model.eval()  # Set the model to evaluation mode
        X_eval_tensor = torch.tensor(X_eval.values, dtype=torch.float32).to(self.device)
        y_eval_tensor = torch.tensor(y_eval.values, dtype=torch.float32).view(-1, 1).to(self.device)

        loss_function = self.get_loss_function()
        with torch.no_grad():
            outputs = self.model(X_eval_tensor)
            loss = loss_function(outputs, y_eval_tensor)
            print(f"\n{dataset_type} Loss: {loss.item():.4f}")
        return loss.item()

    def cross_validate(self, X, X_unscaled, y, k_folds=5):
        """Function to perform cross-validation on the model using training and validation data."""
        kfold = KFold(n_splits=k_folds, shuffle=True)
        fold_results = []

        # Convert to pandas DataFrames if they're not already
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(X_unscaled, np.ndarray):
            X_unscaled = pd.DataFrame(X_unscaled)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            print(f"\nFold {fold+1}/{k_folds}")

            # Split the data into training and validation sets
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            X_train_unscaled, X_val_unscaled = X_unscaled.iloc[train_idx], X_unscaled.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Prepare the data loaders
            train_loader = self.prepare_dataloader(X_train, y_train)
            unscaled_data_loader = self.prepare_unscaled_dataloader(X_train_unscaled)

            # Reset model weights for each fold
            self.model.apply(self.reset_weights)

            # Train the model on the training split
            self.train(train_loader, unscaled_data_loader)

            # Evaluate the model on the validation split
            val_loss = self.evaluate(X_val, y_val, dataset_type="Validation")
            fold_results.append(val_loss)

        # Calculate average validation loss across all folds
        avg_val_loss = np.mean(fold_results)
        print(f"\nCross-validation results: Average Validation Loss = {avg_val_loss:.4f}")
        return avg_val_loss

    @staticmethod
    def reset_weights(m):
        """Function to reset weights of the neural network for each fold."""
        if isinstance(m, nn.Linear):
            m.reset_parameters()

    @staticmethod
    def hyperparameter_search(X_train, X_train_unscaled, y_train, param_grid, epochs, optimizer, loss_function, k_folds=5):
        """Function to perform hyperparameter search with cross-validation."""
        best_params = None
        best_loss = float('inf')

        # Generate all combinations of hyperparameters (learning rate and batch size)
        for lr, batch_size in product(param_grid['lr'], param_grid['batch_size']):
            print(f"\nTesting combination: learning rate = {lr}, batch size = {batch_size}")

            # Initialize model with the current learning rate and batch size
            model = ShipSpeedPredictorModel(
                input_size=X_train.shape[1],
                lr=lr,  # Part of hyperparameter search
                epochs=epochs,  # Manually specified
                optimizer_choice=optimizer,  # Manually specified
                loss_function_choice=loss_function,  # Manually specified
                batch_size=batch_size  # Part of hyperparameter search
            )

            # Perform cross-validation
            avg_val_loss = model.cross_validate(X_train, X_train_unscaled, y_train, k_folds=k_folds)

            # Update the best combination if this one is better
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_params = {'lr': lr, 'batch_size': batch_size}

        print(f"\nBest parameters: {best_params}, with average validation loss: {best_loss:.4f}")
        return best_params, best_loss

if __name__ == "__main__":
    # Load data using the DataProcessor class
    data_processor = DataProcessor(
        file_path='data/Hera/P data_20220607-20230127_Democritos.csv',
        target_column='Power',
        drop_columns=['TIME']
    )
    X_train, X_test, X_train_unscaled, X_test_unscaled, y_train, y_test = data_processor.load_and_prepare_data()

    # Define hyperparameter grid (search for learning rate and batch size)
    param_grid = {
        'lr': [0.001, 0.01],  # Learning rate values to search
        'batch_size': [32, 64]  # Batch size values to search
    }

    # Manually specify other hyperparameters
    epochs = 2
    optimizer = 'Adam'
    loss_function = 'MSE'

    # Perform hyperparameter search with cross-validation
    best_params, best_loss = ShipSpeedPredictorModel.hyperparameter_search(
        X_train, X_train_unscaled, y_train, param_grid, epochs, optimizer, loss_function, k_folds=5
    )

    # Train the final model with the best hyperparameters
    final_model = ShipSpeedPredictorModel(
        input_size=X_train.shape[1],
        lr=best_params['lr'],  # Best learning rate
        epochs=epochs,  # Manually specified
        optimizer_choice=optimizer,  # Manually specified
        loss_function_choice=loss_function,  # Manually specified
        batch_size=best_params['batch_size']  # Best batch size
    )

    # Prepare the data loaders for the final model
    final_train_loader = final_model.prepare_dataloader(X_train, y_train)
    final_unscaled_loader = final_model.prepare_unscaled_dataloader(X_train_unscaled)

    # Train the final model on the entire training set
    final_model.train(final_train_loader, final_unscaled_loader)

    # Evaluate the final model on the test set (after hyperparameter tuning)
    final_model.evaluate(X_test, y_test, dataset_type="Test")
