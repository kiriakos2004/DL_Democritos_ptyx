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
    """Function to customly initialize weights for consistent comparison."""
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            # Use Kaiming uniform initialization for the linear layers
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

class ShipSpeedPredictorModel:
    def __init__(self, input_size, lr=0.001, epochs=100, batch_size=32,
                 optimizer_choice='Adam', loss_function_choice='MSE'):
        self.lr = lr  # Part of hyperparameter search
        self.epochs = epochs  # Manually specified
        self.batch_size = batch_size  # Part of hyperparameter search
        self.optimizer_choice = optimizer_choice  # Manually specified
        self.loss_function_choice = loss_function_choice  # Manually specified
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

    def prepare_dataloader(self, X, y):
        """Function to prepare the DataLoader from data and move tensors to the device."""
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1).to(self.device)

        # Create DataLoader for batching
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        return loader

    def train(self, train_loader):
        """Function to train the model."""
        optimizer = self.get_optimizer()
        loss_function = self.get_loss_function()

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0

            # Determine the total number of batches
            total_batches = len(train_loader)

            # Progress bar for each epoch
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{self.epochs}",
                leave=True,
                total=total_batches
            )

            for batch_index, (X_batch, y_batch) in enumerate(progress_bar):
                optimizer.zero_grad()  # Zero out the gradients
                outputs = self.model(X_batch)  # Forward pass

                # Data-driven loss
                data_loss = loss_function(outputs, y_batch)

                # Backward pass and optimization
                data_loss.backward()
                optimizer.step()

                # Update running loss
                running_loss += data_loss.item()

                # Update progress bar with current loss
                progress_bar.set_postfix({
                    "Loss": f"{running_loss/total_batches:.4f}"
                })

            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {running_loss/total_batches:.4f}")

    def evaluate(self, X_eval, y_eval, dataset_type="Validation", data_processor=None):
        """Function to evaluate the model on the given dataset (validation or test)."""
        self.model.eval()  # Set the model to evaluation mode
        X_eval_tensor = torch.tensor(X_eval.values, dtype=torch.float32).to(self.device)
        y_eval_tensor = torch.tensor(y_eval.values, dtype=torch.float32).view(-1, 1).to(self.device)

        loss_function = self.get_loss_function()
        with torch.no_grad():
            outputs = self.model(X_eval_tensor)
            loss = loss_function(outputs, y_eval_tensor)
            print(f"\n{dataset_type} Loss: {loss.item():.4f}")

            if data_processor:
                # Inverse transform outputs and y_eval to original scale
                outputs_original = data_processor.inverse_transform_y(outputs.cpu().numpy())
                y_eval_original = data_processor.inverse_transform_y(y_eval_tensor.cpu().numpy())

                # Calculate evaluation metrics (e.g., RMSE)
                rmse = np.sqrt(np.mean((outputs_original - y_eval_original) ** 2))
                print(f"{dataset_type} RMSE: {rmse:.2f}")

        return loss.item()

    def cross_validate(self, X, y, data_processor, k_folds=5):
        """Function to perform cross-validation on the model using training and validation data."""
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            print(f"\nFold {fold+1}/{k_folds}")

            # Split the data into training and validation sets
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Prepare the data loaders
            train_loader = self.prepare_dataloader(X_train, y_train)

            # Reset model weights for each fold
            self.model.apply(self.reset_weights)

            # Train the model on the training split
            self.train(train_loader)

            # Evaluate the model on the validation split
            val_loss = self.evaluate(X_val, y_val, dataset_type="Validation", data_processor=data_processor)
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
    def hyperparameter_search(X_train, y_train, param_grid, epochs, optimizer, loss_function, data_processor, k_folds=5):
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
                batch_size=batch_size
            )

            # Perform cross-validation
            avg_val_loss = model.cross_validate(
                X_train, y_train, data_processor, k_folds=k_folds
            )

            # Update the best combination if this one is better
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_params = {'lr': lr, 'batch_size': batch_size}

        print(f"\nBest parameters: {best_params}, with average validation loss: {best_loss:.4f}")
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
        print(f"y_train shape: {y_train.shape}")

        # Define hyperparameter grid (search for learning rate and batch size only)
        param_grid = {
            'lr': [0.001, 0.01],        # Learning rate values to search
            'batch_size': [32, 64]      # Batch size values to search
        }

        # Manually specify other hyperparameters
        epochs = 8
        optimizer = 'Adam'
        loss_function = 'MSE'

        # Perform hyperparameter search with cross-validation
        best_params, best_loss = ShipSpeedPredictorModel.hyperparameter_search(
            X_train, y_train, param_grid, epochs, optimizer, loss_function, data_processor, k_folds=5
        )

        # Train the final model with the best hyperparameters
        final_model = ShipSpeedPredictorModel(
            input_size=X_train.shape[1],
            lr=best_params['lr'],                    # Best learning rate
            epochs=epochs,                           # Manually specified
            optimizer_choice=optimizer,              # Manually specified
            loss_function_choice=loss_function,      # Manually specified
            batch_size=best_params['batch_size']     # Best batch size
        )

        # Prepare the data loaders for the final model
        final_train_loader = final_model.prepare_dataloader(X_train, y_train)

        # Train the final model on the entire training set
        final_model.train(final_train_loader)

        # Evaluate the final model on the test set (after hyperparameter tuning)
        final_model.evaluate(X_test, y_test, dataset_type="Test", data_processor=data_processor)