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
from read_data_fragment import DataProcessor
import matplotlib.pyplot as plt
import json

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
        def __init__(self, input_size, dropout_rate=0.2):
            super().__init__()
            self.fc1 = nn.Linear(input_size, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, 64)
            self.fc5 = nn.Linear(64, 32)
            self.fc6 = nn.Linear(32, 16)
            self.fc7 = nn.Linear(16, 1)

            # Add Dropout layers
            self.dropout = nn.Dropout(p=dropout_rate)

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
            return optim.LBFGS(self.model.parameters(), lr=self.lr, max_iter=500, history_size=10, line_search_fn="strong_wolfe")
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
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)

        return loader

    def train(self, train_loader, val_loader=None, live_plot=False, patience=10, min_delta=1e-4):
        """Function to train the model with early stopping."""
        optimizer = self.get_optimizer()
        loss_function = self.get_loss_function()

        # Lists to store loss values
        train_losses = []
        val_losses = []

        # Early stopping variables
        best_val_loss = float('inf')
        epochs_no_improve = 0

        # Initialize live plotting if enabled
        if live_plot:
            plt.ion()  # Enable interactive mode
            fig, ax = plt.subplots()

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0

            # Progress bar for each epoch
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{self.epochs}",
                leave=False
            )

            for X_batch, y_batch in progress_bar:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                data_loss = loss_function(outputs, y_batch)
                data_loss.backward()
                optimizer.step()

                # Update running loss
                running_loss += data_loss.item()

            # Compute average training loss for the epoch
            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)

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
                print(f"Epoch [{epoch+1}/{self.epochs}], Training Loss: {avg_train_loss:.8f}, Validation Loss: {avg_val_loss:.8f}")

                # Early stopping logic
                if avg_val_loss < best_val_loss - min_delta:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break
            else:
                val_losses.append(None)
                print(f"Epoch [{epoch+1}/{self.epochs}], Training Loss: {avg_train_loss:.8f}")

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

    def cross_validate(self, X, y, data_processor, k_folds=5):
        """Function to perform cross-validation on the model using training and validation data."""
        kfold = KFold(n_splits=k_folds)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            print(f"\nFold {fold+1}/{k_folds}")

            # Split the data into training and validation sets
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            # Prepare the data loaders
            train_loader = self.prepare_dataloader(X_train_fold, y_train_fold)
            val_loader = self.prepare_dataloader(X_val_fold, y_val_fold)

            # Reset model weights for each fold
            self.model.apply(self.reset_weights)

            # Train the model on the training split without live plotting
            self.train(train_loader, val_loader=val_loader, live_plot=False)

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
    def hyperparameter_search(X_train, y_train, param_grid, epochs_cv, optimizer, loss_function, data_processor, k_folds=5):
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
                epochs=epochs_cv,  # Use epochs for cross-validation
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

        print(f"\nBest parameters: {best_params}, with average validation loss: {best_loss:.8f}")

        # Save the best hyperparameters to a JSON file
        with open('best_hyperparameters.json', 'w') as f:
            json.dump({'best_params': best_params, 'best_loss': best_loss}, f)

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
        X_train, X_test, X_train_unscaled, X_test_unscaled, y_train, y_test, y_train_unscaled, y_test_unscaled = result

        # Print dataset shapes
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")

        # Define hyperparameter grid (search for learning rate and batch size only)
        param_grid = {
            'lr': [0.001, 0.01],  # Learning rate values to search
            'batch_size': [512]    # Batch size values to search
        }

        # Manually specify other hyperparameters
        epochs_cv = 50      # Number of epochs during cross-validation
        epochs_final = 500  # Number of epochs during final training
        optimizer = 'Adam'
        loss_function = 'MSE'

        # Perform hyperparameter search with cross-validation
        best_params, best_loss = ShipSpeedPredictorModel.hyperparameter_search(
            X_train, y_train, param_grid, epochs_cv, optimizer, loss_function, data_processor, k_folds=5
        )

        # Split X_train and y_train into training and validation sets for final training
        X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
            X_train, y_train, test_size=0.1, shuffle=True
        )

        # Train the final model with the best hyperparameters
        final_model = ShipSpeedPredictorModel(
            input_size=X_train.shape[1],
            lr=best_params['lr'],                    # Best learning rate
            epochs=epochs_final,                     # Number of epochs for final training
            optimizer_choice=optimizer,              # Manually specified
            loss_function_choice=loss_function,      # Manually specified
            batch_size=best_params['batch_size']     # Best batch size
        )

        # Prepare the data loaders for the final model
        final_train_loader = final_model.prepare_dataloader(X_train_final, y_train_final)
        final_val_loader = final_model.prepare_dataloader(X_val_final, y_val_final)

        # Train the final model on the training set, with validation data and live plotting enabled
        final_model.train(final_train_loader, val_loader=final_val_loader, live_plot=True)

        # Prepare the test data for predictions
        X_test_unscaled = X_test_unscaled[X_train_unscaled.columns]
        X_test_scaled = data_processor.scaler_X.transform(X_test_unscaled)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(final_model.device)

        # Make predictions on the test set
        final_model.model.eval()
        with torch.no_grad():
            y_pred_scaled = final_model.model(X_test_tensor)
            y_pred_scaled_np = y_pred_scaled.cpu().numpy()
            y_pred = data_processor.inverse_transform_y(y_pred_scaled_np).flatten()

        # Get actual power values from the unscaled test target
        y_actual = y_test_unscaled.values.flatten()

        # Create a DataFrame with actual and predicted power
        results_df = pd.DataFrame({
            'Actual Power': y_actual,
            'Predicted Power': y_pred
        })

        # Save the results to a CSV file
        output_csv_file = 'power_predictions.csv'
        results_df.to_csv(output_csv_file, index=False)
        print(f"Predictions saved to {output_csv_file}")

        # Calculate and print evaluation metrics
        rmse = np.sqrt(np.mean((y_actual - y_pred) ** 2))
        mae = np.mean(np.abs(y_actual - y_pred))
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test MAE: {mae:.4f}")

    else:
        print("Error in loading and preparing data.")