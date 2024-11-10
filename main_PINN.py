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
    def __init__(self, input_size, lr=0.001, epochs=100, batch_size=32,
                 optimizer_choice='Adam', loss_function_choice='MSE', alpha=1.0, beta=0.1, gamma=0.1):
        self.lr = lr  # Part of hyperparameter search
        self.epochs = epochs  # Manually specified
        self.batch_size = batch_size  # Part of hyperparameter search
        self.optimizer_choice = optimizer_choice  # Manually specified
        self.loss_function_choice = loss_function_choice  # Manually specified
        self.alpha = alpha  # Weight for data loss, part of hyperparameter search
        self.beta = beta    # Weight for physics loss, part of hyperparameter search
        self.gamma = gamma  # Weight for boundary condition loss, part of hyperparameter search
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
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        return loader

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
        x_collocation = torch.tensor(x_collocation_scaled, dtype=torch.float32, device=self.device)
        x_collocation.requires_grad = True
        return x_collocation

    def sample_boundary_points(self, num_points, X_train_unscaled, feature_indices, data_processor):
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
                x_boundary_unscaled[col] = np.random.uniform(low=x_min[col], high=x_max[col], size=num_points)

        # Scale the boundary points
        x_boundary_scaled = data_processor.scaler_X.transform(x_boundary_unscaled)
        x_boundary = torch.tensor(x_boundary_scaled, dtype=torch.float32, device=self.device)
        x_boundary.requires_grad = True
        return x_boundary

    def compute_pde_residual(self, x_collocation, feature_indices):
        """Function to compute PDE residuals at collocation points."""
        x_collocation.requires_grad = True
        outputs = self.model(x_collocation)

        # Compute derivative of outputs with respect to x_collocation
        outputs_x = torch.autograd.grad(
            outputs=outputs,
            inputs=x_collocation,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True
        )[0]

        # 'Speed-Through-Water' is V
        V_idx = feature_indices['Speed-Through-Water']
        V = x_collocation[:, V_idx].view(-1, 1)
        outputs_V = outputs_x[:, V_idx].view(-1, 1)

        # Define simplified PDE: dP/dV + aP - bV^2 = 0
        a = torch.tensor(0.1, device=self.device)
        b = torch.tensor(0.2, device=self.device)
        residual = outputs_V + a * outputs - b * V**2

        return residual

    def compute_boundary_loss(self, x_boundary):
        """Function to compute boundary condition loss."""
        outputs_boundary = self.model(x_boundary)
        boundary_loss = torch.mean(outputs_boundary**2)  # Enforce P = 0 when V = 0
        return boundary_loss

    def train(self, train_loader, X_train_unscaled, feature_indices, data_processor, val_loader=None, live_plot=False):
        """Function to train the model, including PDE residuals and boundary conditions."""
        optimizer = self.get_optimizer()
        loss_function = self.get_loss_function()

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
            running_pde_loss = 0.0
            running_boundary_loss = 0.0

            if self.optimizer_choice == 'LBFGS':
                # For LBFGS, process the entire dataset as a single batch
                X_batch, y_batch = train_loader.dataset.tensors

                def closure():
                    optimizer.zero_grad()

                    # Data-driven loss
                    outputs = self.model(X_batch)
                    data_loss = loss_function(outputs, y_batch)

                    # Sample collocation points for PDE residuals
                    x_collocation = self.sample_collocation_points(len(X_batch), X_train_unscaled, data_processor)
                    pde_residual = self.compute_pde_residual(x_collocation, feature_indices)
                    pde_loss = torch.mean(pde_residual**2)

                    # Sample boundary points for boundary condition loss
                    x_boundary = self.sample_boundary_points(len(X_batch), X_train_unscaled, feature_indices, data_processor)
                    boundary_loss = self.compute_boundary_loss(x_boundary)

                    # Combine losses
                    total_loss = self.alpha * data_loss + self.beta * pde_loss + self.gamma * boundary_loss

                    # Backward pass
                    total_loss.backward()
                    return total_loss

                optimizer.step(closure)
                total_loss = closure()
                running_loss = total_loss.item()
                running_data_loss = (self.alpha * loss_function(self.model(X_batch), y_batch)).item()
                running_pde_loss = (self.beta * torch.mean(self.compute_pde_residual(
                    self.sample_collocation_points(len(X_batch), X_train_unscaled, data_processor), feature_indices
                )**2)).item()
                running_boundary_loss = (self.gamma * self.compute_boundary_loss(
                    self.sample_boundary_points(len(X_batch), X_train_unscaled, feature_indices, data_processor)
                )).item()

                # Update progress bar
                progress_bar = tqdm(total=1, desc=f"Epoch {epoch+1}/{self.epochs}", leave=True)
                progress_bar.set_postfix({
                    "Total Loss": f"{running_loss:.8f}",
                    "Data Loss": f"{running_data_loss:.8f}",
                    "PDE Loss": f"{running_pde_loss:.8f}",
                    "Boundary Loss": f"{running_boundary_loss:.8f}"
                })
                progress_bar.update(1)
                progress_bar.close()

                # Store losses for plotting
                train_losses.append(running_loss)
                if val_loader is not None:
                    val_loss = self.evaluate_on_loader(val_loader)
                    val_losses.append(val_loss)
            else:
                total_batches = len(train_loader)

                progress_bar = tqdm(
                    enumerate(train_loader),
                    desc=f"Epoch {epoch+1}/{self.epochs}",
                    leave=True,
                    total=total_batches
                )

                for batch_index, (X_batch, y_batch) in progress_bar:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    optimizer.zero_grad()  # Zero out the gradients

                    # Data-driven loss
                    outputs = self.model(X_batch)
                    data_loss = loss_function(outputs, y_batch)

                    # Sample collocation points for PDE residuals
                    x_collocation = self.sample_collocation_points(self.batch_size, X_train_unscaled, data_processor)
                    pde_residual = self.compute_pde_residual(x_collocation, feature_indices)
                    pde_loss = torch.mean(pde_residual**2)

                    # Sample boundary points for boundary condition loss
                    x_boundary = self.sample_boundary_points(self.batch_size, X_train_unscaled, feature_indices, data_processor)
                    boundary_loss = self.compute_boundary_loss(x_boundary)

                    # Combine losses
                    total_loss = self.alpha * data_loss + self.beta * pde_loss + self.gamma * boundary_loss

                    # Backward pass and optimization
                    total_loss.backward()
                    optimizer.step()

                    # Update running losses
                    running_loss += total_loss.item()
                    running_data_loss += data_loss.item()
                    running_pde_loss += pde_loss.item()
                    running_boundary_loss += boundary_loss.item()

                    # Update progress bar with current losses
                    progress_bar.set_postfix({
                        "Total Loss": f"{running_loss / (batch_index + 1):.8f}",
                        "Data Loss": f"{running_data_loss / (batch_index + 1):.8f}",
                        "PDE Loss": f"{running_pde_loss / (batch_index + 1):.8f}",
                        "Boundary Loss": f"{running_boundary_loss / (batch_index + 1):.8f}"
                    })

                # Compute average losses for the epoch
                avg_total_loss = running_loss / total_batches
                avg_data_loss = running_data_loss / total_batches
                avg_pde_loss = running_pde_loss / total_batches
                avg_boundary_loss = running_boundary_loss / total_batches

                # Store losses for plotting
                train_losses.append(avg_total_loss)

                # Compute validation loss if validation data is provided
                if val_loader is not None:
                    val_loss = self.evaluate_on_loader(val_loader)
                    val_losses.append(val_loss)
                    print(f"Epoch [{epoch+1}/{self.epochs}], Total Loss: {avg_total_loss:.8f}, "
                          f"Validation Loss: {val_loss:.8f}")
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
            val_loader = self.prepare_dataloader(X_val_fold, y_val_fold)

            # Reset model weights for each fold
            self.model.apply(self.reset_weights)

            # Train the model on the training split without live plotting
            self.train(train_loader, X_train_unscaled_fold, feature_indices, data_processor, val_loader=val_loader, live_plot=False)

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
        with open("best_hyperparameters_PINN.txt", "w") as f:
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
        required_columns = ['Speed-Through-Water', 'Draft_Fore', 'Draft_Aft']
        for col in required_columns:
            if col not in feature_indices:
                raise ValueError(f"Required column '{col}' not found in data")

        # Define hyperparameter grid (search for learning rate, batch size, alpha, beta, gamma)
        param_grid = {
            'lr': [0.001, 0.01],        # Learning rate values to search
            'batch_size': [128, 256],    # Batch size values to search
            'alpha': [0.8, 1.0],        # Alpha values to search
            'beta': [0.1, 0.2],         # Beta values to search
            'gamma': [0.1, 0.2]         # Gamma values to search
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

        # Create the final model instance
        final_model = ShipSpeedPredictorModel(
            input_size=X_train.shape[1],
            lr=best_params['lr'],                    # Best learning rate
            epochs=epochs_final,                     # Number of epochs for final training
            optimizer_choice=optimizer,              # Manually specified
            loss_function_choice=loss_function,      # Manually specified
            batch_size=best_params['batch_size'],    # Best batch size
            alpha=best_params['alpha'],              # Best alpha
            beta=best_params['beta'],                # Best beta
            gamma=best_params['gamma']               # Best gamma
        )

        # Prepare the data loaders for the final model
        final_train_loader = final_model.prepare_dataloader(X_train_final, y_train_final)
        final_val_loader = final_model.prepare_dataloader(X_val_final, y_val_final)

        # Train the final model on the training set, with validation data and live plotting enabled
        final_model.train(
            final_train_loader, X_train_unscaled_final, feature_indices, data_processor,
            val_loader=final_val_loader, live_plot=True
        )

        # Evaluate the final model on the test set (after hyperparameter tuning)
        final_model.evaluate(X_test, y_test, dataset_type="Test", data_processor=data_processor)
