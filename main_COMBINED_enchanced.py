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
                 optimizer_choice='Adam', loss_function_choice='MSE',
                 debug_mode=False):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_choice = optimizer_choice
        self.loss_function_choice = loss_function_choice
        self.device = self.get_device()
        self.debug_mode = debug_mode  # Enable or disable debug mode

        # Constants for physics-based loss
        self.rho = 1025.0      # Water density (kg/m³)
        self.nu = 1e-6         # Kinematic viscosity of water (m²/s)
        self.g = 9.81          # Gravitational acceleration (m/s²)

        # Ship dimensions (adjust as per your ship's specifications)
        self.S = 9950.0        # Wetted surface area in m²
        self.L = 264.0         # Ship length in meters
        self.B = 50.0          # Beam (width) of the ship in meters
        self.k = 1.3           # Empirical correction

        self.current_epoch = 0

        self.initial_physics_weight = 5.0   # Physics loss starts with moderate importance
        self.final_physics_weight = 30.0    # Physics loss becomes very important
        self.initial_pde_weight = 2.0       # PDE loss starts with no influence
        self.final_pde_weight = 10.0         # PDE loss becomes moderately important
        self.boundary_weight = 1.0          # Boundary loss stays constant

        # Set random seed for reproducibility
        torch.manual_seed(42)

        # Initialize the model
        self.model = self.ShipSpeedPredictor(input_size).to(self.device)
        initialize_weights(self.model)  # Apply custom initialization

    class ShipSpeedPredictor(nn.Module):
        def __init__(self, input_size, dropout_rate=0.2):
            super().__init__()
            self.fc1 = nn.Linear(input_size, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 64)
            self.fc4 = nn.Linear(64, 32)
            self.fc5 = nn.Linear(32, 16)
            self.fc6 = nn.Linear(16, 1)

            # Add Dropout layers
            self.dropout = nn.Dropout(p=dropout_rate)

            # Initialize trainable parameters with starting values in middle of their ranges
            self.C_wave = nn.Parameter(torch.tensor(0.0008, dtype=torch.float32))
            self.eta_D = nn.Parameter(torch.tensor(0.8, dtype=torch.float32))
            self.C_resid = nn.Parameter(torch.tensor(0.06, dtype=torch.float32))

            # Register bounds as buffers (they won't be trained but will move to correct device)
            self.register_buffer('C_wave_min', torch.tensor(0.0007))
            self.register_buffer('C_wave_max', torch.tensor(0.001))
            self.register_buffer('eta_D_min', torch.tensor(0.7))
            self.register_buffer('eta_D_max', torch.tensor(0.85))
            self.register_buffer('C_resid_min', torch.tensor(0.05))
            self.register_buffer('C_resid_max', torch.tensor(0.07))

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
            x = self.fc6(x)
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
    
    def get_progressive_weight(self, current_epoch, max_epochs, start_weight, end_weight):
        """
        Calculates a weight that smoothly increases from start_weight to end_weight
        during training using a sigmoid curve for smooth transition.
        
        Args:
            current_epoch: Current training epoch number
            max_epochs: Total number of training epochs
            start_weight: Initial weight value
            end_weight: Final weight value
        
        Returns:
            float: Current weight value based on training progress
        """
        progress = current_epoch / max_epochs
        # Using sigmoid function for smooth transition
        transition = 1 / (1 + np.exp(-10 * (progress - 0.5)))
        return start_weight + (end_weight - start_weight) * transition


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
        """Compute PDE residuals at collocation points with scaling adjustments."""
        x_collocation.requires_grad_(True)
        outputs = self.model(x_collocation)

        outputs_x = torch.autograd.grad(
            outputs=outputs,
            inputs=x_collocation,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True
        )[0]

        V_idx = feature_indices['Speed-Through-Water']
        V = x_collocation[:, V_idx].view(-1, 1)

        mean_P = torch.tensor(data_processor.scaler_y.mean_, dtype=torch.float32, device=self.device)
        std_P = torch.tensor(data_processor.scaler_y.scale_, dtype=torch.float32, device=self.device)
        mean_V = torch.tensor(data_processor.scaler_X.mean_[V_idx], dtype=torch.float32, device=self.device)
        std_V = torch.tensor(data_processor.scaler_X.scale_[V_idx], dtype=torch.float32, device=self.device)

        outputs_unscaled = outputs * std_P + mean_P
        V_unscaled = V * std_V + mean_V
        V_unscaled_safe = torch.clamp(V_unscaled, min=1e-2)

        outputs_V = outputs_x[:, V_idx].view(-1, 1)
        outputs_V_unscaled = (outputs_V * std_P) / std_V

        residual = outputs_V_unscaled - (3 * outputs_unscaled) / V_unscaled_safe
        scaling_factor = torch.tensor(1e5, dtype=torch.float32, device=self.device)
        residual_normalized = residual / scaling_factor

        return residual_normalized

    def compute_boundary_loss(self, feature_indices, data_processor, X_unscaled, scale=1e7):
        V_idx = feature_indices['Speed-Through-Water']
        x_min = X_unscaled.min()
        x_max = X_unscaled.max()

        num_points = self.batch_size

        mean_P = torch.tensor(data_processor.scaler_y.mean_, dtype=torch.float32, device=self.device)
        std_P = torch.tensor(data_processor.scaler_y.scale_, dtype=torch.float32, device=self.device)

        # ============================================
        # Condition 1: Power must be non-negative
        # ============================================
        V_col = X_unscaled.columns[V_idx]
        x_boundary_unscaled = pd.DataFrame(columns=X_unscaled.columns)
        
        # Sample random speeds across the full range
        x_boundary_unscaled[V_col] = np.random.uniform(
            low=x_min[V_col], 
            high=x_max[V_col], 
            size=num_points
        )

        # For other columns, sample random in [min, max]
        for col in X_unscaled.columns:
            if col != V_col:
                x_boundary_unscaled[col] = np.random.uniform(
                    low=x_min[col], 
                    high=x_max[col], 
                    size=num_points
                )

        x_boundary_scaled = data_processor.scaler_X.transform(x_boundary_unscaled)
        x_boundary = torch.tensor(x_boundary_scaled, dtype=torch.float32, device=self.device)
        outputs = self.model(x_boundary)

        # Convert outputs to unscaled power values (kW)
        outputs_unscaled = outputs * std_P + mean_P

        # Penalize negative power values
        negative_power_violation = torch.relu(-outputs_unscaled)
        boundary_loss_negative = torch.mean(negative_power_violation**2) / scale

        # ============================================
        # Condition 2: Power must not exceed NCR
        # ============================================
        NCR = 16794.0  # kW
        
        # More gradual penalty for exceeding NCR
        ncr_violation = torch.relu(outputs_unscaled - (NCR))
        boundary_loss_ncr = torch.mean(ncr_violation**2) / scale

        # Combine boundary losses
        boundary_loss = boundary_loss_negative + boundary_loss_ncr
        return boundary_loss

    def calculate_physics_loss(self, V, predicted_power_scaled, trim,
                            H_s, theta_ship, theta_wave, data_processor):

        # Constrain parameters to physical ranges using torch.clamp
        C_wave_constrained = torch.clamp(self.model.C_wave, 
                                        self.model.C_wave_min, 
                                        self.model.C_wave_max)
        eta_D_constrained = torch.clamp(self.model.eta_D, 
                                    self.model.eta_D_min, 
                                    self.model.eta_D_max)
        C_resid_constrained = torch.clamp(self.model.C_resid, 
                                        self.model.C_resid_min, 
                                        self.model.C_resid_max)


        # Add parameter constraint loss
        parameter_loss = (
            torch.relu(-self.model.C_wave + self.model.C_wave_min) + 
            torch.relu(self.model.C_wave - self.model.C_wave_max) +
            torch.relu(-self.model.eta_D + self.model.eta_D_min) + 
            torch.relu(self.model.eta_D - self.model.eta_D_max) +
            torch.relu(-self.model.C_resid + self.model.C_resid_min) + 
            torch.relu(self.model.C_resid - self.model.C_resid_max)
        )
            
        # Convert physical constants to tensors matching input type and device
        g_tensor = torch.tensor(self.g, dtype=V.dtype, device=V.device)
        L_tensor = torch.tensor(self.L, dtype=V.dtype, device=V.device)
        rho_tensor = torch.tensor(self.rho, dtype=V.dtype, device=V.device)
        nu_tensor = torch.tensor(self.nu, dtype=V.dtype, device=V.device)
        S_tensor = torch.tensor(self.S, dtype=V.dtype, device=V.device)
        B_tensor = torch.tensor(self.B, dtype=V.dtype, device=V.device)
        k_tensor = torch.tensor(self.k, dtype=V.dtype, device=V.device)
        
        # Speed-dependent calculations
        V_ref = torch.tensor(5.144, dtype=V.dtype, device=V.device)    # 10 knots reference
        V_design = torch.tensor(6.17, dtype=V.dtype, device=V.device)  # 12 knots design speed
        
        # Enhance basic resistance modeling
        Re = V * L_tensor / nu_tensor
        Re = torch.clamp(Re, min=1e-5)
        Fn = V / torch.sqrt(g_tensor * L_tensor)
        
        # Modified form factor - less aggressive at low speeds
        speed_ratio = V / V_design
        k_speed = k_tensor * (1 + 0.15 * torch.relu(speed_ratio - 0.8)**2)
        
        # Enhanced frictional resistance
        C_f = 0.075 / (torch.log10(Re) - 2) ** 2
        R_f = 0.5 * rho_tensor * V**2 * S_tensor * C_f
        
        # Modified wave resistance with reduced angle dependency
        Fn_crit = torch.tensor(0.15, dtype=V.dtype, device=V.device)
        wave_speed_factor = torch.exp(2.0 * torch.relu(Fn - Fn_crit))
        
        # Reduce the impact of wave angle
        theta_rel_wave = torch.abs(theta_wave - theta_ship) % 360
        theta_rel_wave = torch.where(theta_rel_wave > 180, 360 - theta_rel_wave, theta_rel_wave)
        theta_rel_wave_rad = theta_rel_wave * torch.pi / 180
        
        # Modified wave resistance calculation with reduced angle dependency
        angle_factor = 1.0 + 0.2 * torch.cos(theta_rel_wave_rad)  # Much smaller angle effect
        R_wave = 0.5 * rho_tensor * g_tensor * H_s * S_tensor * C_wave_constrained * wave_speed_factor * angle_factor
        
        # Speed-dependent residuary resistance
        C_resid_speed = C_resid_constrained * (1 + 0.25 * torch.relu(speed_ratio - 0.9)**2)
        R_resid = C_resid_speed * R_f
        
        # Air and appendage resistance
        R_air = 0.5 * torch.tensor(1.225, dtype=V.dtype, device=V.device) * \
                torch.tensor(0.8, dtype=V.dtype, device=V.device) * (B_tensor * 15.0) * V**2
        R_app = torch.tensor(0.04, dtype=V.dtype, device=V.device) * R_f
        
        # Enhance propulsive efficiency modeling
        eta_base = torch.clamp(eta_D_constrained, min=0.5, max=0.85)
        speed_penalty = torch.relu(speed_ratio - 0.9)**2
        eta_D_speed = eta_base * (1 - 0.15 * speed_penalty)
        eta_D_speed = torch.clamp(eta_D_speed, min=0.5, max=0.85)
        
        # Total resistance with speed-dependent scaling
        R_total = (R_f * (1 + k_speed) + R_resid + R_wave + R_air + R_app)
        
        # Add high-speed compensation factor
        high_speed_factor = 1.0 + 0.3 * torch.relu(speed_ratio - 1.0)**2
        R_total = (R_f * (1 + k_speed) + R_resid + R_wave + R_air + R_app)
        high_speed_factor = 1.0 + 0.3 * torch.relu(speed_ratio - 1.0)**2
        R_total = R_total * high_speed_factor
        
        P_total = (V * R_total) / eta_D_speed
        P_total = P_total / 1000  # Convert to kilowatts
        
        # Scale for loss calculation
        scaler_y_mean = torch.tensor(data_processor.scaler_y.mean_, dtype=V.dtype, device=V.device)
        scaler_y_scale = torch.tensor(data_processor.scaler_y.scale_, dtype=V.dtype, device=V.device)
        P_total_scaled = (P_total - scaler_y_mean) / scaler_y_scale
        
        # Combine physics loss with parameter constraint loss
        physics_loss = torch.mean((predicted_power_scaled.squeeze() - P_total_scaled) ** 2)
        total_loss = physics_loss + 1000.0 * parameter_loss  # High weight for constraint violation
        
        return total_loss, P_total_scaled

    def compute_total_loss(self, data_loss, physics_loss, pde_loss, boundary_loss):
        """
        Computes the total loss with progressive weighting for physics and PDE terms.
        The data loss and boundary loss maintain constant weights, while physics
        and PDE loss weights increase gradually during training.
        
        Args:
            data_loss: Loss based on predicted vs actual values
            physics_loss: Loss based on physical equations
            pde_loss: Loss based on PDE constraints
            boundary_loss: Loss based on boundary conditions
        
        Returns:
            torch.Tensor: Total weighted loss
        """
        # Calculate current weights based on training progress
        physics_weight = self.get_progressive_weight(
            self.current_epoch, 
            self.epochs,
            self.initial_physics_weight,
            self.final_physics_weight
        )
        
        pde_weight = self.get_progressive_weight(
            self.current_epoch,
            self.epochs,
            self.initial_pde_weight,
            self.final_pde_weight
        )
        
        # Combine all losses with their respective weights
        total_loss = (data_loss + 
                    physics_weight * physics_loss +
                    pde_weight * pde_loss +
                    self.boundary_weight * boundary_loss)
        
        # Print detailed loss information during training if in debug mode
        if self.debug_mode and self.current_epoch % 10 == 0:
            print(f"\nEpoch {self.current_epoch} loss components:")
            print(f"Data loss: {data_loss.item():.4f}")
            print(f"Physics loss (w={physics_weight:.2f}): {physics_loss.item():.4f}")
            print(f"PDE loss (w={pde_weight:.2f}): {pde_loss.item():.4f}")
            print(f"Boundary loss (w={self.boundary_weight:.2f}): {boundary_loss.item():.4f}")
            print(f"Total loss: {total_loss.item():.4f}")
        
        return total_loss

    def train(self, 
            train_loader, 
            X_train_unscaled, 
            feature_indices, 
            data_processor,
            unscaled_data_loader, 
            val_loader=None, 
            val_unscaled_loader=None, 
            X_val_unscaled=None,
            live_plot=False,
            enable_early_stopping=False,
            patience=10):

        optimizer = self.get_optimizer()
        loss_function = self.get_loss_function()

        train_losses = []
        val_losses = []

        # -- Early Stopping Tracking Variables --
        best_val_loss = float('inf')
        epochs_no_improve = 0


        if live_plot:
            plt.ion()
            fig, ax = plt.subplots()

        speed_idx = feature_indices['Speed-Through-Water']
        fore_draft_idx = feature_indices['Draft_Fore']
        aft_draft_idx = feature_indices['Draft_Aft']
        h_s_idx = feature_indices.get('Significant_Wave_Height')
        theta_ship_idx = feature_indices.get('True_Heading')
        theta_wave_idx = feature_indices.get('Mean_Wave_Direction')

        for epoch in range(self.epochs):
            self.current_epoch = epoch  # Add this line to track current epoch
            self.model.train()
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
                optimizer.zero_grad()

                outputs = self.model(X_batch)
                data_loss = loss_function(outputs, y_batch)

                V = X_unscaled_batch[:, speed_idx] * 0.51444
                trim = X_unscaled_batch[:, fore_draft_idx] - X_unscaled_batch[:, aft_draft_idx]

                if h_s_idx is not None and theta_ship_idx is not None and theta_wave_idx is not None:
                    H_s = X_unscaled_batch[:, h_s_idx]
                    theta_ship = X_unscaled_batch[:, theta_ship_idx]
                    theta_wave = X_unscaled_batch[:, theta_wave_idx]
                else:
                    H_s = torch.zeros_like(V)
                    theta_ship = torch.zeros_like(V)
                    theta_wave = torch.zeros_like(V)

                physics_loss, _ = self.calculate_physics_loss(
                    V, 
                    outputs, 
                    trim, 
                    H_s, 
                    theta_ship, 
                    theta_wave, 
                    data_processor
                )

                x_collocation = self.sample_collocation_points(
                    self.batch_size, 
                    X_train_unscaled, 
                    data_processor
                )
                pde_residual = self.compute_pde_residual(
                    x_collocation, 
                    feature_indices, 
                    data_processor
                )
                pde_loss = torch.mean(pde_residual**2)

                boundary_loss = self.compute_boundary_loss(
                    feature_indices, 
                    data_processor, 
                    X_train_unscaled
                )

                total_loss = self.compute_total_loss(
                    data_loss, 
                    physics_loss, 
                    pde_loss, 
                    boundary_loss
                )

                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item()
                running_data_loss += data_loss.item()
                running_pde_loss += pde_loss.item()
                running_boundary_loss += boundary_loss.item()
                running_physics_loss += physics_loss.item()

                progress_bar.set_postfix({
                    "Total Loss": f"{running_loss / (batch_index + 1):.8f}",
                    "Data Loss": f"{running_data_loss / (batch_index + 1):.8f}",
                    "PDE Loss": f"{running_pde_loss / (batch_index + 1):.8f}",
                    "Boundary Loss": f"{running_boundary_loss / (batch_index + 1):.8f}",
                    "Physics Loss": f"{running_physics_loss / (batch_index + 1):.8f}",
                })

            avg_total_loss = running_loss / total_batches
            avg_data_loss = running_data_loss / total_batches
            avg_pde_loss = running_pde_loss / total_batches
            avg_boundary_loss = running_boundary_loss / total_batches
            avg_physics_loss = running_physics_loss / total_batches

            train_losses.append(avg_total_loss)

            # Evaluate on validation set if provided
            if val_loader is not None and val_unscaled_loader is not None and X_val_unscaled is not None:
                val_total_loss, val_data_loss, val_pde_loss, val_boundary_loss, val_physics_loss = \
                    self.evaluate_on_loader(
                        val_loader, 
                        val_unscaled_loader, 
                        feature_indices, 
                        data_processor, 
                        X_val_unscaled
                    )
                val_losses.append(val_total_loss)

                print(f"Epoch [{epoch+1}/{self.epochs}], "
                    f"Total Loss: {avg_total_loss:.8f}, Data Loss: {avg_data_loss:.8f}, "
                    f"PDE Loss: {avg_pde_loss:.8f}, Boundary Loss: {avg_boundary_loss:.8f}, "
                    f"Physics Loss: {avg_physics_loss:.8f}, "
                    f"Validation Total Loss: {val_total_loss:.8f}, "
                    f"Validation Data Loss: {val_data_loss:.8f}, "
                    f"Validation PDE Loss: {val_pde_loss:.8f}, "
                    f"Validation Boundary Loss: {val_boundary_loss:.8f}, "
                    f"Validation Physics Loss: {val_physics_loss:.8f}, "
                    f"Learning Rate: {optimizer.param_groups[0]['lr']:.1e}")

                # -------------------------------
                # Early Stopping Check
                # -------------------------------
                if enable_early_stopping:
                    # If the current validation loss is the best so far, reset patience counter
                    if val_total_loss < best_val_loss:
                        best_val_loss = val_total_loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        print(f"Epochs no improvement: {epochs_no_improve}/{patience}")
                    
                    # If no improvement for 'patience' epochs, stop training
                    if epochs_no_improve >= patience:
                        print("Early stopping triggered due to no improvement in validation loss.")
                        break

            else:
                val_losses.append(None)
                print(f"Epoch [{epoch+1}/{self.epochs}], "
                    f"Total Loss: {avg_total_loss:.8f}, "
                    f"Data Loss: {avg_data_loss:.8f}, PDE Loss: {avg_pde_loss:.8f}, "
                    f"Boundary Loss: {avg_boundary_loss:.8f}, Physics Loss: {avg_physics_loss:.8f}, "
                    f"Learning Rate: {optimizer.param_groups[0]['lr']:.1e}")

            if live_plot:
                ax.clear()
                ax.plot(range(1, epoch+2), train_losses, label='Training Total Loss')
                if val_loader is not None:
                    # Only plot valid validation losses (non-None)
                    valid_val_losses = [v for v in val_losses if v is not None]
                    ax.plot(
                        [i for i, v in enumerate(val_losses) if v is not None],
                        valid_val_losses, 
                        label='Validation Total Loss'
                    )
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('Training and Validation Total Loss over Epochs')
                ax.legend()
                plt.pause(0.01)        

        if live_plot:
            plt.ioff()
            plt.show()
            fig.savefig('training_validation_loss_plot.png')

    def evaluate_on_loader(self, data_loader, unscaled_data_loader, feature_indices, data_processor, X_val_unscaled):
        """
        Evaluate the model on the given data_loader and unscaled_data_loader,
        but sample PDE and boundary points from the validation subset (X_val_unscaled).
        """
        self.model.eval()
        loss_function = self.get_loss_function()
        running_total_loss = 0.0
        running_data_loss = 0.0
        running_pde_loss = 0.0
        running_boundary_loss = 0.0
        running_physics_loss = 0.0
        total_batches = len(data_loader)

        speed_idx = feature_indices['Speed-Through-Water']
        fore_draft_idx = feature_indices['Draft_Fore']
        aft_draft_idx = feature_indices['Draft_Aft']
        h_s_idx = feature_indices.get('Significant_Wave_Height')
        theta_ship_idx = feature_indices.get('True_Heading')
        theta_wave_idx = feature_indices.get('Mean_Wave_Direction')

        with torch.no_grad():
            for (X_batch, y_batch), (X_unscaled_batch,) in zip(data_loader, unscaled_data_loader):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                X_unscaled_batch = X_unscaled_batch.to(self.device)

                outputs = self.model(X_batch)
                data_loss = loss_function(outputs, y_batch)

                V = X_unscaled_batch[:, speed_idx] * 0.51444
                trim = X_unscaled_batch[:, fore_draft_idx] - X_unscaled_batch[:, aft_draft_idx]

                if h_s_idx is not None and theta_ship_idx is not None and theta_wave_idx is not None:
                    H_s = X_unscaled_batch[:, h_s_idx]
                    theta_ship = X_unscaled_batch[:, theta_ship_idx]
                    theta_wave = X_unscaled_batch[:, theta_wave_idx]
                else:
                    H_s = torch.zeros_like(V)
                    theta_ship = torch.zeros_like(V)
                    theta_wave = torch.zeros_like(V)

                physics_loss, _ = self.calculate_physics_loss(V, outputs, trim, H_s, theta_ship, theta_wave, data_processor)

                # PDE computations require gradients, enable them temporarily
                with torch.enable_grad():
                    x_collocation = self.sample_collocation_points(len(X_batch), X_val_unscaled, data_processor)
                    x_collocation.requires_grad_(True)
                    pde_residual = self.compute_pde_residual(x_collocation, feature_indices, data_processor)
                    pde_loss = torch.mean(pde_residual**2)

                # Compute boundary loss with the same approach used in training
                boundary_loss = self.compute_boundary_loss(feature_indices, data_processor, X_val_unscaled)

                total_loss = self.compute_total_loss(data_loss, physics_loss, pde_loss, boundary_loss)

                running_total_loss += total_loss.item()
                running_data_loss += data_loss.item()
                running_pde_loss += pde_loss.item()
                running_boundary_loss += boundary_loss.item()
                running_physics_loss += physics_loss.item()

        avg_total_loss = running_total_loss / total_batches
        avg_data_loss = running_data_loss / total_batches
        avg_pde_loss = running_pde_loss / total_batches
        avg_boundary_loss = running_boundary_loss / total_batches
        avg_physics_loss = running_physics_loss / total_batches

        print(f"Validation Loss Breakdown: "
              f"Total: {avg_total_loss:.8f}, "
              f"Data: {avg_data_loss:.8f}, "
              f"PDE: {avg_pde_loss:.8f}, "
              f"Boundary: {avg_boundary_loss:.8f}, "
              f"Physics: {avg_physics_loss:.8f}")

        return avg_total_loss, avg_data_loss, avg_pde_loss, avg_boundary_loss, avg_physics_loss

    def evaluate(self, X_eval, y_eval, dataset_type="Validation", data_processor=None):
        self.model.eval()
        X_eval_tensor = torch.tensor(X_eval.values, dtype=torch.float32).to(self.device)
        y_eval_tensor = torch.tensor(y_eval.values, dtype=torch.float32).view(-1, 1).to(self.device)

        loss_function = self.get_loss_function()
        with torch.no_grad():
            outputs = self.model(X_eval_tensor)
            loss = loss_function(outputs, y_eval_tensor)
            print(f"\n{dataset_type} Loss: {loss.item():.8f}")

            if data_processor:
                outputs_original = data_processor.inverse_transform_y(outputs.cpu().numpy())
                y_eval_original = data_processor.inverse_transform_y(y_eval_tensor.cpu().numpy())

                rmse = np.sqrt(np.mean((outputs_original - y_eval_original) ** 2))
                print(f"{dataset_type} RMSE: {rmse:.4f}")

        return loss.item()

    def cross_validate(self, X, X_unscaled, y, feature_indices, data_processor, k_folds=5):
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
                       unscaled_data_loader, val_loader=val_loader, val_unscaled_loader=val_unscaled_loader,
                       X_val_unscaled=X_val_unscaled_fold, live_plot=False)
            val_loss = self.evaluate(X_val_fold, y_val_fold, dataset_type="Validation", data_processor=data_processor)
            fold_results.append(val_loss)

        avg_val_loss = np.mean(fold_results)
        print(f"\nCross-validation results: Average Validation Loss = {avg_val_loss:.8f}")
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

        print(f"\nBest parameters: {best_params}, with average validation loss: {best_loss:.8f}")
        with open("best_hyperparameters.txt", "w") as f:
            f.write(f"Best parameters: {best_params}\n")
            f.write(f"Best average validation loss: {best_loss:.8f}\n")

        return best_params, best_loss


if __name__ == "__main__":
    data_processor = DataProcessor(
        file_path='data/Aframax/P data_20200213-20200726_Democritos.csv',
        target_column='Power',
        keep_columns_file='columns_to_keep.txt'
    )
    result = data_processor.load_and_prepare_data()
    if result is not None:
        X_train, X_test, X_train_unscaled, X_test_unscaled, y_train, y_test, \
            y_train_unscaled, y_test_unscaled = result

        data_processor.X_train_unscaled = X_train_unscaled

        print(f"X_train shape: {X_train.shape}")
        print(f"X_train_unscaled shape: {X_train_unscaled.shape}")
        print(f"y_train shape: {y_train.shape}")

        assert list(X_train.columns) == list(X_train_unscaled.columns), \
            "Column mismatch between scaled and unscaled data"

        feature_indices = {col: idx for idx, col in enumerate(X_train_unscaled.columns)}

        required_columns = ['Speed-Through-Water', 'Draft_Fore', 'Draft_Aft']
        for col in required_columns:
            if col not in feature_indices:
                raise ValueError(f"Required column '{col}' not found in data")

        param_grid = {
            'lr': [0.0001],
            'batch_size': [1024]
        }

        epochs_cv = 1
        epochs_final = 900
        optimizer = 'Adam'
        loss_function = 'MSE'

        best_params, best_loss = ShipSpeedPredictorModel.hyperparameter_search(
            X_train, X_train_unscaled, y_train, feature_indices,
            param_grid, epochs_cv, optimizer, loss_function, data_processor, k_folds=5
        )

        # Split final training/validation
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
            debug_mode=False
        )

        final_train_loader = final_model.prepare_dataloader(X_train_final, y_train_final)
        final_unscaled_loader = final_model.prepare_unscaled_dataloader(X_train_unscaled_final)
        final_val_loader = final_model.prepare_dataloader(X_val_final, y_val_final)
        final_val_unscaled_loader = final_model.prepare_unscaled_dataloader(X_val_unscaled_final)

        final_model.train(
            final_train_loader, X_train_unscaled_final, feature_indices, data_processor,
            unscaled_data_loader=final_unscaled_loader,
            val_loader=final_val_loader,
            val_unscaled_loader=final_val_unscaled_loader,
            X_val_unscaled=X_val_unscaled_final,
            live_plot=True,
            enable_early_stopping=False,
            patience=40  
        )

        with open('best_hyperparameters.json', 'w') as f:
            json.dump(best_params, f)

        # Evaluate on test set
        final_model.evaluate(X_test, y_test, dataset_type="Test", data_processor=data_processor)

        # ------------------------------------------------------
        # Prepare for final CSV output with extra columns
        # ------------------------------------------------------
        # 1) Ensure consistent column order
        X_test_unscaled = X_test_unscaled[X_train_unscaled.columns]

        # 2) Convert X_test_unscaled for inference
        X_test_scaled = data_processor.scaler_X.transform(X_test_unscaled)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(final_model.device)

        final_model.model.eval()
        with torch.no_grad():
            y_pred_scaled = final_model.model(X_test_tensor)
            y_pred_scaled_np = y_pred_scaled.cpu().numpy()
            # Convert scaled predictions back to original power units
            y_pred = data_processor.inverse_transform_y(y_pred_scaled_np).flatten()

        y_actual = y_test_unscaled.values.flatten()

        # ------------------------------------------------------
        # Gather unscaled test columns
        # ------------------------------------------------------
        speed_unscaled = X_test_unscaled['Speed-Through-Water'].values

        # Draft columns (already in unscaled domain)
        draft_fore_unscaled = X_test_unscaled['Draft_Fore'].values
        draft_aft_unscaled = X_test_unscaled['Draft_Aft'].values

        # Significant Wave Height (if present in your columns)
        if 'Significant_Wave_Height' in X_test_unscaled.columns:
            hs_unscaled = X_test_unscaled['Significant_Wave_Height'].values
        else:
            # If it's not in the columns, fill with zeros or some default
            hs_unscaled = np.zeros(len(X_test_unscaled))

        # Compute relative wave angle from True_Heading & Mean_Wave_Direction
        # If they exist in the DataFrame
        if 'True_Heading' in X_test_unscaled.columns and 'Mean_Wave_Direction' in X_test_unscaled.columns:
            theta_ship_unscaled = X_test_unscaled['True_Heading'].values
            theta_wave_unscaled = X_test_unscaled['Mean_Wave_Direction'].values

            theta_rel_angle = np.abs(theta_wave_unscaled - theta_ship_unscaled) % 360
            theta_rel_angle = np.where(theta_rel_angle > 180, 360 - theta_rel_angle, theta_rel_angle)
        else:
            # Fallback if columns are missing
            theta_rel_angle = np.zeros(len(X_test_unscaled))

        # ------------------------------------------------------
        # Build final DataFrame with extra columns
        # ------------------------------------------------------
        results_df = pd.DataFrame({
            'Speed-Through-Water': speed_unscaled,
            'Draft_Fore': draft_fore_unscaled,
            'Draft_Aft': draft_aft_unscaled,
            'Significant_Wave_Height': hs_unscaled,
            'Relative_Wave_Angle': theta_rel_angle,
            'Actual Power': y_actual,
            'Predicted Power': y_pred
        })

        # Save to CSV
        output_csv_file = 'power_predictions.csv'
        results_df.to_csv(output_csv_file, index=False)
        print(f"Predictions saved to {output_csv_file}")

        rmse = np.sqrt(np.mean((y_actual - y_pred) ** 2))
        mae = np.mean(np.abs(y_actual - y_pred))
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test MAE: {mae:.4f}")
    else:
        print("Error in loading and preparing data.")