import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class PhysicsValidator:
    def __init__(self):
        # Constants for physics-based calculations (from original model)
        self.rho = 1025.0      # Water density (kg/m³)
        self.nu = 1e-6         # Kinematic viscosity of water (m²/s)
        self.g = 9.81          # Gravitational acceleration (m/s²)
        self.S = 9950.0        # Wetted surface area in m²
        self.L = 264.0         # Ship length in meters
        self.B = 50.0          # Beam (width) of the ship in meters
        self.k = 1.3           # Empirical correction
        
        # Initialize physics parameters (typical values from naval architecture)
        self.C_wave = 0.0007   # Wave resistance coefficient
        self.eta_D = 0.80      # Propulsion system efficiency
        self.C_resid = 0.05    # Residuary resistance coefficient

    def calculate_power(self, V, trim, H_s, theta_ship, theta_wave):
        """
        Calculate theoretical power based on physics equations.
        All inputs should be numpy arrays or tensors of the same length.
        
        Args:
            V: Speed through water in m/s
            trim: Difference between fore and aft draft
            H_s: Significant wave height
            theta_ship: Ship heading
            theta_wave: Wave direction
        """
        # Convert inputs to PyTorch tensors if they aren't already
        if not isinstance(V, torch.Tensor):
            V = torch.tensor(V, dtype=torch.float32)
            trim = torch.tensor(trim, dtype=torch.float32)
            H_s = torch.tensor(H_s, dtype=torch.float32)
            theta_ship = torch.tensor(theta_ship, dtype=torch.float32)
            theta_wave = torch.tensor(theta_wave, dtype=torch.float32)

        # Convert constants to tensors
        g_tensor = torch.tensor(self.g, dtype=torch.float32)
        L_tensor = torch.tensor(self.L, dtype=torch.float32)
        rho_tensor = torch.tensor(self.rho, dtype=torch.float32)
        nu_tensor = torch.tensor(self.nu, dtype=torch.float32)
        S_tensor = torch.tensor(self.S, dtype=torch.float32)
        B_tensor = torch.tensor(self.B, dtype=torch.float32)
        k_tensor = torch.tensor(self.k, dtype=torch.float32)

        # Reference speeds
        V_ref = torch.tensor(5.144, dtype=torch.float32)    # 10 knots reference
        V_design = torch.tensor(6.17, dtype=torch.float32)  # 12 knots design speed
        
        # Basic calculations
        Re = V * L_tensor / nu_tensor
        Re = torch.clamp(Re, min=1e-5)  # Prevent division by zero
        Fn = V / torch.sqrt(g_tensor * L_tensor)
        speed_ratio = V / V_design
        
        # Form factor calculation
        k_speed = k_tensor * (1 + 0.15 * torch.relu(speed_ratio - 0.8)**2)
        
        # Frictional resistance
        C_f = 0.075 / (torch.log10(Re) - 2) ** 2
        R_f = 0.5 * rho_tensor * V**2 * S_tensor * C_f
        
        # Wave resistance with angle dependency
        Fn_crit = torch.tensor(0.15, dtype=torch.float32)
        wave_speed_factor = torch.exp(2.0 * torch.relu(Fn - Fn_crit))
        
        # Calculate relative wave angle
        theta_rel_wave = torch.abs(theta_wave - theta_ship) % 360
        theta_rel_wave = torch.where(theta_rel_wave > 180, 360 - theta_rel_wave, theta_rel_wave)
        theta_rel_wave_rad = theta_rel_wave * torch.pi / 180
        
        # Wave resistance
        angle_factor = 1.0 + 0.2 * torch.cos(theta_rel_wave_rad)
        R_wave = 0.5 * rho_tensor * g_tensor * H_s * S_tensor * self.C_wave * wave_speed_factor * angle_factor
        
        # Residuary resistance
        C_resid_speed = self.C_resid * (1 + 0.25 * torch.relu(speed_ratio - 0.9)**2)
        R_resid = C_resid_speed * R_f
        
        # Air and appendage resistance
        R_air = 0.5 * torch.tensor(1.225, dtype=torch.float32) * \
                torch.tensor(0.8, dtype=torch.float32) * (B_tensor * 15.0) * V**2
        R_app = torch.tensor(0.04, dtype=torch.float32) * R_f
        
        # Calculate total resistance with speed-dependent factors
        R_total = (R_f * (1 + k_speed) + R_resid + R_wave + R_air + R_app)
        
        # Add high-speed compensation
        high_speed_factor = 1.0 + 0.3 * torch.relu(speed_ratio - 1.0)**2
        R_total = R_total * high_speed_factor
        
        # Calculate propulsion efficiency
        eta_base = torch.clamp(torch.tensor(self.eta_D, dtype=torch.float32), min=0.5, max=0.85)
        speed_penalty = torch.relu(speed_ratio - 0.9)**2
        eta_D_speed = eta_base * (1 - 0.15 * speed_penalty)
        eta_D_speed = torch.clamp(eta_D_speed, min=0.6, max=0.85)
        
        # Calculate total power
        P_total = (V * R_total) / eta_D_speed  # Power in Watts
        P_total = P_total / 1000  # Convert to kilowatts
        
        return P_total.numpy()

def main():
    # Load the CSV file and select specific rows
    print("Loading data...")
    # Read the full CSV first
    df_full = pd.read_csv('data/Aframax/P data_20200213-20200726_Democritos_test.csv')
    
    # Select rows from 10010 to 11000 (inclusive)
    # We subtract 1 from the start index because Python uses 0-based indexing
    start_row = 10010 - 1  # Adjust for 0-based indexing
    end_row = 11000
    df = df_full.iloc[start_row:end_row].copy()
    
    # Reset the index to make it sequential starting from 0
    df.reset_index(drop=True, inplace=True)
    
    print(f"Selected {len(df)} rows from the dataset (rows {start_row + 1} to {end_row})")
    
    # Initialize the physics validator
    validator = PhysicsValidator()
    
    # Convert speed from knots to m/s
    speed_ms = df['Speed-Through-Water'] * 0.51444
    
    # Calculate trim
    trim = df['Draft_Fore'] - df['Draft_Aft']
    
    # Get wave data (if available, otherwise use zeros)
    H_s = df['Significant_Wave_Height'] if 'Significant_Wave_Height' in df.columns else np.zeros(len(df))
    theta_ship = df['True_Heading'] if 'True_Heading' in df.columns else np.zeros(len(df))
    theta_wave = df['Mean_Wave_Direction'] if 'Mean_Wave_Direction' in df.columns else np.zeros(len(df))
    
    print("Calculating theoretical power...")
    # Calculate theoretical power
    theoretical_power = validator.calculate_power(
        speed_ms,
        trim,
        H_s,
        theta_ship,
        theta_wave
    )
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Speed_Knots': df['Speed-Through-Water'],
        'Draft_Fore': df['Draft_Fore'],
        'Draft_Aft': df['Draft_Aft'],
        'Actual_Power': df['Power'],
        'Theoretical_Power': theoretical_power,
        'Power_Difference': df['Power'] - theoretical_power,
        'Power_Difference_Percentage': ((df['Power'] - theoretical_power) / df['Power']) * 100
    })
    
    # Save results
    results_df.to_csv('physics_validation_results.csv', index=False)
    print("Results saved to physics_validation_results.csv")
    
    # Calculate and print summary statistics
    rmse = np.sqrt(np.mean((df['Power'] - theoretical_power) ** 2))
    mae = np.mean(np.abs(df['Power'] - theoretical_power))
    mape = np.mean(np.abs((df['Power'] - theoretical_power) / df['Power'])) * 100
    
    print("\nValidation Metrics:")
    print(f"RMSE: {rmse:.2f} kW")
    print(f"MAE: {mae:.2f} kW")
    print(f"MAPE: {mape:.2f}%")
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    plt.scatter(df['Speed-Through-Water'], df['Power'], alpha=0.5, label='Actual Power')
    plt.scatter(df['Speed-Through-Water'], theoretical_power, alpha=0.5, label='Theoretical Power')
    plt.xlabel('Speed Through Water (knots)')
    plt.ylabel('Power (kW)')
    plt.title('Actual vs Theoretical Power Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('power_comparison_plot.png')
    print("Comparison plot saved as power_comparison_plot.png")

if __name__ == "__main__":
    main()