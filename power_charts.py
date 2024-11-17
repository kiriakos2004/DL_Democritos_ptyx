from bayes_opt import BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt
from read_data import DataProcessor

# Initialize DataProcessor with appropriate file path and target column
file_path = 'data/Aframax/P data_20200213-20200726_Democritos.csv'  # Update with the actual path
target_column = 'Power'
keep_columns_file = 'columns_to_keep.txt'  # Path to the text file with columns to keep

data_processor = DataProcessor(file_path, target_column, keep_columns_file)

# Load and prepare data
result = data_processor.load_and_prepare_data()
if result is not None:
    # Unpack all returned values
    (X_train, X_test, X_train_unscaled, X_test_unscaled,
     y_train, y_test, y_train_unscaled, y_test_unscaled) = result

    # Use the unscaled data for calculation
    V_knots = X_train_unscaled['Speed-Through-Water'].values
    fore_draft = X_train_unscaled['Draft_Fore'].values
    aft_draft = X_train_unscaled['Draft_Aft'].values
    trim = fore_draft - aft_draft
    H_s = X_train_unscaled['Significant_Wave_Height'].values
    theta_ship = X_train_unscaled['True_Heading'].values
    theta_wave = X_train_unscaled['Mean_Wave_Direction'].values

    # Use a subset of the data to reduce memory usage
    subset_size = 120000  # Adjust as needed to control memory usage
    V_knots = V_knots[:subset_size]
    trim = trim[:subset_size]
    H_s = H_s[:subset_size]
    theta_ship = theta_ship[:subset_size]
    theta_wave = theta_wave[:subset_size]
    y_train_unscaled_subset = y_train_unscaled[:subset_size].values

    # Constants for shaft power calculation
    rho = 1025.0      # Water density (kg/m³)
    S = 9950.0        # Wetted surface area in m²
    S_APP = 150.0     # Wetted surface area of appendages in m²
    A_t = 50.0        # Transom area in m²
    L = 230.0         # Ship length in meters
    nu = 1e-6         # Kinematic viscosity of water (m²/s)
    g = 9.81          # Gravitational acceleration (m/s²)

    # Define the function to calculate shaft power with tunable parameters
    def calculate_physics_based_shaft_power(V_knots, trim, H_s, theta_ship, theta_wave, k_wave, STWAVE1, alpha_trim, C_a, k, L_t, eta_D):
        V = V_knots * 0.51444  # Convert knots to m/s
        V = np.clip(V, 1e-5, None)  # Avoid division by zero
        H_s = np.clip(H_s, 0.0, None)

        # Reynolds number and frictional resistance coefficient
        Re = V * L / nu
        Re = np.clip(Re, 1e-5, None)
        C_f = 0.075 / (np.log10(Re) - 2) ** 2

        # Frictional resistance
        R_F = 0.5 * rho * V**2 * S * C_f

        # Wave-making resistance
        STWAVE2 = 1 + alpha_trim * trim
        C_W = STWAVE1 * STWAVE2
        R_W = 0.5 * rho * V**2 * S * C_W

        # Appendage resistance
        R_APP = 0.5 * rho * V**2 * S_APP * C_f

        # Transom stern resistance
        F_nt = V / np.sqrt(g * L_t)
        R_TR = 0.5 * rho * V**2 * A_t * (1 - F_nt)

        # Correlation allowance resistance
        R_C = 0.5 * rho * V**2 * S * C_a

        # Added wave resistance
        theta_rel_wave = np.abs(theta_wave - theta_ship) % 360
        theta_rel_wave = np.where(theta_rel_wave > 180, 360 - theta_rel_wave, theta_rel_wave)
        theta_rel_wave_rad = np.deg2rad(theta_rel_wave)
        R_AW = 0.5 * rho * V**2 * S * k_wave * H_s**2 * (1 + np.cos(theta_rel_wave_rad))

        # Total resistance
        R_T = R_F * (1 + k) + R_W + R_APP + R_TR + R_C + R_AW

        # Shaft power in kilowatts
        P_S = ((V * R_T) / eta_D) / 1000
        return P_S

    # Define the objective function for Bayesian Optimization
    def objective(k_wave, STWAVE1, alpha_trim, C_a, k, L_t, eta_D):
        # Randomly sample 18,000 points from the subset data to reduce memory usage
        sample_indices = np.random.choice(len(V_knots), 18000, replace=False)
        V_knots_sample = V_knots[sample_indices]
        trim_sample = trim[sample_indices]
        H_s_sample = H_s[sample_indices]
        theta_ship_sample = theta_ship[sample_indices]
        theta_wave_sample = theta_wave[sample_indices]
        y_sample = y_train_unscaled_subset[sample_indices]

        # Calculate shaft power with the current parameters
        P_S = calculate_physics_based_shaft_power(
            V_knots_sample, trim_sample, H_s_sample, theta_ship_sample, theta_wave_sample, k_wave, STWAVE1, alpha_trim, C_a, k, L_t, eta_D
        )

        # Calculate the mean absolute error between calculated and observed power
        error = np.mean(np.abs(P_S - y_sample))

        # Return the negative error (since BayesianOptimization maximizes by default)
        return -error

    # Set up the Bayesian Optimization with a range for additional parameters
    optimizer = BayesianOptimization(
        f=objective,
        pbounds={
            'k_wave': (1e-8, 1e-9),        # Narrowed range for k_wave
            'STWAVE1': (0.0001, 0.005),    # Range for STWAVE1
            'alpha_trim': (0.05, 0.2),     # Range for alpha_trim
            'C_a': (0.0001, 0.001),        # Range for correlation allowance coefficient C_a
            'k': (0.1, 0.2),               # Range for form factor k
            'L_t': (15.0, 25.0),           # Range for transom length L_t
            'eta_D': (0.90, 0.95)           # Range for propulsive efficiency eta_D
        },
        verbose=2,
        random_state=42
    )

    # Run the optimization
    optimizer.maximize(init_points=5, n_iter=50)  # Adjust iterations if needed

    # Retrieve the best parameter values
    best_params = optimizer.max['params']
    best_k_wave = best_params['k_wave']
    best_STWAVE1 = best_params['STWAVE1']
    best_alpha_trim = best_params['alpha_trim']
    best_C_a = best_params['C_a']
    best_k = best_params['k']
    best_L_t = best_params['L_t']
    best_eta_D = best_params['eta_D']
    print(f"Optimal parameters: k_wave={best_k_wave}, STWAVE1={best_STWAVE1}, alpha_trim={best_alpha_trim}, C_a={best_C_a}, k={best_k}, L_t={best_L_t}, eta_D={best_eta_D}")

    # Calculate shaft power with the optimal parameters
    P_S_optimized = calculate_physics_based_shaft_power(
        V_knots, trim, H_s, theta_ship, theta_wave, best_k_wave, best_STWAVE1, best_alpha_trim, best_C_a, best_k, best_L_t, best_eta_D
    )

    # Plot both the original power and the optimized calculated shaft power
    plt.figure(figsize=(12, 6))
    plt.plot(y_train_unscaled_subset, label='Original Power (CSV)', color='blue')
    plt.plot(P_S_optimized, label='Optimized Calculated Shaft Power', color='green')
    plt.title('Original Power vs. Optimized Calculated Shaft Power')
    plt.xlabel('Index')
    plt.ylabel('Power (kW)')
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("Failed to load and prepare data.")