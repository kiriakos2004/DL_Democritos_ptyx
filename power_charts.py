import matplotlib.pyplot as plt
import numpy as np
from read_data import DataProcessor

# Initialize DataProcessor with appropriate file path and target column
file_path = 'data/Aframax/P data_20200213-20200726_Democritos.csv'  # Update with the actual path
target_column = 'Power'
drop_columns = ['TIME']  # Specify any columns you want to drop

data_processor = DataProcessor(file_path, target_column, drop_columns)

# Load and prepare data
result = data_processor.load_and_prepare_data()
if result is not None:
    X_train, X_test, X_train_unscaled, X_test_unscaled, y_train, y_test = result

    # Use the unscaled data for calculation
    V_knots = X_train_unscaled['Speed-Through-Water'].values
    fore_draft = X_train_unscaled['Draft_Fore'].values
    aft_draft = X_train_unscaled['Draft_Aft'].values
    trim = fore_draft - aft_draft

    # Constants for shaft power calculation
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


    def calculate_shaft_power(V_knots, trim, rho, S, S_APP, A_t,
                              C_a, k, STWAVE1, alpha_trim, eta_D, L, nu, g, L_t):
        V = V_knots * 0.51444  # Convert knots to m/s
        V = np.clip(V, 1e-5, None)
        Re = V * L / nu
        Re = np.clip(Re, 1e-5, None)
        C_f = 0.075 / (np.log10(Re) - 2) ** 2
        R_F = 0.5 * rho * V**2 * S * C_f
        STWAVE2 = 1 + alpha_trim * trim
        C_W = STWAVE1 * STWAVE2
        R_W = 0.5 * rho * V**2 * S * C_W
        R_APP = 0.5 * rho * V**2 * S_APP * C_f
        F_nt = V / np.sqrt(g * L_t)
        R_TR = 0.5 * rho * V**2 * A_t * (1 - F_nt)
        R_C = 0.5 * rho * V**2 * S * C_a
        R_T = R_F * (1 + k) + R_W + R_APP + R_TR + R_C
        P_S = ((V * R_T) / eta_D) / 1000  # Convert to kW
        return P_S

    # Calculate shaft power
    P_S = calculate_shaft_power(V_knots, trim, rho, S, S_APP, A_t,
                                C_a, k, STWAVE1, alpha_trim, eta_D, L, nu, g, L_t)

    # Plot both the original power and the calculated shaft power
    plt.figure(figsize=(12, 6))
    plt.plot(y_train.values, label='Original Power (CSV)', color='blue')
    plt.plot(P_S, label='Calculated Shaft Power', color='green')
    plt.title('Original Power vs. Calculated Shaft Power')
    plt.xlabel('Index')
    plt.ylabel('Power (kW)')
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("Failed to load and prepare data.")
