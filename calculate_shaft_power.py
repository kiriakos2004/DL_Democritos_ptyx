import numpy as np

def calculate_shaft_power(V_knots, trim, rho, S, S_APP, A_t,
                          C_a, k, STWAVE1, alpha_trim, eta_D, L, nu, g, L_t):
    """
    Calculate the shaft power (P_S) based on vessel speed, trim, and physical constants.
    
    Parameters:
    - V_knots (float or ndarray): Vessel speed in knots.
    - trim (float or ndarray): Trim value (fore draft - aft draft) in meters.
    - Other physical constants as specified.
    
    Returns:
    - P_S (float or ndarray): Calculated shaft power (in kilowatts).
    """
    # Convert speed from knots to meters per second
    V = V_knots * 0.51444  # 1 knot = 0.51444 m/s

    # Ensure V is not zero to avoid division by zero errors
    V = np.clip(V, 1e-5, None)

    # Calculate Reynolds number Re
    Re = V * L / nu

    # Avoid log of zero or negative numbers
    Re = np.clip(Re, 1e-5, None)

    # Calculate frictional resistance coefficient C_f using ITTC-1957 formula
    C_f = 0.075 / (np.log10(Re) - 2) ** 2

    # Frictional Resistance (R_F)
    R_F = 0.5 * rho * V**2 * S * C_f
    
    # Wave-Making Resistance (R_W)
    STWAVE2 = 1 + alpha_trim * trim  # Dynamic correction factor involving trim
    C_W = STWAVE1 * STWAVE2
    R_W = 0.5 * rho * V**2 * S * C_W
    
    # Appendage Resistance (R_APP)
    R_APP = 0.5 * rho * V**2 * S_APP * C_f
    
    # Calculate F_nt (Transom Froude Number)
    F_nt = V / np.sqrt(g * L_t)
    
    # Transom Stern Resistance (R_TR)
    R_TR = 0.5 * rho * V**2 * A_t * (1 - F_nt)
    
    # Correlation Allowance Resistance (R_C)
    R_C = 0.5 * rho * V**2 * S * C_a
    
    # Total Resistance (R_T)
    R_T = R_F * (1 + k) + R_W + R_APP + R_TR + R_C
    
    # Calculate shaft power (P_S)
    P_S = ((V * R_T) / eta_D) / 1000  # Convert to kilowatts if necessary

    return P_S

if __name__ == "__main__":
    # Example inputs
    V_knots = np.array([6.78, 11.0])       # Speeds in knots
    fore_draft = np.array([6.3, 14.5])     # Fore drafts in meters
    aft_draft = np.array([7.6, 15.1])      # Aft drafts in meters

    # Calculate trim
    trim = fore_draft - aft_draft

    # Constants for physics-based calculations
    rho = 1025.0      # Water density (kg/m³)
    S = 9950.0        # Wetted surface area in m²
    S_APP = 150.0     # Wetted surface area of appendages in m²
    A_t = 50.0        # Transom area in m²
    C_a = 0.00045     # Correlation allowance coefficient
    k = 0.15          # Form factor (dimensionless)
    STWAVE1 = 0.001   # Base wave resistance coefficient
    alpha_trim = 0.1  # Effect of trim on wave resistance
    eta_D = 0.7      # Propulsive efficiency
    L = 230.0         # Ship length in meters
    nu = 1e-6         # Kinematic viscosity of water (m²/s)
    g = 9.81          # Gravitational acceleration (m/s²)
    L_t = 20.0        # Transom length in meters

    # Calculate shaft power with custom parameters
    P_S = calculate_shaft_power(
        V_knots, trim, rho, S, S_APP, A_t,
        C_a, k, STWAVE1, alpha_trim, eta_D, L, nu, g, L_t
    )

    for speed, fore_d, aft_d, power in zip(V_knots, fore_draft, aft_draft, P_S):
        print(f"At speed {speed} knots, fore draft {fore_d} m, aft draft {aft_d} m, the shaft power is {power:.2f} kilowatts.")
