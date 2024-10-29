# test.py

import numpy as np

def calculate_shaft_power(V_knots, fore_draft, aft_draft, rho, S, S_APP, A_t,
                          F_nt, C_f, C_a, k,STWAVE1, alpha_trim, eta_D):
    """
    Calculate the shaft power (P_S) based on vessel speed, fore draft, aft draft, and physical constants.

    Parameters:
    - V (float or ndarray): Speed of the vessel (in m/s).
    - fore_draft (float or ndarray): Fore draft of the vessel (in meters).
    - aft_draft (float or ndarray): Aft draft of the vessel (in meters).
    - rho (float): Water density (kg/m³). Default is 1025.0.
    - S (float): Wetted surface area of the hull (m²). Default is 9950.0.
    - S_APP (float): Wetted surface area of appendages (m²). Default is 150.0.
    - A_t (float): Transom area (m²). Default is 50.0.
    - F_nt (float): Transom Froude number (dimensionless). Default is 0.3.
    - C_f (float): Frictional resistance coefficient (dimensionless). Default is 0.0025.
    - C_a (float): Correlation allowance coefficient (dimensionless). Default is 0.00045.
    - k (float): Form factor (dimensionless). Default is 0.15.
    - STWAVE1 (float): Base wave resistance coefficient (dimensionless). Default is 0.001.
    - alpha_trim (float): Coefficient representing the effect of trim on wave resistance. Default is 0.1.
    - eta_D (float): Propulsive efficiency (dimensionless). Default is 0.65.

    Returns:
    - P_S (float or ndarray): Calculated shaft power (in KWatts).
    """
    # Convert speed from knots to meters per second
    V = V_knots * 0.51444  # 1 knot = 0.51444 m/s

    # Calculate trim as the difference between fore draft and aft draft
    trim = fore_draft - aft_draft

    # Frictional Resistance (R_F)
    R_F = 0.5 * rho * V**2 * S * C_f

    # Wave-Making Resistance (R_W)
    STWAVE2 = 1 + alpha_trim * trim  # Dynamic correction factor involving trim
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
    P_S = ((V * R_T) / eta_D)/1000

    return P_S

if __name__ == "__main__":
    import numpy as np

    # Example inputs
    V = np.array([6.78, 11.0]) # Speeds in knots
    fore_draft = np.array([6.3, 14.5]) # Fore drafts in meters
    aft_draft = np.array([7.6, 15.1])  # Aft drafts in meters

    # Ship-specific constants
    rho = 1025.0      # kg/m³
    S = 9995.0        # m²
    S_APP = 200.0     # m²
    A_t = 60.0        # m²
    F_nt = 0.28       # Dimensionless
    C_f = 0.0023      # Dimensionless
    C_a = 0.0005      # Dimensionless
    k = 0.2           # Dimensionless
    STWAVE1 = 0.0008  # Dimensionless
    alpha_trim = 0.12 # Dimensionless
    eta_D = 0.84       # Dimensionless

    # Calculate shaft power with custom parameters
    P_S = calculate_shaft_power(
        V, fore_draft, aft_draft, rho, S, S_APP, A_t, F_nt, C_f, C_a, k,
        STWAVE1, alpha_trim, eta_D
    )

    for speed, fore_d, aft_d, power in zip(V, fore_draft, aft_draft, P_S):
        print(f"At speed {speed} knots, fore draft {fore_d} m, aft draft {aft_d} m, the shaft power is {power:.2f} KWatts")
