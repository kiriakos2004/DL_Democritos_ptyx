import pandas as pd
import numpy as np

def compute_physics_power(
    speed_knots,
    draft_fore,
    draft_aft,
    H_s,
    theta_ship,
    theta_wave,
    C_wave=0.01,
    eta_D=0.85,
    C_resid=0.03
):
    """
    Computes the physics-based power (kW) using the same formula 
    from 'calculate_physics_loss' in your NN code.

    speed_knots: Speed Through Water [knots]
    draft_fore, draft_aft: Drafts [meters]
    H_s: Significant Wave Height [m]
    theta_ship, theta_wave: headings [degrees]
    C_wave, eta_D, C_resid: physics coefficients (fixed for this script)
    """

    # 1) Convert knots -> m/s
    V = speed_knots * 0.51444  # 1 knot ≈ 0.51444 m/s


    # 2) Some constants from your code
    rho = 1025.0       # kg/m^3
    nu = 1e-6          # m^2/s (kinematic viscosity)
    g = 9.81           # m/s^2
    S = 9950.0         # m^2 (wetted surface)
    L = 229.0          # m (length)
    B = 32.0           # m (beam)
    k = 1.1            # empirical correction

    # 3) Trim (not used directly in formula, but you had a clamp on it)
    trim = draft_fore - draft_aft
    # If you'd like, clamp the trim:
    trim = np.clip(trim, -5.0, 5.0)

    # 4) Reynolds Number
    V_safe = np.maximum(V, 1e-5)   # avoid zero/negative speeds
    Re = (V_safe * L) / nu
    Re_safe = np.maximum(Re, 1e-5)

    # 5) Frictional Resistance Coefficient
    #    C_f = 0.075 / (log10(Re) - 2)^2
    #    Use np.log10 for base-10 logs
    C_f = 0.075 / (np.log10(Re_safe) - 2) ** 2

    # 6) Frictional Resistance
    R_f = (1 + k) * 0.5 * rho * (V_safe ** 2) * S * C_f

    # 7) Residuary Resistance
    R_resid_value = C_resid * R_f

    # 8) Wave-Induced Resistance
    #    wave amplitude A = H_s / 2
    A = np.maximum(H_s, 0.0) / 2.0
    S_wave = L * B

    # 9) Relative Wave Angle
    #    angle between wave direction and ship heading
    #    theta_rel_wave = abs(theta_wave - theta_ship) % 360
    #    if > 180 => 360 - angle
    theta_rel_wave = np.abs(theta_wave - theta_ship) % 360
    theta_rel_wave = np.where(theta_rel_wave > 180, 360 - theta_rel_wave, theta_rel_wave)
    # convert degrees -> radians
    theta_rel_wave_rad = np.deg2rad(theta_rel_wave)

    R_wave = 0.5 * rho * g * A * S_wave * C_wave * np.cos(theta_rel_wave_rad) ** 2

    # 10) Total Resistance
    R_total = R_f + R_resid_value + R_wave

    # 11) Required Propulsion Power [Watts]
    #     P_total_watts = (V * R_total) / eta_D
    P_total_watts = (V_safe * R_total) / eta_D

    # convert to kW
    P_total_kW = P_total_watts / 1000.0
    return P_total_kW


def main():
    # --- Modify these paths as needed ---
    input_csv = "data/Aframax/P data_20200213-20200726_Democritos.csv"
    output_csv = "physics_equations_comparison.csv"

    # Read the CSV
    df = pd.read_csv(input_csv)

    # For demonstration, let's assume columns are exactly:
    # 'Power', 'Speed-Through-Water', 'Draft_Fore', 'Draft_Aft',
    # 'Significant_Wave_Height', 'True_Heading', 'Mean_Wave_Direction'
    required_cols = [
        "Power",
        "Speed-Through-Water",
        "Draft_Fore",
        "Draft_Aft",
        "Significant_Wave_Height",
        "True_Heading",
        "Mean_Wave_Direction",
    ]

    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in CSV. Setting default zeros.")
            df[col] = 0.0

    # Drop any rows that have NaN in these columns (optional)
    df = df.dropna(subset=required_cols)

    # Extract arrays
    power_actual = df["Power"].values  # from CSV

    speed_knots = df["Speed-Through-Water"].values
    draft_fore = df["Draft_Fore"].values
    draft_aft = df["Draft_Aft"].values
    H_s = df["Significant_Wave_Height"].values
    theta_ship = df["True_Heading"].values
    theta_wave = df["Mean_Wave_Direction"].values

    # We'll pick constants for wave/resid/eta
    C_wave = 0.01   # typical guess from the code
    eta_D = 0.85    # typical guess
    C_resid = 0.03  # typical guess

    # Compute physics-based predictions
    physics_predicted = compute_physics_power(
        speed_knots,
        draft_fore,
        draft_aft,
        H_s,
        theta_ship,
        theta_wave,
        C_wave=C_wave,
        eta_D=eta_D,
        C_resid=C_resid
    )

    # Build new DataFrame with 2 columns
    result_df = pd.DataFrame({
        "Actual_Power": power_actual, 
        "Physics_Predicted_Power": physics_predicted
    })

    # Save to CSV
    result_df.to_csv(output_csv, index=False)
    print(f"Comparison CSV saved to {output_csv}")
    print("Sample rows:")
    print(result_df.head(10))

if __name__ == "__main__":
    main()