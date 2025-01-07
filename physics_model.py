import pandas as pd
import numpy as np

def compute_physics_power(
    speed_knots,
    draft_fore,
    draft_aft,
    H_s,
    theta_ship,
    theta_wave,
    C_wave=0.001,
    eta_D=0.85,
    C_resid=0.05
):
    """
    Computes the physics-based power (kW) and returns:
      - R_f (Frictional Resistance)
      - R_resid_value (Residuary Resistance)
      - R_wave (Wave-Induced Resistance)
      - P_total_kW (Total Required Propulsion Power in kW)
    """

    # 1) Convert knots -> m/s
    V = speed_knots * 0.51444  # 1 knot ≈ 0.51444 m/s

    # 2) Some constants
    rho = 1025.0       # kg/m^3
    nu = 1e-6          # m^2/s (kinematic viscosity)
    g = 9.81           # m/s^2
    S = 9950.0         # m^2 (wetted surface)
    L = 229.0          # m (length)
    B = 32.0           # m (beam)
    k = 0.9            # empirical correction

    # 3) Trim
    trim = draft_fore - draft_aft
    trim = np.clip(trim, -5.0, 5.0)

    # 4) Reynolds Number
    V_safe = np.maximum(V, 1e-5)  # avoid zero speeds
    Re = (V_safe * L) / nu
    Re_safe = np.maximum(Re, 1e-5)

    # 5) Frictional Resistance Coefficient
    C_f = 0.075 / (np.log10(Re_safe) - 2) ** 2

    # 6) Frictional Resistance (R_f)
    R_f = (1 + k) * 0.5 * rho * (V_safe ** 2) * S * C_f

    # 7) Residuary Resistance (R_resid_value)
    R_resid_value = C_resid * R_f

    # 8) Wave-Induced Resistance (R_wave)
    A = np.maximum(H_s, 0.0) / 2.0
    S_wave = L * B
    # Relative wave direction
    theta_rel_wave = np.abs(theta_wave - theta_ship) % 360
    theta_rel_wave = np.where(theta_rel_wave > 180, 360 - theta_rel_wave, theta_rel_wave)
    theta_rel_wave_rad = np.deg2rad(theta_rel_wave)
    R_wave = 0.5 * rho * g * A * S_wave * C_wave * np.cos(theta_rel_wave_rad) ** 2

    # 9) Total Resistance (R_total)
    R_total = R_f + R_resid_value + R_wave

    # 10) Required Propulsion Power [Watts]
    P_total_watts = (V_safe * R_total) / eta_D

    # Convert to kW
    P_total_kW = P_total_watts / 1000.0

    return R_f, R_resid_value, R_wave, P_total_kW

def main():
    input_csv = "data/Dan/P data_20210428-20211111_Democritos.csv"
    output_csv = "physics_equations_comparison.csv"

    df = pd.read_csv(input_csv)

    required_cols = [
        "Power",
        "Speed-Through-Water",
        "Draft_Fore",
        "Draft_Aft",
        "Significant_Wave_Height",
        "True_Heading",
        "Mean_Wave_Direction",
    ]

    # Ensure all required columns exist
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in CSV. Setting default zeros.")
            df[col] = 0.0

    # Drop rows where these columns have NaN
    df = df.dropna(subset=required_cols)

    power_actual = df["Power"].values
    speed_knots = df["Speed-Through-Water"].values
    draft_fore = df["Draft_Fore"].values
    draft_aft = df["Draft_Aft"].values
    H_s = df["Significant_Wave_Height"].values
    theta_ship = df["True_Heading"].values
    theta_wave = df["Mean_Wave_Direction"].values

    # Change these values as needed to tune power predictions:
    C_wave = 0.002
    eta_D = 0.85
    C_resid = 0.07

    # Compute physics-based resistances & power
    R_f, R_resid_value, R_wave, physics_predicted = compute_physics_power(
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

    # Create a DataFrame that includes both power columns,
    # the resistances, and any relevant input columns:
    result_df = pd.DataFrame({
        "Actual_Power": power_actual,
        "Physics_Predicted_Power": physics_predicted,
        "Frictional_Resistance": R_f,
        "Residuary_Resistance": R_resid_value,
        "Wave_Resistance": R_wave,
        "Speed-Through-Water": speed_knots,
        "Draft_Fore": draft_fore,
        "Draft_Aft": draft_aft,
        "Significant_Wave_Height": H_s
    })

    # Save to CSV
    result_df.to_csv(output_csv, index=False)
    print(f"Output saved to {output_csv}")

if __name__ == "__main__":
    main()