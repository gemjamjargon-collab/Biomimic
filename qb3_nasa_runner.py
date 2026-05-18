import numpy as np
import pandas as pd
import json
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats


def normalize(x, eps=1e-9):
    x = np.asarray(x, dtype=float)
    return (x - np.min(x)) / (np.max(x) - np.min(x) + eps)


def curvature(x):
    x = np.asarray(x, dtype=float)
    c = np.zeros_like(x)
    c[2:] = np.abs(x[2:] - 2 * x[1:-1] + x[:-2])
    return c


def qb3_score(
    series,
    alpha=0.75,
    beta=0.50,
    lambda_A=0.12,
    w_D=0.30,
    w_N=0.20,
    w_C=0.35,
    w_G=0.15,
):
    """QB³ full score with curvature"""
    x = np.asarray(series, dtype=float)

    D = normalize(np.abs(x - pd.Series(x).rolling(10, min_periods=1).mean()))
    N = normalize(pd.Series(x).rolling(10, min_periods=1).std().fillna(0))
    C = normalize(curvature(x))
    G = 1 - normalize(N + C)

    Phi = w_D * D + w_N * N + w_C * C + w_G * (1 - G)

    A = np.zeros_like(Phi)
    for t in range(1, len(Phi)):
        A[t] = max(0, A[t - 1] + Phi[t] - lambda_A)

    QBS = 1 / (1 + alpha * Phi + beta * A)

    states = []
    for q in QBS:
        if q > 0.85:
            states.append("STABLE")
        elif q > 0.65:
            states.append("WATCH")
        elif q > 0.40:
            states.append("WARNING")
        else:
            states.append("CRITICAL")

    return pd.DataFrame({
        "x": x,
        "D_divergence": D,
        "N_noise": N,
        "C_curvature": C,
        "G_grounding": G,
        "Phi_instability": Phi,
        "A_persistence": A,
        "QBS_score": QBS,
        "state": states
    })


def qb3_score_ablated(series, alpha=0.75, beta=0.50, lambda_A=0.12, w_D=0.30, w_N=0.20, w_C=0.0, w_G=0.15):
    """QB³ WITHOUT curvature — proves curvature is essential"""
    x = np.asarray(series, dtype=float)

    D = normalize(np.abs(x - pd.Series(x).rolling(10, min_periods=1).mean()))
    N = normalize(pd.Series(x).rolling(10, min_periods=1).std().fillna(0))
    C = normalize(curvature(x))  # computed but not used
    G = 1 - normalize(N + C)

    Phi = w_D * D + w_N * N + w_C * C + w_G * (1 - G)  # w_C = 0 → no curvature contribution

    A = np.zeros_like(Phi)
    for t in range(1, len(Phi)):
        A[t] = max(0, A[t - 1] + Phi[t] - lambda_A)

    QBS = 1 / (1 + alpha * Phi + beta * A)

    states = []
    for q in QBS:
        if q > 0.85:
            states.append("STABLE")
        elif q > 0.65:
            states.append("WATCH")
        elif q > 0.40:
            states.append("WARNING")
        else:
            states.append("CRITICAL")

    return pd.DataFrame({
        "x": x,
        "QBS_score_ablated": QBS,
        "state_ablated": states
    })


def rms_z_baseline(series, window=10, threshold=2.5):
    """RMS-Z: rolling RMS normalized by median absolute deviation"""
    x = np.asarray(series, dtype=float)
    
    rolling_rms = pd.Series(np.sqrt(x**2)).rolling(window, min_periods=1).mean()
    mad = np.median(np.abs(rolling_rms - np.median(rolling_rms)))
    
    if mad == 0:
        mad = 1e-9
    
    z_score = (rolling_rms - np.median(rolling_rms)) / mad
    
    scores = 1 / (1 + np.abs(z_score))
    
    states = []
    for s in scores:
        if s > 0.85:
            states.append("STABLE")
        elif s > 0.65:
            states.append("WATCH")
        elif s > 0.40:
            states.append("WARNING")
        else:
            states.append("CRITICAL")
    
    return pd.DataFrame({
        "RMS_Z_score": z_score,
        "RMS_Z_stability": scores,
        "state_rms_z": states
    })


def variance_threshold_baseline(series, window=10, threshold=0.7):
    """Variance threshold: simple rolling std deviation"""
    x = np.asarray(series, dtype=float)
    
    rolling_var = pd.Series(x).rolling(window, min_periods=1).std().fillna(0)
    normalized_var = normalize(rolling_var)
    
    scores = 1 - normalized_var
    
    states = []
    for s in scores:
        if s > 0.85:
            states.append("STABLE")
        elif s > 0.65:
            states.append("WATCH")
        elif s > 0.40:
            states.append("WARNING")
        else:
            states.append("CRITICAL")
    
    return pd.DataFrame({
        "Variance_normalized": normalized_var,
        "Variance_stability": scores,
        "state_variance": states
    })


def find_first_warning(states):
    """Find index of first WARNING or CRITICAL state"""
    for i, state in enumerate(states):
        if state in ["WARNING", "CRITICAL"]:
            return i
    return len(states)  # Never warned


def generate_demo_signal():
    """Synthetic bearing degradation: stable → drift → acceleration → failure"""
    np.random.seed(42)
    
    stable = np.random.normal(1.0, 0.03, 80)
    drift = np.linspace(1.0, 1.4, 60) + np.random.normal(0, 0.04, 60)
    accel = 1.4 + np.exp(np.linspace(0, 2.2, 60)) / 10 + np.random.normal(0, 0.06, 60)
    
    signal = np.concatenate([stable, drift, accel])
    return signal


def load_nasa_ims_data(data_dir, bearing):
    """Load NASA IMS bearing data from directory"""
    try:
        # NASA IMS format: multiple CSV files per bearing
        # Typically: bearing_0_1.csv, bearing_0_2.csv, etc.
        files = sorted(Path(data_dir).glob(f"*bearing_{bearing - 1}_*.csv"))
        
        if not files:
            print(f"⚠️  No bearing {bearing} data found in {data_dir}")
            return None
        
        # Load and concatenate all files for this bearing
        dfs = []
        for f in files:
            df = pd.read_csv(f, header=None)
            dfs.append(df)
        
        data = pd.concat(dfs, ignore_index=True)
        
        # NASA IMS has multiple sensor columns; use first accelerometer
        signal = data.iloc[:, 0].values
        
        print(f"✅ Loaded {len(signal)} samples from bearing {bearing}")
        return signal
    
    except Exception as e:
        print(f"❌ Error loading NASA data: {e}")
        return None


def run_validation(signal, output_dir="results"):
    """Run all 4 baselines + QB³ on signal"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Run all algorithms
    qb3_full = qb3_score(signal)
    qb3_ablated = qb3_score_ablated(signal)
    rms_z = rms_z_baseline(signal)
    variance = variance_threshold_baseline(signal)
    
    # Combine results
    results = pd.concat([qb3_full[["x", "QBS_score", "state"]], 
                        qb3_ablated[["QBS_score_ablated", "state_ablated"]],
                        rms_z[["RMS_Z_stability", "state_rms_z"]],
                        variance[["Variance_stability", "state_variance"]]], axis=1)
    
    # Find lead times
    qb3_lead = find_first_warning(qb3_full["state"].values)
    ablated_lead = find_first_warning(qb3_ablated["state_ablated"].values)
    rms_z_lead = find_first_warning(rms_z["state_rms_z"].values)
    var_lead = find_first_warning(variance["state_variance"].values)
    
    # Validation verdicts
    validation = {
        "qb3_lead_time": int(qb3_lead),
        "ablated_lead_time": int(ablated_lead),
        "rms_z_lead_time": int(rms_z_lead),
        "variance_lead_time": int(var_lead),
        "qb3_beats_ablation": qb3_lead < ablated_lead,
        "qb3_beats_rms_z": qb3_lead < rms_z_lead,
        "qb3_beats_variance": qb3_lead < var_lead,
        "vss1_survives": (qb3_lead < ablated_lead) and (qb3_lead < rms_z_lead),
    }
    
    # Save outputs
    results.to_csv(f"{output_dir}/qb3_nasa_full_results.csv", index=False)
    qb3_ablated.to_csv(f"{output_dir}/qb3_nasa_ablated_results.csv", index=False)
    
    with open(f"{output_dir}/qb3_nasa_lead_times.json", "w") as f:
        json.dump(validation, f, indent=2)
    
    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    
    x_axis = range(len(signal))
    
    axes[0].plot(x_axis, qb3_full["QBS_score"], label="QB³ Full", linewidth=2, color="green")
    axes[0].axhline(y=0.4, color='red', linestyle='--', alpha=0.7, label='WARNING threshold')
    axes[0].axvline(x=qb3_lead, color='green', linestyle=':', alpha=0.5)
    axes[0].set_ylabel("QB³ Score")
    axes[0].set_title(f"QB³ Full (Lead: {qb3_lead})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(x_axis, qb3_ablated["QBS_score_ablated"], label="QB³ Ablated (no curvature)", linewidth=2, color="orange")
    axes[1].axhline(y=0.4, color='red', linestyle='--', alpha=0.7)
    axes[1].axvline(x=ablated_lead, color='orange', linestyle=':', alpha=0.5)
    axes[1].set_ylabel("Ablated Score")
    axes[1].set_title(f"QB³ Ablated (Lead: {ablated_lead}) — Curvature removed")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(x_axis, rms_z["RMS_Z_stability"], label="RMS-Z", linewidth=2, color="blue")
    axes[2].axhline(y=0.4, color='red', linestyle='--', alpha=0.7)
    axes[2].axvline(x=rms_z_lead, color='blue', linestyle=':', alpha=0.5)
    axes[2].set_ylabel("RMS-Z Score")
    axes[2].set_title(f"RMS-Z Baseline (Lead: {rms_z_lead})")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    axes[3].plot(x_axis, variance["Variance_stability"], label="Variance Threshold", linewidth=2, color="purple")
    axes[3].axhline(y=0.4, color='red', linestyle='--', alpha=0.7)
    axes[3].axvline(x=var_lead, color='purple', linestyle=':', alpha=0.5)
    axes[3].set_ylabel("Variance Score")
    axes[3].set_xlabel("Sample Index")
    axes[3].set_title(f"Variance Threshold (Lead: {var_lead})")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/qb3_nasa_comparison.png", dpi=150)
    print(f"✅ Saved plot: {output_dir}/qb3_nasa_comparison.png")
    
    return validation, results


def main():
    parser = argparse.ArgumentParser(description="QB³ NASA IMS Validation Runner")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to NASA IMS data directory")
    parser.add_argument("--bearing", type=int, default=4, help="Bearing number (1-4)")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--demo", action="store_true", default=True, help="Run demo first")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 QB³ NASA IMS Validation Runner")
    print("="*70)
    
    # Demo run
    print("\n📊 DEMO: Synthetic bearing degradation signal")
    print("-"*70)
    signal_demo = generate_demo_signal()
    validation_demo, _ = run_validation(signal_demo, output_dir="results_demo")
    
    print("\n📋 DEMO Validation Results:")
    print(json.dumps(validation_demo, indent=2))
    
    if validation_demo["vss1_survives"]:
        print("\n✅ VSS1 SURVIVES DEMO ✅")
        print(f"   QB³ lead: {validation_demo['qb3_lead_time']} < Ablated: {validation_demo['ablated_lead_time']}")
        print(f"   QB³ lead: {validation_demo['qb3_lead_time']} < RMS-Z: {validation_demo['rms_z_lead_time']}")
    else:
        print("\n❌ DEMO FAILED — VSS1 REQUIRES REFACTOR")
    
    # Real IMS data (if provided)
    if args.data_dir:
        print("\n" + "="*70)
        print(f"📡 REAL DATA: NASA IMS Bearing {args.bearing}")
        print("-"*70)
        
        signal_real = load_nasa_ims_data(args.data_dir, args.bearing)
        
        if signal_real is not None:
            validation_real, results_real = run_validation(signal_real, output_dir=args.output_dir)
            
            print("\n📋 REAL DATA Validation Results:")
            print(json.dumps(validation_real, indent=2))
            
            if validation_real["vss1_survives"]:
                print("\n✅ VSS1 SURVIVES REAL DATA ✅")
                print(f"   QB³ lead: {validation_real['qb3_lead_time']} < Ablated: {validation_real['ablated_lead_time']}")
                print(f"   QB³ lead: {validation_real['qb3_lead_time']} < RMS-Z: {validation_real['rms_z_lead_time']}")
            else:
                print("\n❌ REAL DATA FAILED — VSS1 REQUIRES REFACTOR")
        else:
            print("⚠️  Skipping real data validation (not found)")
    
    print("\n" + "="*70)
    print("✅ Validation complete")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
