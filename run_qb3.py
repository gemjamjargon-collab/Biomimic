import numpy as np
import pandas as pd


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
    x = np.asarray(series, dtype=float)

    # Core signals
    D = normalize(np.abs(x - pd.Series(x).rolling(10, min_periods=1).mean()))
    N = normalize(pd.Series(x).rolling(10, min_periods=1).std().fillna(0))
    C = normalize(curvature(x))

    # Grounding/coherence proxy: stable when low volatility/curvature
    G = 1 - normalize(N + C)

    # Instability fusion
    Phi = w_D * D + w_N * N + w_C * C + w_G * (1 - G)

    # Persistence accumulator
    A = np.zeros_like(Phi)
    for t in range(1, len(Phi)):
        A[t] = max(0, A[t - 1] + Phi[t] - lambda_A)

    # Bounded stability score
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


if __name__ == "__main__":
    # Demo signal: stable → drift → acceleration → failure
    np.random.seed(42)

    stable = np.random.normal(1.0, 0.03, 80)
    drift = np.linspace(1.0, 1.4, 60) + np.random.normal(0, 0.04, 60)
    accel = 1.4 + np.exp(np.linspace(0, 2.2, 60)) / 10 + np.random.normal(0, 0.06, 60)

    signal = np.concatenate([stable, drift, accel])

    result = qb3_score(signal)

    print(result.tail(25))
    print("\nFirst WARNING index:")
    print(result[result["state"].isin(["WARNING", "CRITICAL"])].head(1))

    result.to_csv("qb3_runner_output.csv", index=False)
    print("\nSaved: qb3_runner_output.csv")
