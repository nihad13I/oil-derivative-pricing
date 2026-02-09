import numpy as np
import matplotlib.pyplot as plt

def realistic_jump_diffusion(S0, theta, sigma, lam, mu_j, sigma_j, kappa, T, dt, n_paths):
    """
    S0: Initial Price (e.g., 85)
    theta: Long-term mean from calibrator (e.g., 82.5)
    sigma: Clean volatility (e.g., 0.18)
    lam: Jump intensity (e.g., 4.2)
    mu_j/sigma_j: Jump size parameters
    kappa: REVERSION SPEED (Higher = more stable prices)
    """
    n_steps = int(T / dt)
    # Work in Log-Space to prevent price explosions
    x = np.zeros((n_steps + 1, n_paths))
    x[0] = np.log(S0)

    # Pre-calculate the log-mean level
    log_theta = np.log(theta)

    for t in range(1, n_steps + 1):
        # 1. Diffusion: Standard Brownian Motion
        dw = np.random.standard_normal(n_paths) * np.sqrt(dt)

        # 2. Mean Reversion: Pulls the LOG-price back to log_theta
        # This prevents the price from drifting away permanently
        drift = kappa * (log_theta - x[t-1]) * dt

        # 3. Jumps: Poisson arrivals with Normal magnitude
        # We model jumps directly in the log-return
        jump_count = np.random.poisson(lam * dt, n_paths)
        jump_magnitude = np.random.normal(mu_j, sigma_j, n_paths)
        total_jump = jump_count * jump_magnitude

        # Update Log-Price
        x[t] = x[t-1] + drift + (sigma * dw) + total_jump

    # Convert back from Log-Space to Dollar-Space
    return np.exp(x)

# --- REASONABLE SETTINGS ---
# Use a higher Kappa (e.g., 1.5 to 3.0) to keep oil near the mean
S_start = 85.0
params = {
    'theta': 82.50,    # From your calibrator
    'sigma': 0.18,     # Clean Sigma
    'lam': 4.20,       # Annual Jumps
    'mu_j': 0.01,      # Keep mean jump small (1%)
    'sigma_j': 0.05,   # Jump volatility (5%)
    'kappa': 2.5,      # STRENGTH of the pull back to $82.50
    'T': 1.0,
    'dt': 1/252,
    'n_paths': 50
}

paths = realistic_jump_diffusion(S_start, **params)

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(paths, alpha=0.3, color='blue')
plt.axhline(y=params['theta'], color='red', linestyle='--', label=f'Equilibrium (${params["theta"]})')
plt.title("Stabilized Brent Crude Simulation: Log-Normal Jump Diffusion")
plt.ylabel("Price (USD)")
plt.xlabel("Trading Days")
plt.legend()
plt.show()
