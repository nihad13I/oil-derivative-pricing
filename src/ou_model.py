import numpy as np
import matplotlib.pyplot as plt

# --- Parameters from your Real-World Calibration ---
S0 = 81.20      # Current spot price (check your data_fetch plot for the last point)
theta = 79.75   # Your calculated Long-term Mean
sigma = 0.3393  # Your calculated Annualized Volatility
kappa = 2.5     # Reversion Speed (Adjust this to see how fast it returns to $79.75)
T = 1.0         # 1 Year simulation
dt = 0.001
steps = int(T/dt)

# --- Ornstein-Uhlenbeck (OU) Simulation ---
prices_ou = np.zeros(steps)
prices_ou[0] = S0

for t in range(1, steps):
    # Standard Brownian Motion increment
    dW = np.random.normal(0, np.sqrt(dt))

    # Ito's Lemma Applied: dX = kappa(theta - X)dt + sigma*dW
    # Note: In commodity models, we often use sigma * X * dW for percentage vol
    drift = kappa * (theta - prices_ou[t-1]) * dt
    diffusion = sigma * prices_ou[t-1] * dW

    prices_ou[t] = prices_ou[t-1] + drift + diffusion

# --- Plotting Results ---
plt.figure(figsize=(10, 5))
plt.plot(prices_ou, color='darkblue', label='Stochastic Path (Simulated Oil)')
plt.axhline(y=theta, color='red', linestyle='--', label=f'Mean Reversion Level (${theta})')
plt.title(f'Mean Reverting Price Simulation (Theta=${theta}, Sigma={sigma*100:.1f}%)')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
