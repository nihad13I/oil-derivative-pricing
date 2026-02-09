import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Download Brent Crude Data
oil_data = yf.download('BZ=F', start='2021-01-01', end='2026-02-01')

# --- ROBUST COLUMN SELECTION ---
# Handle the Multi-Index structure often returned by yfinance
if isinstance(oil_data.columns, pd.MultiIndex):
    # Check if 'Close' is in the levels (usually level 0)
    if 'Close' in oil_data.columns.get_level_values(0):
        prices = oil_data['Close']
    else:
        prices = oil_data.iloc[:, 0] # Fallback to first column
else:
    # Standard Index handling
    if 'Close' in oil_data.columns:
        prices = oil_data['Close']
    else:
        prices = oil_data.iloc[:, 0]

prices = prices.dropna()

# --- DATA PREPARATION ---
# Use .values to avoid index-related plotting errors (Glyph 9)
price_vals = prices.values.flatten().astype(float)
log_returns = np.diff(np.log(price_vals))

# --- PARAMETER CALCULATIONS ---
theta = np.mean(price_vals)
std_dev = np.std(log_returns)
threshold = 3 * std_dev
jumps = log_returns[np.abs(log_returns) > threshold]
non_jumps = log_returns[np.abs(log_returns) <= threshold]

# Calculate Jump Parameters
num_years = (len(prices) / 252)
lam = len(jumps) / num_years
mu_j = np.mean(jumps) if len(jumps) > 0 else 0
sigma_clean = np.std(non_jumps) * np.sqrt(252)

# --- 2-PART VISUALIZATION ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
plt.subplots_adjust(hspace=0.3)

# Subplot 1: Price Levels & Mean Reversion
ax1.plot(price_vals, color='black', label='Brent Crude Price', alpha=0.7)
ax1.axhline(y=theta, color='red', linestyle='--', linewidth=2, label=f'Mean Reversion (Theta): ${theta:.2f}')
ax1.set_title('Phase 1: Mean Reversion Equilibrium Analysis', fontweight='bold')
ax1.set_ylabel('Price (USD)')
ax1.legend(loc='upper left')

# Subplot 2: Returns & Jump Detection
ax2.plot(log_returns, color='gray', label='Daily Returns', alpha=0.4)
ax2.scatter(np.where(np.abs(log_returns) > threshold)[0],
            jumps, color='orange', label='Significant Jumps', zorder=5)
ax2.axhline(y=threshold, color='blue', linestyle=':', label='Jump Threshold (3-Sigma)')
ax2.axhline(y=-threshold, color='blue', linestyle=':')
ax2.set_title('Phase 2: Statistical Jump Identification', fontweight='bold')
ax2.set_ylabel('Log Returns')
ax2.legend(loc='upper left')

plt.show()

print(f"--- Final Calibrated Values ---")
print(f"Theta: {theta:.2f}")
print(f"Sigma (Clean): {sigma_clean*100:.2f}%")
print(f"Lambda: {lam:.2f} jumps/yr")
print(f"Mu_j (Mean Jump): {mu_j*100:.2f}%")
