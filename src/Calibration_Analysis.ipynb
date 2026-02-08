import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# 1. Brent Crude Data
# We use 'Close' as it is more reliable for futures data
oil_data = yf.download('BZ=F', start='2021-01-01', end='2026-02-01')

# Fix for multi-index columns and potential KeyError
if 'Adj Close' in oil_data.columns:
    prices = oil_data['Adj Close']
else:
    prices = oil_data['Close']

# Drop any missing values (NaNs) which can break calculations
prices = prices.dropna()

# 2. Calculate Parameters for the OU Process
theta = np.mean(prices.values)  # Long-term mean price

# Calculate daily log returns to get sigma
log_returns = np.diff(np.log(prices.values.flatten()))
daily_std = np.std(log_returns)
sigma = daily_std * np.sqrt(252) # Annualized volatility

print(f"--- Calibration Results ---")
print(f"Long-term Mean (Theta): ${theta:.2f}")
print(f"Annualized Volatility (Sigma): {sigma*100:.2f}%")

# 3. Visualization
plt.figure(figsize=(12, 6))
plt.plot(prices.index, prices.values, label='Brent Crude Spot (Market)')
plt.axhline(y=theta, color='red', linestyle='--', label=f'Mean Reversion Level (${theta:.2f})')
plt.title('Brent Crude Oil: Historical Equilibrium Analysis')
plt.xlabel('Year')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
