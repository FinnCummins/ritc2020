import numpy as np
from scipy.stats import norm
from volatility_forecast import garch_volatility_forecast, simple_moving_average_volatility

def black_scholes_call(S, K, T, r, sigma):
    '''
    Calculate the Black-Scholes price of a European call option.
    
    Parameters:
    - S (float): Current stock (or ETF) price.
    - K (float): Strike price of the option.
    - T (float): Time to expiration in years.
    - r (float): Risk-free interest rate.
    - sigma (float): Volatility of the underlying asset.
    
    Returns:
    - call_price (float): Price of the call option.
    '''
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    '''
    Calculate the Black-Scholes price of a European put option.
    
    Parameters:
    - S (float): Current stock (or ETF) price.
    - K (float): Strike price of the option.
    - T (float): Time to expiration in years.
    - r (float): Risk-free interest rate.
    - sigma (float): Volatility of the underlying asset.
    
    Returns:
    - put_price (float): Price of the put option.
    '''
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price


# Sample historical ETF prices (hypothetical data)
historical_prices = np.array([100, 101, 102, 103, 102, 101, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])

# Calculate log returns
returns = np.log(historical_prices[1:] / historical_prices[:-1])

# Forecast volatility using GARCH or SMA model
sigma_garch = garch_volatility_forecast(returns)
sigma_sma = simple_moving_average_volatility(returns)

# Black-Scholes variables (example values)
S = historical_prices[-1]  # Current price of the ETF
K = 105  # Strike price of the option
T = 1/12  # Time to expiration in years (1 month)
r = 0.00  # Annual risk-free interest rate (0%)
sigma = sigma_garch  # Using GARCH forecasted volatility

# Calculate call option price
call_price = black_scholes_call(S, K, T, r, sigma)
print("Call Option Price:", call_price)

