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


historical_prices = np.array([100, 101, 102, 103, 102, 101, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])

returns = np.log(historical_prices[1:] / historical_prices[:-1])

# You can use either the GARCH or SMA model for this example
sigma_garch = garch_volatility_forecast(returns)
sigma_sma = simple_moving_average_volatility(returns)

S = historical_prices[-1]
K = 105 
T = 1/12
r = 0.00
sigma = sigma_garch

call_price = black_scholes_call(S, K, T, r, sigma)
print("Call Option Price:", call_price)

