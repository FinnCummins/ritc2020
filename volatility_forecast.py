import numpy as np
import pandas as pd
from arch import arch_model

def garch_volatility_forecast(returns, rescale_factor=100):
    '''
    Forecast volatility using the GARCH(1,1) model with rescaling.
    
    Parameters:
    - returns (pd.Series): A pandas Series of historical log returns of the asset.
    - rescale_factor (float): The factor by which to rescale returns for optimal model fitting.
    
    Returns:
    - forecasted_volatility (float): The forecasted next period volatility, rescaled back to original scale.
    '''
    # Rescale returns
    rescaled_returns = returns * rescale_factor

    # Define and fit a GARCH(1,1) model on rescaled data
    garch_model = arch_model(rescaled_returns, vol='Garch', p=1, q=1)
    res = garch_model.fit(disp='off')
    
    # Forecast the next period volatility and rescale back
    forecast = res.forecast(horizon=1)
    forecasted_volatility = np.sqrt(forecast.variance.iloc[-1] / (rescale_factor**2)).iloc[0]
    return forecasted_volatility


def simple_moving_average_volatility(returns, window=5):
    '''
    Calculate volatility using a Simple Moving Average (SMA).
    
    Parameters:
    - returns (pd.Series or np.ndarray): A pandas Series or numpy array of historical log returns of the asset.
    - window (int): The number of periods to use for the moving average.
    
    Returns:
    - sma_volatility (float): The calculated SMA volatility.
    '''
    # Ensure returns is a pandas Series
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)

    # Calculate the moving average of the squared returns
    squared_returns = returns**2
    sma_volatility = np.sqrt(squared_returns.rolling(window=window).mean())
    return sma_volatility.iloc[-1]
