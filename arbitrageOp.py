import pandas as pd
import numpy as np

historical_prices = {'stock_A': [10, 11, 12, 13, 14], 'stock_B': [20, 19, 18, 17, 16]}
tender_offers = {'stock_A': 15, 'stock_B': 18}

def calculate_tender_offer(stock, offer_price, historical_prices, threshold=0.05):
    current_price = historical_prices[stock][-1]
    if offer_price > current_price * (1 + threshold):
        return True
    return False

def execute_trade(stock, is_profitable):
    if is_profitable:
        return f"Executed trade on {stock}"
    return f"No trade executed for {stock}"

for stock, offer_price in tender_offers.items():
    is_profitable = calculate_tender_offer(stock, offer_price, historical_prices)
    trade_status = execute_trade(stock, is_profitable)
    print(trade_status)