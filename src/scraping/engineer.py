import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

def calculate_technical_indicators(df):
    """Calculate technical indicators from Bitcoin price data."""
    # Create a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Calculate multiple period SMAs
    for period in [5, 14, 21, 50]:
        data[f'btc_sma_{period}'] = SMAIndicator(close=data['btc_close'], window=period).sma_indicator()
        data[f'btc_ema_{period}'] = EMAIndicator(close=data['btc_close'], window=period).ema_indicator()
    
    # Calculate MA differences
    data['btc_sma_14_50_diff'] = data['btc_sma_14'] - data['btc_sma_50']
    data['btc_ema_14_50_diff'] = data['btc_ema_14'] - data['btc_ema_50']
    
    # Calculate MA ratios
    data['btc_sma_14_50_ratio'] = data['btc_sma_14'] / data['btc_sma_50']
    
    # Calculate MA slopes (5-period change)
    for period in [14, 21, 50]:
        data[f'btc_sma_{period}_slope'] = data[f'btc_sma_{period}'] - data[f'btc_sma_{period}'].shift(5)
        data[f'btc_ema_{period}_slope'] = data[f'btc_ema_{period}'] - data[f'btc_ema_{period}'].shift(5)
    
    # Calculate price distance from MAs
    data['btc_close_ema_21_dist'] = data['btc_close'] - data['btc_ema_21']
    data['btc_close_ema_21_dist_norm'] = (data['btc_close'] - data['btc_ema_21']) / data['btc_close']
    
    # Calculate RSI
    data['btc_rsi_14'] = RSIIndicator(close=data['btc_close'], window=14).rsi()
    
    # Calculate MACD
    macd = MACD(close=data['btc_close'])
    data['btc_macd'] = macd.macd()
    data['btc_macd_signal'] = macd.macd_signal()
    data['btc_macd_diff'] = macd.macd_diff()
    
    # Calculate Bollinger Bands
    bb = BollingerBands(close=data['btc_close'], window=20, window_dev=2)
    data['btc_bb_high'] = bb.bollinger_hband()
    data['btc_bb_low'] = bb.bollinger_lband()
    data['btc_bb_mid'] = bb.bollinger_mavg()
    data['btc_bb_width'] = bb.bollinger_wband()
    
    return data

def main():
    df = pd.read_csv('data/raw/dataset.csv')
    df = calculate_technical_indicators(df)
    df.to_csv('data/raw/final_dataset_with_technical_indicators.csv', index=False)

if __name__ == "__main__":
    main()