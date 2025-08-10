import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from statsmodels.tsa.seasonal import seasonal_decompose

class StockAnalysis:
    def __init__(self, data):
        """
        Initialize the class with the dataset.
        """
        self.data = data.copy()
        self.data['Daily Returns'] = self.data['Close'].pct_change()
    
    def plot_closing_price(self):
        """Visualizes the closing price over time."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data['Close'], label='Closing Price', color='blue')
        plt.xlabel('Time')
        plt.ylabel('Normalized Close Price')
        plt.title('Tesla Closing Price Over Time')
        plt.legend()
        plt.show()
    
    def calculate_daily_returns(self):
        """Plots daily percentage changes to observe volatility."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data['Daily Returns'], label='Daily Returns', color='red')
        plt.xlabel('Time')
        plt.ylabel('Percentage Change')
        plt.title('Daily Returns Over Time')
        plt.legend()
        plt.show()

    def detect_outliers(self):
        """Identifies outliers using the Z-score method."""
        
        # Reset index to avoid duplicate label issues
        self.data = self.data.reset_index()

        # Handle NaN and infinite values
        self.data['Daily Returns'] = self.data['Daily Returns'].replace([np.inf, -np.inf], np.nan)
        self.data = self.data.dropna(subset=['Daily Returns'])

        # Compute Z-score only on valid values
        valid_returns = self.data['Daily Returns']
        
        if valid_returns.empty:
            print("No valid data for outlier detection.")
            return pd.DataFrame()  # Return an empty DataFrame if no valid data exists

        z_scores = pd.Series(zscore(valid_returns), index=valid_returns.index)

        # Assign Z-score while keeping original DataFrame shape
        self.data['Z-Score'] = z_scores.reindex(self.data.index)

        # Identify outliers where |Z-Score| > 3
        outliers = self.data[self.data['Z-Score'].abs() > 3]

        print(f"Identified {len(outliers)} outlier days with extreme returns.")
        
        return outliers
    
    def decompose_time_series(self, period=30):
        """Decomposes time series into trend, seasonality, and residuals."""
        decomposition = seasonal_decompose(self.data['Close'], model='additive', period=period)
        
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(decomposition.trend, label='Trend', color='blue')
        plt.legend()
        
        plt.subplot(3, 1, 2)
        plt.plot(decomposition.seasonal, label='Seasonality', color='green')
        plt.legend()
        
        plt.subplot(3, 1, 3)
        plt.plot(decomposition.resid, label='Residuals', color='red')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def rolling_volatility_analysis(self, window=30):
        """Analyzes volatility using rolling means and standard deviations."""
        self.data['Rolling Mean'] = self.data['Close'].rolling(window=window).mean()
        self.data['Rolling Std'] = self.data['Close'].rolling(window=window).std()

        plt.figure(figsize=(12, 6))
        plt.plot(self.data['Close'], label='Close Price', alpha=0.5)
        plt.plot(self.data['Rolling Mean'], label=f'{window}-Day Rolling Mean', color='blue')
        plt.fill_between(self.data.index, 
                         self.data['Rolling Mean'] - self.data['Rolling Std'], 
                         self.data['Rolling Mean'] + self.data['Rolling Std'], 
                         color='gray', alpha=0.3, label='Volatility (Rolling Std)')
        
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title(f'{window}-Day Rolling Mean & Volatility')
        plt.legend()
        plt.show()
    
    def risk_analysis(self, confidence_level=0.95):
        """Calculates Value at Risk (VaR) and Sharpe Ratio."""
        mean_return = self.data['Daily Returns'].mean()
        std_return = self.data['Daily Returns'].std()
        
        # Value at Risk (VaR) - Parametric Method
        VaR = mean_return - std_return * 1.65  # 95% confidence level
        
        # Sharpe Ratio (Assuming risk-free rate = 0)
        sharpe_ratio = mean_return / std_return
        
        print(f"VaR (95% confidence): {VaR:.4f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")