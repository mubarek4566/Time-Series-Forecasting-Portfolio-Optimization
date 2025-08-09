import yfinance as yf
import pandas as pd

class StockData:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

    
    def download_data(self):
        """Download stock data from Yahoo Finance for multiple tickers and merge into one DataFrame."""
        merged_df = pd.DataFrame()

        for ticker in self.tickers:
            print(f"Downloading {ticker}...")
            df = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=False,
                timeout=30
            )

            # Flatten column names with ticker prefix
            df.columns = [f"{ticker}_{col}" for col in df.columns]

            if merged_df.empty:
                merged_df = df
            else:
                merged_df = merged_df.join(df, how="outer")

        self.data = merged_df
        return self.data
