import yfinance as yf
import pandas as pd

class StockData:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.close_df = None
        self.high_df = None
        self.low_df = None
        self.open_df = None
        self.volume_df = None
        self.df_merged = None
    
    def download_data(self):
        """Download stock data from Yahoo Finance."""
        self.data = yf.download(self.tickers, start=self.start_date, end=self.end_date)
        return self.data

    def extract_price_categories(self):
        """Extract different price categories into separate DataFrames."""
        self.close_df = self.data.xs('Close', axis=1, level=0)
        self.high_df = self.data.xs('High', axis=1, level=0)
        self.low_df = self.data.xs('Low', axis=1, level=0)
        self.open_df = self.data.xs('Open', axis=1, level=0)
        self.volume_df = self.data.xs('Volume', axis=1, level=0)
        return self.close_df,self.high_df,self.low_df,self.open_df,self.volume_df

    def reset_indexes(self):
        """Reset index of DataFrames."""
        self.close_df = self.close_df.reset_index()
        self.high_df = self.high_df.reset_index()
        self.low_df = self.low_df.reset_index()
        self.open_df = self.open_df.reset_index()
        self.volume_df = self.volume_df.reset_index()
        return self.close_df,self.high_df,self.low_df,self.open_df,self.volume_df
    
    def melt_data(self):
        """Melt DataFrames from wide to long format."""
        self.close_df = self.close_df.melt(id_vars=['Date'], value_vars=self.tickers, 
                                            var_name='Ticker', value_name='Close')
        self.high_df = self.high_df.melt(id_vars=['Date'], value_vars=self.tickers, 
                                          var_name='Ticker', value_name='High')
        self.low_df = self.low_df.melt(id_vars=['Date'], value_vars=self.tickers, 
                                        var_name='Ticker', value_name='Low')
        self.volume_df = self.volume_df.melt(id_vars=['Date'], value_vars=self.tickers, 
                                             var_name='Ticker', value_name='Volume')
        self.open_df = self.open_df.melt(id_vars=['Date'], value_vars=self.tickers, 
                                         var_name='Ticker', value_name='Open')
        return self.close_df,self.high_df,self.low_df,self.open_df,self.volume_df
    
    def merge_data(self):
        """Merge all melted DataFrames into a single DataFrame."""
        self.df_merged = pd.merge(self.open_df, self.low_df, on=['Date', 'Ticker'])
        self.df_merged = pd.merge(self.df_merged, self.volume_df, on=['Date', 'Ticker'])
        self.df_merged = pd.merge(self.df_merged, self.high_df, on=['Date', 'Ticker'])
        self.df_merged = pd.merge(self.df_merged, self.close_df, on=['Date', 'Ticker'])

    def get_merged_data(self):
        """Return the final merged DataFrame."""
        return self.df_merged