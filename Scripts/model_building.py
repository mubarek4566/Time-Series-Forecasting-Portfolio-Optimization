import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler, StandardScaler  
from sklearn.model_selection import GridSearchCV

class ARIMAmodel:
    def __init__(self, data, order=(1, 1, 1)):
        self.data = data
        self.data1 = self.data.asfreq('B')  # Set frequency to business days
        self.order = order
        self.model = None
        self.fitted_model = None
    
    def train_model(self):
        self.model = ARIMA(self.data1, order=self.order)
        self.fitted_model = self.model.fit()
        joblib.dump(self.fitted_model, 'arima_model.pkl')
    
    def forecast(self, steps=30):
        self.fitted_model = joblib.load('arima_model.pkl')
        forecast_result = self.fitted_model.get_forecast(steps=steps)
        forecast_mean = forecast_result.predicted_mean
        confidence_intervals = forecast_result.conf_int()
        return forecast_mean, confidence_intervals
    
    def evaluate(self, test_data):
        predictions, _ = self.forecast(steps=len(test_data))
        mae = mean_absolute_error(test_data, predictions)
        rmse = np.sqrt(mean_squared_error(test_data, predictions))
        mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
        r2 = r2_score(test_data, predictions)
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, "R Squered:": r2}
    def predict_future(self, steps=60):
        return self.forecast(steps=steps)