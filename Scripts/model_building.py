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


class SARIMA:
    def __init__(self, data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        # Ensure the index is a DateTimeIndex with frequency
        self.data = data.copy()
        self.data.index = pd.to_datetime(self.data.index)
        self.data = self.data.asfreq(self._infer_frequency())

        # self.data = self.data.dropna()
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        self.scaler = StandardScaler()
        
        # Normalize the data
        self.data_scaled = self.scaler.fit_transform(self.data.values.reshape(-1, 1))
        # self.data_scaled = self.data_scaled.dropna()
    def _infer_frequency(self):
        """Infer the frequency of the time series."""
        inferred_freq = pd.infer_freq(self.data.index)
        return inferred_freq if inferred_freq else 'D'  # Default to daily frequency

    def train_model(self):
        """Train and save the SARIMA model."""
        self.model = SARIMAX(
            self.data_scaled,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.fitted_model = self.model.fit(disp=False, maxiter=200)
        joblib.dump(self.fitted_model, 'sarima_model.pkl')

    def forecast(self, steps=30):
        """Generate forecasts for a given number of steps."""
        self.fitted_model = joblib.load('sarima_model.pkl')

        forecast_result = self.fitted_model.get_forecast(steps=steps)
        forecast_mean = forecast_result.predicted_mean
        confidence_intervals = forecast_result.conf_int()

        # Convert forecasted values back to original scale
        forecast_mean = self.scaler.inverse_transform(forecast_mean.reshape(-1, 1)).flatten()

        # Generate date index for forecasts
        forecast_index = pd.date_range(start=self.data.index[-1], periods=steps + 1, freq=self._infer_frequency())[1:]
        forecast_series = pd.Series(forecast_mean, index=forecast_index)

        return forecast_series, confidence_intervals

    def evaluate(self, test_data):
        """Evaluate the model using MAE, RMSE, MAPE, and R-squared."""
        test_data = test_data.copy()
        test_data.index = pd.to_datetime(test_data.index)
        test_data = test_data.asfreq(self._infer_frequency())

        predictions, _ = self.forecast(steps=len(test_data))
        predictions = predictions[:len(test_data)]  # Ensure predictions match test data length

        # Handle potential NaN values
        mask = ~np.isnan(test_data) & ~np.isnan(predictions)  # Mask valid values
        test_data_clean = test_data[mask]
        predictions_clean = predictions[mask]

        if len(test_data_clean) == 0:
            raise ValueError("No valid data points after removing NaN values. Check your dataset.")

        # Compute evaluation metrics
        mae = mean_absolute_error(test_data_clean, predictions_clean)
        rmse = np.sqrt(mean_squared_error(test_data_clean, predictions_clean))
        mape = np.mean(np.abs((test_data_clean - predictions_clean) / test_data_clean)) * 100
        r2 = r2_score(test_data_clean, predictions_clean)

        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R-squared': r2}


    def predict_future(self, steps=60):
        """Predict future values for a given number of steps."""
        return self.forecast(steps=steps)