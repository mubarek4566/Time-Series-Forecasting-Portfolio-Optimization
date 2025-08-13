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
    
class LSTMForecast:
    # 'batch_size': 32, 'dropout_rate': 0.2, 'epochs': 20, 'lstm_units': 100
    def __init__(self, data, lstm_units=100):
        self.data = data
        self.lstm_units = lstm_units
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # Fit the scaler once and store scaled data
        self.scaled_data = self.scaler.fit_transform(self.data.values.reshape(-1, 1))
    
    def train_model(self):
        X_train, y_train = [], []

        # Prepare training sequences
        for i in range(60, len(self.scaled_data)):
            X_train.append(self.scaled_data[i-60:i, 0])
            y_train.append(self.scaled_data[i, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Define LSTM model
        self.model = Sequential([
            Input(shape=(X_train.shape[1], 1)),  # Explicit Input layer
            LSTM(units=self.lstm_units, return_sequences=True),
            Dropout(0.2),
            LSTM(units=self.lstm_units, return_sequences=True),
            Dropout(0.2),
            LSTM(units=self.lstm_units),
            Dropout(0.2),
            Dense(units=1)
        ])

        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(X_train, y_train, epochs=20, batch_size=32)
        self.model.save('lstm_model.h5')

    def forecast(self, steps=30):
        """ Generate future predictions based on the last 60 time steps. """
        self.model = load_model('lstm_model.h5')
        X_input = self.scaled_data[-60:].reshape(1, 60, 1)  # Use stored scaled data
        predictions = []

        for _ in range(steps):
            pred = self.model.predict(X_input, verbose=0)
            predictions.append(pred[0, 0])  # Extract scalar value
            X_input = np.append(X_input[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

        # Convert predictions back to original scale
        predictions = np.array(predictions).reshape(-1, 1)
        return self.scaler.inverse_transform(predictions).flatten()

    def evaluate(self, test_data):
        """ Evaluate model performance using MAE, RMSE, MAPE, and R^2 score. """
        predictions = self.forecast(steps=len(test_data))

        # Ensure predictions and test_data are 1D arrays
        test_data = test_data.values.flatten()
        predictions = predictions.flatten()

        # Compute error metrics
        mae = mean_absolute_error(test_data, predictions)
        rmse = np.sqrt(mean_squared_error(test_data, predictions))
        mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
        r2 = r2_score(test_data, predictions)

        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, "R Squared": r2}

    def predict_future(self, steps=60):
        """ Predict future values for given steps. """
        return self.forecast(steps=steps)