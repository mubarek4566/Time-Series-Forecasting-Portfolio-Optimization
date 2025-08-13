import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import datetime
# LSTM Model creation function with the specified parameters
def create_model(time_steps=60, lstm_units=100, dropout_rate=0.2):
    model = Sequential([
        LSTM(units=lstm_units, return_sequences=True, input_shape=(time_steps, 1)),
        Dropout(dropout_rate),
        LSTM(units=lstm_units, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(units=lstm_units),
        Dropout(dropout_rate),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# LSTM Forecaster class with the model creation and training methods
class LSTM_search:
    def __init__(self, data, time_steps=60):
        self.data = data
        self.time_steps = time_steps
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaled_data = self.scaler.fit_transform(self.data['Close'].values.reshape(-1, 1))
        self.model = None

    def prepare_data(self):
        """Prepare the data for training by creating sequences of time_steps"""
        X, y = [], []
        for i in range(self.time_steps, len(self.scaled_data)):
            X.append(self.scaled_data[i - self.time_steps:i, 0])
            y.append(self.scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM
        return X, y

    def train_model(self, batch_size=32, dropout_rate=0.2, epochs=20, lstm_units=100):
        """Train the LSTM model with hyperparameters and save it"""
        X, y = self.prepare_data()
        self.model = create_model(self.time_steps, lstm_units, dropout_rate)
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)
        # Save the trained model
        self.model.save('lstm_model.h5')  # Save model to file

    def forecast(self, steps=30):
        """Generate future predictions based on the last `time_steps`"""
        last_data = self.scaled_data[-self.time_steps:].reshape(1, self.time_steps, 1)
        predictions = []
        for _ in range(steps):
            pred = self.model.predict(last_data, verbose=0)
            predictions.append(pred[0, 0])  # Get scalar value from the prediction
            last_data = np.append(last_data[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

        return self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    def load_trained_model(self, model_path='lstm_model.h5'):
        """Load a pre-trained model from file"""
        self.model = load_model(model_path)
        return self.model


    def evaluate_performance(self, actual_values, predicted_values):
        """Evaluate the performance of the model using common metrics"""
        mae = mean_absolute_error(actual_values, predicted_values)
        mse = mean_squared_error(actual_values, predicted_values)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual_values, predicted_values)

        print(f'Mean Absolute Error (MAE): {mae}')
        print(f'Mean Squared Error (MSE): {mse}')
        print(f'Root Mean Squared Error (RMSE): {rmse}')
        print(f'R-squared (R²): {r2}')

    def forecast_next_day(self):
        """Forecast the market value for the next day"""
        forecast = self.forecast(steps=1)
        next_day = datetime.datetime.today() + datetime.timedelta(days=1)
        return next_day.strftime('%Y-%m-%d'), forecast[0]

    def forecast_next_30_days(self):
        """Forecast the market value for the next 30 days"""
        forecast = self.forecast(steps=30)
        next_30_days = [datetime.datetime.today() + datetime.timedelta(days=i) for i in range(1, 31)]
        forecast_30_days = list(zip([d.strftime('%Y-%m-%d') for d in next_30_days], forecast))
        return forecast_30_days

    def forecast_selected_date(self, start_date, end_date):
        """Forecast the market value for a selected date range"""
        start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        days = (end - start).days + 1

        forecast = self.forecast(steps=days)
        forecast_dates = [start + datetime.timedelta(days=i) for i in range(days)]
        forecast_selected_dates = list(zip([d.strftime('%Y-%m-%d') for d in forecast_dates], forecast))
        return forecast_selected_dates
    

    def evaluate_and_forecast(self, test_size=0.2, batch_size=32, dropout_rate=0.2, epochs=20, lstm_units=100):
        """Train, evaluate the model, and forecast the next 30 days."""
        # Split the data into train and test sets
        train_size = int(len(self.data) * (1 - test_size))
        train_data, test_data = self.data[:train_size], self.data[train_size:]

        # Initialize the forecaster with training data and train the model
        forecaster_train = LSTMForecaster(train_data)
        forecaster_train.train_model(batch_size=batch_size, dropout_rate=dropout_rate, epochs=epochs, lstm_units=lstm_units)

        # Prepare test data and make predictions
        X_test, y_test = forecaster_train.prepare_data()
        predicted_values = forecaster_train.model.predict(X_test)

        # Inverse transform the predictions and actual values
        predicted_values = forecaster_train.scaler.inverse_transform(predicted_values.reshape(-1, 1)).flatten()
        actual_values = forecaster_train.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        # Evaluate the performance
        forecaster_train.evaluate_performance(actual_values, predicted_values)

        # Forecast the next 30 days
        forecast_30_days = forecaster_train.forecast_next_30_days()

        return forecast_30_days
    
    def predict_test_data(self, test_data):
        """Load the trained model and predict test data"""
        if self.model is None:
            self.load_trained_model()  # Load the saved model if not already loaded
        
        # Scale the test data using the same scaler
        scaled_test_data = self.scaler.transform(test_data['Close'].values.reshape(-1, 1))
        
        # Prepare test data sequences
        X_test = []
        for i in range(self.time_steps, len(scaled_test_data)):
            X_test.append(scaled_test_data[i - self.time_steps:i, 0])
        
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  # Reshape for LSTM
        
        # Predict using the trained model
        predicted_scaled = self.model.predict(X_test)
        
        # Inverse transform the predictions to get actual values
        predicted_values = self.scaler.inverse_transform(predicted_scaled)
        
        return predicted_values.flatten()



