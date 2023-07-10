import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import timedelta

# Load the dataset
data = pd.read_csv('BTC.csv')

# Convert the date column to datetime and set it as the index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Sort the data by date
data.sort_index(ascending=True, inplace=True)

# Fill missing values with the previous day's value
data.fillna(method='ffill', inplace=True)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data['Value'] = scaler.fit_transform(data['Value'].values.reshape(-1, 1))

# Define a function to create a dataset for time series
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# Create training set
look_back = 60  # we will use 60 days of data to predict the next day
train_data = data.values
X_train, y_train = create_dataset(train_data, look_back)

# Reshape the input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=2)

# Use the model to predict the next 30 days
future_days = 30
predictions = []
current_batch = train_data[-look_back:]
current_batch = current_batch.reshape((1, 1, look_back))

# Get the last valid date from the dataset
last_date = data.dropna().index[-1]
future_dates = [last_date]  # Start a new list for the future dates

for i in range(future_days):
    current_pred = model.predict(current_batch)[0]
    predictions.append(current_pred)
    current_batch = np.append(current_batch[:, :, 1:], [[current_pred]], axis=2)
    last_date += timedelta(days=1)  # Update the last date
    future_dates.append(last_date)  # Append the new date to the list

# Invert the predictions to original scale
predictions = scaler.inverse_transform(predictions)

# Remove the first date from the future_dates list (it's the last date in the original data)
future_dates = future_dates[1:]

# Convert predictions to dataframes for easier handling
predictions_df = pd.DataFrame(data=predictions, index=future_dates, columns=['Value'])

# Save the predictions to a CSV file
predictions_df.to_csv('future_predictions.csv')
