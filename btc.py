import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

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
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# Split the data into training set and test set
training_size = int(len(data) * 0.8)
test_size = len(data) - training_size
train_data, test_data = data.values[0:training_size, :], data.values[training_size:len(data), :]

# Reshape the data and create time steps
look_back = 60  # we will use 60 days of data to predict the next day
X_train, y_train = create_dataset(train_data, look_back)
X_test, y_test = create_dataset(test_data, look_back)

# Reshape the input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=2)

# Use the model to make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invert the predictions to original scale
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# Get the dates from the index
train_dates = data.index[look_back:len(train_predict)+look_back]
test_dates = data.index[len(train_predict)+(look_back*2)+1:len(data)-1]

# Convert predictions to dataframes for easier handling
train_df = pd.DataFrame(data=train_predict, index=train_dates, columns=['Value'])
test_df = pd.DataFrame(data=test_predict, index=test_dates, columns=['Value'])

# Print the predictions with dates
print("Training set predictions: ")
print(train_df)
print("Test set predictions: ")
print(test_df)
