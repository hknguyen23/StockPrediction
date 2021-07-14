from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 20, 10

df = pd.read_csv("./data/abc.csv")
df.head()

df["Date"] = pd.to_datetime(df.Date)
df.index = df['Date']

plt.figure(figsize=(16, 8))
plt.plot(df["Close"], label='Close Price history')

data = df.sort_index(ascending=True, axis=0)
new_dataset = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])

for i in range(0, len(data)):
    new_dataset["Date"][i] = data['Date'][i]
    new_dataset["Close"][i] = data["Close"][i]

scaler = MinMaxScaler(feature_range=(0, 1))

new_dataset.index = new_dataset['Date']
new_dataset.drop("Date", axis=1, inplace=True)
final_dataset = new_dataset.values

train_index = int(len(df) * 0.7)

train_data = final_dataset[:train_index, :]
valid_data = final_dataset[train_index+1:, :]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(final_dataset)

x_train_data, y_train_data = [], []

for i in range(60, len(train_data)):
    x_train_data.append(scaled_data[i-60:i, 0])
    y_train_data.append(scaled_data[i, 0])

x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

x_train_data = np.reshape(
    x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True,
               input_shape=(x_train_data.shape[1], 1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

inputs_data = new_dataset[len(new_dataset)-len(valid_data)-60:].values
inputs_data = inputs_data.reshape(-1, 1)
inputs_data = scaler.transform(inputs_data)

lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose=2)
lstm_model.save("lstm_model.h5")
