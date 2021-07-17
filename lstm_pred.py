import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
import pandas as pd
import numpy as np

df_lstm = pd.read_csv("./data/abc.csv")

df_lstm['Date'] = pd.to_datetime(df_lstm['Date'])
df_lstm.index = df_lstm['Date']

data = df_lstm.sort_index(ascending=True, axis=0)
df_lstm = data
new_data = pd.DataFrame(index=range(0, len(df_lstm)),
                        columns=['Date', 'Close'])

for i in range(0, len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

new_data.index = new_data['Date']
new_data.drop('Date', axis=1, inplace=True)

dataset = new_data.values

train_index = int(len(df_lstm) * 0.7)
train_dataset = dataset[:train_index, :]
test_dataset = dataset[train_index:, :]

scaler = MinMaxScaler(feature_range=(0, 1))
train_dataset = scaler.fit_transform(train_dataset)
test_dataset = scaler.transform(dataset)

def create_dataset(df):
    x = []
    y = []
    for i in range(60, df.shape[0]):
        x.append(df[i-60:i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x,y

x_train, y_train = create_dataset(train_dataset)
x_test, y_test = create_dataset(test_dataset)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

model = Sequential()
model.add(LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=96,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1)
model.save("lstm_model.h5")
