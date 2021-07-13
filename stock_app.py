import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

app = dash.Dash()
server = app.server

scaler = MinMaxScaler(feature_range=(0, 1))

df_nse = pd.read_csv("NSE-TATA.csv")

df_nse["Date"] = pd.to_datetime(df_nse['Date'], format="%Y-%m-%d")
df_nse.index = df_nse['Date']

data = df_nse.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0, len(df_nse)), columns=['Date', 'Close'])

for i in range(0, len(data)):
    new_data["Date"][i] = data['Date'][i]
    new_data["Close"][i] = data["Close"][i]

new_data.index = new_data['Date']
new_data.drop("Date", axis=1, inplace=True)

dataset = new_data.values

train = dataset[0:987, :]
valid = dataset[987:, :]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []

for i in range(60, len(train)):
    x_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = load_model("lstm_model.h5")

inputs = new_data[len(new_data)-len(valid)-60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

train = new_data[:987]
valid = new_data[987:]
valid['Prediction'] = closing_price

# Moving Avenger
df_nse['EMA_9'] = df_nse['Close'].ewm(9).mean().shift()
df_nse['SMA_5'] = df_nse['Close'].rolling(5).mean().shift()
df_nse['SMA_10'] = df_nse['Close'].rolling(10).mean().shift()
df_nse['SMA_15'] = df_nse['Close'].rolling(15).mean().shift()
df_nse['SMA_30'] = df_nse['Close'].rolling(30).mean().shift()

# Rate of Change (ROC)


def rate_of_change(data, n):
    N = data['Close'].diff(n)
    D = data['Close'].shift(n)
    ROC = pd.Series(N/D, name='ROC')
    return ROC


df_nse['ROC'] = rate_of_change(df_nse, 5)

app.layout = html.Div([
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='NSE-TATAGLOBAL Stock Data', children=[
            html.Div([
                html.H2("Actual closing price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data",
                    figure={
                        "data": [
                            go.Scatter(x=df_nse['Date'], y=df_nse['Close'])
                        ],
                        "layout": go.Layout(xaxis={'title': 'Date'}, yaxis={'title': 'Close Price'})
                    }
                ),
                html.H2("LSTM Predicted closing price",
                        style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data LSTM",
                    figure={
                        "data": [
                            go.Scatter(x=train.index,
                                       y=train["Close"], name='Train'),
                            go.Scatter(x=valid.index,
                                       y=valid["Prediction"], name='Valid')
                        ],
                        "layout": go.Layout(xaxis={'title': 'Date'}, yaxis={'title': 'Closing Rate'})
                    }
                ),
                html.H2("Rate of Change", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data ROC",
                    figure={
                        "data": [
                            go.Scatter(
                                x=df_nse['Date'], y=df_nse['ROC'], name='ROC')
                        ],
                        "layout":go.Layout(
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'ROC values'}
                        )
                    }
                ),
                html.H2("Moving Avenger", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data MA",
                    figure={
                        "data": [
                            go.Scatter(
                                x=df_nse['Date'], y=df_nse['EMA_9'], name='EMA 9'),
                            go.Scatter(
                                x=df_nse['Date'], y=df_nse['SMA_5'], name='SMA 5'),
                            go.Scatter(
                                x=df_nse['Date'], y=df_nse['SMA_10'], name='SMA 10'),
                            go.Scatter(
                                x=df_nse['Date'], y=df_nse['SMA_15'], name='SMA 15'),
                            go.Scatter(
                                x=df_nse['Date'], y=df_nse['SMA_30'], name='SMA 30'),
                            go.Scatter(
                                x=df_nse['Date'], y=df_nse['Close'], name='Close', opacity=0.2)
                        ]
                    }
                ),
            ])
        ]),
        dcc.Tab(label='XGBoost', children=[

        ])
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)
