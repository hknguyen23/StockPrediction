import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import xgboost as xgb

app = dash.Dash()
server = app.server

scaler = MinMaxScaler(feature_range=(0, 1))

df_nse = pd.read_csv("./data/abc.csv")

df_nse["Date"] = pd.to_datetime(df_nse['Date'])
df_nse.index = df_nse['Date']

data = df_nse.sort_index(ascending=True, axis=0)
df_nse = data
new_data = pd.DataFrame(index=range(0, len(df_nse)), columns=['Date', 'Close'])

for i in range(0, len(data)):
    new_data["Date"][i] = data['Date'][i]
    new_data["Close"][i] = data["Close"][i]

new_data.index = new_data['Date']
new_data.drop("Date", axis=1, inplace=True)

dataset = new_data.values

train_index = int(len(df_nse) * 0.7)

train = dataset[:train_index, :]
valid = dataset[train_index+1:, :]

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

train = new_data[:train_index]
valid = new_data[train_index+1:]
valid['Prediction'] = closing_price

# Rate of Change (ROC)


def rate_of_change(data, n):
    N = data['Close'].diff(n)
    D = data['Close'].shift(n)
    ROC = pd.Series(N/D, name='ROC')
    return ROC


df_nse['ROC'] = rate_of_change(df_nse, 5)

# Moving Avenger
df_nse['EMA_9'] = df_nse['Close'].ewm(9).mean().shift()
df_nse['SMA_5'] = df_nse['Close'].rolling(5).mean().shift()
df_nse['SMA_10'] = df_nse['Close'].rolling(10).mean().shift()
df_nse['SMA_15'] = df_nse['Close'].rolling(15).mean().shift()
df_nse['SMA_30'] = df_nse['Close'].rolling(30).mean().shift()

# RSI


def relative_strength_idx(df, n):
    close = df['Close']
    delta = close.diff()
    delta = delta[1:]
    pricesUp = delta.copy()
    pricesDown = delta.copy()
    pricesUp[pricesUp < 0] = 0
    pricesDown[pricesDown > 0] = 0
    rollUp = pricesUp.rolling(n).mean()
    rollDown = pricesDown.abs().rolling(n).mean()
    rs = rollUp / rollDown
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


df_nse['RSI'] = relative_strength_idx(df_nse, 14).fillna(0)

df = pd.read_csv("./data/abc.csv")
df.head()
df["Date"] = pd.to_datetime(df['Date'])
df.index = df['Date']
df = df.sort_index(ascending=True, axis=0)

# Moving Avenger
df['EMA_9'] = df['Close'].ewm(9).mean().shift()
df['SMA_10'] = df['Close'].rolling(10).mean().shift()
df['SMA_30'] = df['Close'].rolling(30).mean().shift()

# RSI
df['RSI'] = relative_strength_idx(df, 14).fillna(0)

# Rate of Change (ROC)
df['ROC'] = rate_of_change(df, 5)

test_size = 0.15
valid_size = 0.15

test_split_idx = int(df.shape[0] * (1-test_size))
valid_split_idx = int(df.shape[0] * (1-(valid_size+test_size)))

train_df = df.iloc[:valid_split_idx].copy()
valid_df = df.iloc[valid_split_idx+1:test_split_idx].copy()
test_df = df.iloc[test_split_idx+1:].copy()

drop_cols = ['Date', 'Volume', 'Open', 'Low', 'High', 'OpenInt']

train_df = train_df.drop(drop_cols, 1)
valid_df = valid_df.drop(drop_cols, 1)
test_df = test_df.drop(drop_cols, 1)

y_train = train_df['Close'].copy()
X_train = train_df.drop(['Close'], 1)

y_valid = valid_df['Close'].copy()
X_valid = valid_df.drop(['Close'], 1)

y_test = test_df['Close'].copy()
X_test = test_df.drop(['Close'], 1)

# XGBoost
eval_set = [(X_train, y_train), (X_valid, y_valid)]
model_xgb = xgb.XGBRegressor()
model_xgb.load_model('xgboost_model.h5')
model_xgb.fit(X_train, y_train, eval_set=eval_set, verbose=False)

y_pred = model_xgb.predict(X_test)
predicted_prices = df.iloc[test_split_idx+1:].copy()
predicted_prices['Close'] = y_pred

app.layout = html.Div([
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='LSTM', children=[
            html.Div([
                html.H2("LSTM Predicted closing price",
                        style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data LSTM",
                    figure={
                        "data": [
                            go.Scatter(x=df_nse['Date'],
                                       y=df_nse["Close"], name='Truth'),
                            go.Scatter(x=valid.index,
                                       y=valid["Prediction"], name='Prediction')
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
                html.H2("Relative Strength Index",
                        style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data RSI",
                    figure={
                        "data": [
                            go.Scatter(
                                x=df_nse['Date'], y=df_nse['RSI'], name='RSI')
                        ]
                    }
                )
            ])
        ]),
        dcc.Tab(label='XGBoost', children=[
            html.Div([
                html.H2("XGBoost", style={"textAlign": "center"}),
                dcc.Graph(
                    id="XGBoost",
                    figure={
                        "data": [
                            go.Scatter(
                                x=df['Date'],
                                y=df['Close'],
                                name='Truth',
                                marker_color='LightSkyBlue'
                            ),
                            go.Scatter(
                                x=predicted_prices['Date'],
                                y=y_pred,
                                name='Prediction',
                                marker_color='MediumPurple'
                            )
                        ],
                        "layout":go.Layout(
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Close Price'}
                        )
                    }
                )
            ])
        ])
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)
