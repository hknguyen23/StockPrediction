import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("./data/abc.csv")
df.head()
df['Date'] = pd.to_datetime(df['Date'])
df.index = df['Date']

# Moving Avenger
df['EMA_9'] = df['Close'].ewm(9).mean().shift()
df['SMA_10'] = df['Close'].rolling(10).mean().shift()
df['SMA_30'] = df['Close'].rolling(30).mean().shift()

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

df['RSI'] = relative_strength_idx(df, 14).fillna(0)

# Rate of Change (ROC)
def rate_of_change(data, n):
    N = data['Close'].diff(n)
    D = data['Close'].shift(n)
    ROC = pd.Series(N/D, name='ROC')
    return ROC


df['ROC'] = rate_of_change(df, 5)

# Bollinger Bands
def bollinger_bands(data, n):
    MA = data['Close'].rolling(n).mean()
    SD = data['Close'].rolling(n).std()
    data['UpperBB'] = MA + (2 * SD) 
    data['LowerBB'] = MA - (2 * SD)
    return data

BB = bollinger_bands(df, 50)
df['UpperBB'] = BB['UpperBB']
df['LowerBB'] = BB['LowerBB']

test_size = 0.2

test_split_idx = int(df.shape[0] * (1-test_size))

train_df = df.iloc[:test_split_idx].copy()
test_df = df.iloc[test_split_idx:].copy()

drop_cols = ['Date', 'Volume', 'Open', 'Low', 'High', 'OpenInt']

train_df = train_df.drop(drop_cols, 1)
test_df = test_df.drop(drop_cols, 1)

y_train = train_df['Close'].copy()
X_train = train_df.drop(['Close'], 1)

y_test = test_df['Close'].copy()
X_test = test_df.drop(['Close'], 1)

X_train.info()

parameters = {
    'n_estimators': [100],
    'learning_rate': [0.1],
    'max_depth': [12],
    'gamma': [0.01]
}

eval_set = [(X_train, y_train)]
model = xgb.XGBRegressor(objective='reg:squarederror')
clf = GridSearchCV(model, parameters, verbose=False)

clf.fit(X_train, y_train, eval_set=eval_set)

print(f'Best params: {clf.best_params_}')
print(f'Best validation score = {clf.best_score_}')

model = xgb.XGBRegressor(**clf.best_params_, objective='reg:squarederror')
model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

model.save_model('xgboost_model.h5')
