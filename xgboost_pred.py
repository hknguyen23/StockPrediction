import pandas as pd
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("./data/NSE-TATA.csv")
df.head()
df["Date"] = pd.to_datetime(df['Date'])
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

test_size = 0.15
valid_size = 0.15

test_split_idx = int(df.shape[0] * (1-test_size))
valid_split_idx = int(df.shape[0] * (1-(valid_size+test_size)))

train_df = df.iloc[:valid_split_idx].copy()
valid_df = df.iloc[valid_split_idx+1:test_split_idx].copy()
test_df = df.iloc[test_split_idx+1:].copy()

drop_cols = ['Date', 'Open', 'Low', 'High', 'Last',
             'Total Trade Quantity', 'Turnover (Lacs)']

train_df = train_df.drop(drop_cols, 1)
valid_df = valid_df.drop(drop_cols, 1)
test_df = test_df.drop(drop_cols, 1)

y_train = train_df['Close'].copy()
X_train = train_df.drop(['Close'], 1)

y_valid = valid_df['Close'].copy()
X_valid = valid_df.drop(['Close'], 1)

y_test = test_df['Close'].copy()
X_test = test_df.drop(['Close'], 1)

X_train.info()

parameters = {
    'n_estimators': [100],
    'learning_rate': [0.1],
    'max_depth': [8],
}

eval_set = [(X_train, y_train), (X_valid, y_valid)]
model = xgb.XGBRegressor(objective='reg:squarederror')
clf = GridSearchCV(model, parameters, verbose=False)

clf.fit(X_train, y_train, eval_set=eval_set)

print(f'Best params: {clf.best_params_}')
print(f'Best validation score = {clf.best_score_}')

model = xgb.XGBRegressor(**clf.best_params_, objective='reg:squarederror')
model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

plot_importance(model)

model.save_model('xgboost_model.h5')
