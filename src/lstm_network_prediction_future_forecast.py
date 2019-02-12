import numpy as np
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from pandas import DataFrame
from numpy import concatenate
import pandas as pd
import matplotlib.pyplot as plt


def series_to_supervised(data, _scaler, lags=1, y_column=1, num_forecasts=1):
    data = DataFrame(data)
    column_names = list(data.keys())
    values = data.values
    # values = np.diff(np.array(data.values), n=0)
    values = _scaler.fit_transform(values)
    data = DataFrame(values)
    x_cols, y_cols, names = list(), list(), list()
    x_col_names, y_col_names = list(), list()

    for n in range(lags):
        for col in column_names:
            x_col_names.append("{}(t-{})".format(col, n))

    for n in range(num_forecasts):
        y_col_names.append("{}(t+{})".format(column_names[y_column], n))

    for j in range(lags):
        x_cols.append(data.shift(j))

    x_data = pd.concat(x_cols, axis=1)
    x_data.columns = x_col_names

    original_y_data = data[y_column]
    for k in range(num_forecasts):
        y_cols.append(original_y_data.shift(-k))

    y_data = pd.concat(y_cols, axis=1)
    y_data.columns = y_col_names

    x_data = x_data.dropna()
    y_data = y_data.dropna()
    x_data = np.array(x_data).reshape(len(x_data), lags, len(column_names))
    y_data = np.array(y_data)
    x_data = x_data[:len(y_data)]
    y_data = y_data[:len(x_data)]
    return x_data, y_data


def split_data(data_x, data_y, train_split=0.8):
    data_split_point = int(train_split * len(data_x))
    train_X = data_x[:data_split_point]
    train_Y = data_y[:data_split_point]
    test_X = data_x[data_split_point:]
    test_y = data_y[data_split_point:]
    print(train_X.shape, train_Y.shape, test_X.shape, test_y.shape)
    return train_X, train_Y, test_X, test_y


def create_model():
    _model = Sequential()
    _model.add(LSTM(num_forecasts*2, input_shape=(lags, train_X.shape[2]), return_sequences=True))
    _model.add(Dropout(0.2))
    _model.add(LSTM(num_forecasts*5, input_shape=(lags, train_X.shape[2]), return_sequences=False, activation='relu'))
    _model.add(Dropout(0.2))
    _model.add(Dense(num_forecasts))
    _model.compile(loss="mae", optimizer="adam")
    return _model

np.random.seed(50)
lags = 3
num_forecasts = 5
col_to_predict = 2
batch_size = 4
scaler = MinMaxScaler(feature_range=(0, 1))
train_test_split = 0.9

input_dataset = read_csv('../data/full_data_1989_onwards_large_cols.csv', header=0, index_col=0)
col = np.array(input_dataset[input_dataset.keys()[col_to_predict]])
X, Y = series_to_supervised(input_dataset, _scaler=scaler, lags=lags, y_column=col_to_predict, num_forecasts=num_forecasts)
train_X, train_Y, test_X, test_y = split_data(X, Y, train_split=train_test_split)

model = create_model()
model.fit(train_X, train_Y, epochs=50, validation_split=0.2, verbose=2, batch_size=batch_size)

print("Evaluation result:", model.evaluate(test_X, test_y))
predicted = model.predict(test_X, batch_size=batch_size)

# This is rescaling the data with scaling value for the specified column
scaler_val = scaler.scale_.data[col_to_predict]
print(scaler_val)
for arr_index in range(len(predicted)):
    for el_index in range(len(predicted[arr_index])):
        predicted[arr_index][el_index] = predicted[arr_index][el_index]/scaler_val

print(predicted)
train_length = int(len(col)*train_test_split)

plt.title(input_dataset.keys()[col_to_predict])
plt.plot(list(range(len(col))), col)

# This to show the last value's prediction
# plt.plot(list(range(train_length + len(predicted)-1, train_length + num_forecasts + len(predicted)-1)), predicted[-1])

# This to show all values' predictions
i = 0
for ar in predicted:
    plt.plot(list(range(train_length + i, train_length + num_forecasts + i)), ar)
    i += 1
plt.show()
