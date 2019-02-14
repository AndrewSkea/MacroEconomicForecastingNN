from numpy import array
from numpy import hstack
import numpy as np
from pandas import read_csv, DataFrame, concat
from keras.models import Sequential
from keras.layers import LSTM, Dropout
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed


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

    x_data = concat(x_cols, axis=1)
    x_data.columns = x_col_names

    original_y_data = data[y_column]
    for k in range(num_forecasts):
        y_cols.append(original_y_data.shift(-k))

    y_data = concat(y_cols, axis=1)
    y_data.columns = y_col_names

    x_data = x_data.dropna()
    y_data = y_data.dropna()
    x_data = np.array(x_data).reshape(len(x_data), lags, len(column_names))
    y_data = np.array(y_data)
    x_data = x_data[:len(y_data)]
    y_data = y_data[:len(x_data)]
    return x_data, y_data


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], [[x] for x in sequences[end_ix:out_end_ix, 0]]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# data_column_names = []
# data_columns = []
#
#
# input_dataset = read_csv('../data/full_data_1987_onwards.csv', header=0, index_col=0)
#
# for col in input_dataset:


# define input sequence
in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])

# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps_in, n_steps_out = 3, 5
# covert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]
# define model
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(Dropout(0.1))
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=300, verbose=2)
# demonstrate prediction
x_input = array([[60, 65, 125], [70, 75, 145], [80, 85, 165]])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=2)
print(yhat)