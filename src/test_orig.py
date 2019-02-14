from numpy import array
from numpy import hstack
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
from pandas import read_csv, DataFrame, concat
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Bidirectional
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from matplotlib import pyplot as plt


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


def split_data(data_x, data_y, train_split=0.8):
    data_split_point = int(train_split * len(data_x))
    train_X = data_x[:data_split_point]
    train_Y = data_y[:data_split_point]
    test_X = data_x[data_split_point:]
    test_y = data_y[data_split_point:]
    print(train_X.shape, train_Y.shape, test_X.shape, test_y.shape)
    return train_X, train_Y, test_X, test_y


seed = 100
np.random.seed(seed)
random.seed(seed)


data_column_names = []
data_columns = []
train_test_split = 0.8
batch_size = 1
input_dataset = read_csv('../data/full_data_1989_onwards.csv', header=0, index_col=0)
scaler = MinMaxScaler(feature_range=(0, 1))
# values = input_dataset.values
# values = np.diff(input_dataset.values)
values = scaler.fit_transform(input_dataset.values)

for i in range(values.shape[1]):
    col = values[:, i]
    data_columns.append(array(col).reshape((len(col), 1)))

dataset = hstack(data_columns)

# choose a number of time steps
n_steps_in, n_steps_out = 8, 15
# covert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
train_X, train_Y, test_X, test_y = split_data(X, y, train_test_split)
# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]
# define model
model = Sequential()
model.add(LSTM(48, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(14, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1, activation='relu')))
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(train_X, train_Y, epochs=200, verbose=2, validation_split=0.15, batch_size=12)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss vs validation_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# demonstrate prediction
res = model.evaluate(test_X, test_y)
print("Result: ", res)
predicted = model.predict(test_X, batch_size=batch_size)

scaler_val = scaler.scale_.data[0]
for arr_index in range(len(predicted)):
    for el_index in range(len(predicted[arr_index])):
        predicted[arr_index][el_index] = predicted[arr_index][el_index]/scaler_val

col = values[:, 0]
for arr_index in range(len(col)):
    col[arr_index] = col[arr_index]/scaler_val

print(predicted[-5:])
train_length = int(len(col)*train_test_split)

plt.title("{} (this is the last prediction only)".format(input_dataset.keys()[0]))
plt.xlabel("Time units (months)")
plt.ylabel("{} (£s)".format(input_dataset.keys()[0]))
plt.plot(col)

start = len(col)-len(predicted)
inner_len = len(predicted[0])

j = 0
for i in range(len(predicted)):
    if j % 5 == 0:
        plt.plot(list(range(start+i, start+i+inner_len)), predicted[i])
    j += 1

plt.show()