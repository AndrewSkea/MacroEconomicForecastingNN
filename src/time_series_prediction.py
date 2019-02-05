from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

col_to_predict = int(os.environ.get('COLUMN_TO_PREDICT'))
cols_to_use_for_prediction = [int(x) for x in os.environ.get('COLUMNS_FOR_PREDICTION').split(',')]
layer_setup = [int(x) for x in os.environ.get('LAYER_SETUP').replace(']', '').replace('[', '').split(',')]
look_back_period = int(os.environ.get('LOOK_BACK_PERIOD'))
training_split = float(os.environ.get('TRAINING_SPLIT'))
loss_function = str(os.environ.get('LOSS_FUNCTION'))
optimizer = str(os.environ.get('OPTIMIZER'))
num_epochs = int(os.environ.get('NUM_EPOCHS'))
batch_size = int(os.environ.get('BATCH_SIZE'))
LSTM_units = int(os.environ.get('LSTM_UNITS'))


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# load dataset
dataset = read_csv('../data/uk_data.csv', header=0, index_col=0)
values = dataset.values
# integer encode direction
print("Input shape: ", values.shape)
# specify the number of lag hours
n_features = values.shape[1]
# specify columns to plot
groups = [j for j in range(0, n_features)]
i = 1
# plot each column
pyplot.figure()
for group in groups:
    pyplot.subplot(len(groups), 1, i)
    pyplot.plot(values[:, group])
    pyplot.title(dataset.columns[group], y=0.5, loc='right')
    i += 1
pyplot.show()

encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# frame as supervised learning
reframed = series_to_supervised(scaled, look_back_period, 1)
print("Reframed shape: ", reframed.shape)

# split into train and test sets
values = reframed.values
n_train_points = int(training_split * reframed.shape[0])
train = values[:n_train_points, :]
test = values[n_train_points:, :]
# split into input and outputs
n_obs = look_back_period * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
print(train_X.shape, len(train_X), train_y.shape)
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], look_back_period, n_features))
test_X = test_X.reshape((test_X.shape[0], look_back_period, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(5, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss=loss_function, optimizer=optimizer)
# fit network
history = model.fit(train_X, train_y, epochs=num_epochs, batch_size=batch_size,
                    verbose=2, validation_data=(test_X, test_y), shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], look_back_period * n_features))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, -n_features+1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -n_features+1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
pyplot.plot(inv_yhat, label='inv_yhat')
pyplot.plot(inv_y, label='inv_y')
pyplot.legend()
pyplot.show()
