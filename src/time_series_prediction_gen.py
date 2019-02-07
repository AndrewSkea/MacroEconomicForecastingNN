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
from keras.layers import Dense, Dropout
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers import LSTM

col_to_predict = int(os.environ.get('COLUMN_TO_PREDICT'))
cols_to_use_for_prediction = [int(x) for x in os.environ.get('COLUMNS_FOR_PREDICTION').split(',')]
look_back_period = int(os.environ.get('LOOK_BACK_PERIOD'))
training_split = float(os.environ.get('TRAINING_SPLIT'))
loss_function = str(os.environ.get('LOSS_FUNCTION'))
optimizer = str(os.environ.get('OPTIMIZER'))
num_epochs = int(os.environ.get('NUM_EPOCHS'))
steps_per_epoch = int(os.environ.get('STEPS_PER_EPOCH'))
batch_size = int(os.environ.get('BATCH_SIZE'))
LSTM_activation = str(os.environ.get('LSTM_ACTIVATION'))
LSTM_units = int(os.environ.get('LSTM_UNITS'))
DISPLAY_INITIAL_GRAPHS = bool(os.environ.get('DISPLAY_INITIAL_GRAPHS'))

# load dataset
dataset = read_csv('../data/full_data_1981_onwards_no_nan.csv', header=0, index_col=0)
values = dataset.values
# specify the number of lag hours
n_features = values.shape[1]
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
values = scaler.fit_transform(values)

n_train_points = int(training_split * values.shape[0])
train = values[:n_train_points, :]
test = values[n_train_points:, :]

# split into input and outputs
n_obs = look_back_period * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
print("Total number of rows: {}\nRows used for Training: {}\nRows used for testing: {}\nNumber of variables used: {}"
      .format(values.shape[0], train_X.shape[0], test_X.shape[0], train_X.shape[1]))

generator = TimeseriesGenerator(train_X, train_y, look_back_period, batch_size=batch_size)

layers = [
    LSTM(100, activation=LSTM_activation, input_shape=(look_back_period, train_X.shape[1])),
    Dropout(0.1),
    Dense(1, activation='sigmoid')
]

model = Sequential(layers)

model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

model.fit_generator(generator=generator, epochs=num_epochs, verbose=2, steps_per_epoch=steps_per_epoch)

test_generator = TimeseriesGenerator(test_X, test_y, look_back_period, batch_size=batch_size)

accuracy_results = model.evaluate_generator(test_generator)
print("MAE: {}\nAccuracy: {}".format(accuracy_results[0], accuracy_results[1]))

results = model.predict_generator(test_generator)

# invert scaling for forecast
results_with_test_x = concatenate((results, test_X[-len(results):, -n_features+1:]), axis=1)
results_with_test_x = scaler.inverse_transform(results_with_test_x)
results_with_test_x = results_with_test_x[:, 0]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -n_features+1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[-len(results):, 0]

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, results_with_test_x))
print('Test RMSE: %.3f' % rmse)
pyplot.plot(results_with_test_x, label='Prediction')
pyplot.plot(inv_y, label='True')
pyplot.xlabel("Time units (months)")
pyplot.ylabel(dataset.keys()[0])
pyplot.title("{} prediction vs real data".format(dataset.keys()[0]))
pyplot.legend()
pyplot.show()
