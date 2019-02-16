from math import sqrt
import numpy as np
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
import os.path
from keras.models import load_model
import random
import time


class LSTMPredict:
    def __init__(self, dataset, col_to_predict, look_back_period=3, num_forecasts=5, training_split=0.85, lstm_units=16,
                 loss_function='mean_squared_error', optimizer='adam', num_epochs=50, batch_size=1,
                 lstm_activation=None, dropout_rate=0.1, display_prediction_graph=True, save_model=True,
                 file_path='../results/my_model.h5', load_model_from_file=False):

        self.dataset = dataset
        self.original_dataset = dataset
        self.reframed = None
        self.train = None
        self.test = None
        self.model = None
        self.col_to_predict = col_to_predict
        self.look_back_period = look_back_period
        self.num_forecasts = num_forecasts
        self.training_split = training_split
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lstm_activation = lstm_activation
        self.dropout_rate = dropout_rate
        self.lstm_units = lstm_units
        self.display_prediction_graph = display_prediction_graph
        self.save_model = save_model
        self.load_model_file = load_model_from_file
        self.file_path = file_path
        seed = 10
        np.random.seed(seed)
        random.seed(seed)

    def organise_dataset(self):
        self.original_dataset = self.dataset
        self.dataset = self.dataset.iloc[0:]
        self.dataset = (self.dataset - self.dataset.mean()) / self.dataset.std()
        self.original_dataset = (self.original_dataset - self.original_dataset.mean()) / self.original_dataset.std()

        cols = self.dataset.columns.tolist()
        cols.remove(self.col_to_predict)
        cols.append(self.col_to_predict)
        self.dataset = self.dataset[cols]

    def series_to_supervised(self):
        data = self.dataset.values.astype('float32')
        n_vars = 1 if type(data) is list else data.shape[1]
        self.dataset = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(self.look_back_period, 0, -1):
            cols.append(self.dataset.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, self.num_forecasts):
            cols.append(self.dataset.shift(-i).iloc[:, -1])
            if i == 0:
                names += ['VAR(t)']
            else:
                names += ['VAR(t+%d)' % i]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        agg.dropna(inplace=True)
        self.reframed = agg

    def data_split(self):
        # Define and Fit Model
        values = self.reframed.values
        n_train = int(len(values) * self.training_split)
        self.train = values[:n_train, :]
        self.test = values[n_train:, :]

    def fit_lstm(self):
        # reshape training into [samples, timesteps, features]
        X, y = self.train[:, :-self.num_forecasts], self.train[:, -self.num_forecasts:]
        X = X.reshape(X.shape[0], self.look_back_period, int(X.shape[1] / self.look_back_period))

        # design network
        self.model = Sequential()
        self.model.add(LSTM(self.lstm_units, batch_input_shape=(self.batch_size, X.shape[1], X.shape[2]),
                            stateful=True))
        self.model.add(Dense(self.num_forecasts))
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer)
        # fit network
        start, updating_t = time.time(), time.time()
        for i in range(self.num_epochs):
            self.model.fit(X, y, epochs=1, batch_size=self.batch_size, verbose=0, shuffle=False)
            self.model.reset_states()
            print("Epoch {}/{} - {}s".format(i, self.num_epochs, round(time.time() - updating_t, 2)))
            updating_t = time.time()
        print("Completed in {}s".format(round(time.time()-start, 2)))

    # make one forecast with an LSTM,
    def forecast_lstm(self, x_data):
        # reshape input pattern to [samples, timesteps, features]
        x_data = x_data.reshape(1, self.look_back_period, int(len(x_data) / self.look_back_period))
        # make forecast
        forecast = self.model.predict(x_data, batch_size=self.batch_size)
        # convert to array
        return [x for x in forecast[0, :]]

    def make_forecasts(self):
        forecasts = list()
        for i in range(len(self.test)):
            X = self.test[i, :-self.num_forecasts]
            # make forecast
            forecast = self.forecast_lstm(X)
            # store the forecast
            forecasts.append(forecast)
        return forecasts

    def evaluate_forecasts(self, y, forecasts):
        for i in range(self.num_forecasts):
            actual = [row[i] for row in y]
            predicted = [forecast[i] for forecast in forecasts]
            rmse = sqrt(mean_squared_error(actual, predicted))
            print('t+%d RMSE: %f' % ((i + 1), rmse))

    # plot the forecasts in the context of the original dataset, multiple segments
    def plot_forecasts(self, forecasts, linestyle=None):
        # plot the entire dataset in blue
        series = self.original_dataset[self.col_to_predict].values
        n_test = self.test.shape[0] + self.num_forecasts - 1
        pyplot.figure()
        if linestyle is None:
            pyplot.plot(series, label='observed')
        else:
            pyplot.plot(series, linestyle, label='observed')
        pyplot.legend(loc='upper right')
        # plot the forecasts in red
        for i in range(len(forecasts)):
            if i % self.num_forecasts == 0:
                off_s = len(series) - n_test + 2 + i - 1
                off_e = off_s + len(forecasts[i]) + 1
                xaxis = [x for x in range(off_s, off_e)]
                yaxis = [series[off_s]] + forecasts[i]
                pyplot.plot(xaxis, yaxis, 'r')
        pyplot.show()

    def start(self):
        # fit model
        self.organise_dataset()
        self.series_to_supervised()
        self.data_split()

        if self.load_model_file and os.path.exists(self.file_path):
            self.model = load_model(self.file_path)
        else:
            self.fit_lstm()
            if self.save_model:
                self.model.save(self.file_path)

        forecasts = self.make_forecasts()

        # evaluate forecasts
        actual = [row[-self.num_forecasts:] for row in self.test]
        self.evaluate_forecasts(y=actual, forecasts=forecasts)

        # plot forecasts
        self.plot_forecasts(forecasts=forecasts)


data = read_csv('../data/full_data_1989_onwards_large_cols.csv', header=0, index_col=0)

LSTMPredict(
    dataset=data,
    col_to_predict='GDP',
    look_back_period=15,
    num_forecasts=5,
    training_split=0.8,
    lstm_units=12,
    num_epochs=25
).start()
