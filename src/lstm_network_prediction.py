from math import sqrt
import numpy as np
import random
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers import LSTM
from tkinter import *
from tkinter.ttk import *


class LSTMPrediction:
    def __init__(self, dataset, col_to_predict, look_back_period=3, training_split=0.85, hidden_layer_setup=(64,),
                 loss_function='mean_squared_error', optimizer='adam', num_epochs=50, steps_per_epoch=20, batch_size=4,
                 lstm_activation='tanh', lstm_units=64, display_initial_graphs=True, dropout_rate=0.1,
                 display_prediction_graph=True, display_loss_graphs=True, validation_split=0.1):

        self.dataset = dataset
        self.col_to_predict = col_to_predict
        self.look_back_period = look_back_period
        self.training_split = training_split
        self.validation_split = validation_split
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.lstm_activation = lstm_activation
        self.dropout_rate = dropout_rate
        self.lstm_units = lstm_units
        self.hidden_layer_setup = hidden_layer_setup
        self.display_initial_graphs = display_initial_graphs
        self.display_prediction_graph = display_prediction_graph
        self.display_loss_graphs = display_loss_graphs

        self.values = None
        self.number_features = 0
        self.values = None
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None
        self.validate_X = None
        self.validate_y = None
        self.generator = None
        self.validate_generator = None
        self.lstm_activation = None
        self.model = None
        self.history = None
        self.test_generator = None
        self.test_results = None
        self.validation_results = None
        self.scaler = None
        self.scaled_rmse = None
        self.results_with_test_x = None
        self.results_with_validation = None
        self.rmse = None
        self.inv_y = None

        seed = 10
        np.random.seed(seed)
        random.seed(seed)

    def validate_file(self):
        # specify the number of lag hours
        column_list = list(self.dataset.keys())
        column_list.insert(0, column_list.pop(column_list.index(self.col_to_predict)))
        self.dataset = self.dataset.ix[:, column_list]

        self.values = self.dataset.values
        self.number_features = self.values.shape[1]
        encoder = LabelEncoder()
        self.values[:, 4] = encoder.fit_transform(self.values[:, 4])
        self.values = self.values.astype('float32')

    def scale_dataset(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.values = self.scaler.fit_transform(self.values)

    def split_train_test(self):
        train_test = int(self.values.shape[0] * (1 - self.validation_split))
        n_train_points = int(self.training_split * train_test)
        train = self.values[:n_train_points, :]
        test = self.values[n_train_points:train_test, :]
        validation = self.values[train_test:, :]

        # split into input and outputs
        n_obs = self.look_back_period * self.number_features
        self.train_X, self.train_y = train[:, :n_obs], train[:, -self.number_features]
        self.test_X, self.test_y = test[:, :n_obs], test[:, -self.number_features]
        self.validate_X, self.validate_y = validation[:, :n_obs], validation[:, -self.number_features]
        print("Total number of rows: {}\nRows used for Training: {}\nRows used for testing: {}\n"
              "Rows for out-of-sample validation: {}\nNumber of variables used: {}".format(
            self.values.shape[0], self.train_X.shape[0], self.test_X.shape[0], self.validate_X.shape[0],
            self.train_X.shape[1])
        )

    def create_train_generator(self):
        self.generator = TimeseriesGenerator(self.train_X, self.train_y, self.look_back_period,
                                             batch_size=self.batch_size)

    def create_test_generator(self):
        self.test_generator = TimeseriesGenerator(self.test_X, self.test_y, self.look_back_period,
                                                  batch_size=self.batch_size)

    def create_validation_generator(self):
        self.validate_generator = TimeseriesGenerator(self.validate_X, self.validate_y, self.look_back_period,
                                                      batch_size=self.batch_size)

    def create_network(self):
        layers = [
            LSTM(self.lstm_units, activation=self.lstm_activation, go_backwards=True,
                 input_shape=(self.look_back_period, self.train_X.shape[1]), return_sequences=True),
        ]

        if self.dropout_rate > 0:
            layers.append(Dropout(self.dropout_rate))

        for num in self.hidden_layer_setup:
            layers.append(LSTM(num, return_sequences=True))

        layers.append(LSTM(1, return_sequences=False))

        self.model = Sequential(layers)

        self.model.compile(loss=self.loss_function, optimizer=self.optimizer)
        self.history = self.model.fit_generator(generator=self.generator, epochs=self.num_epochs, verbose=2,
                                                shuffle=False, validation_data=self.test_generator,
                                                steps_per_epoch=self.steps_per_epoch)

    def get_results_test_generator(self):
        self.model.reset_states()
        accuracy_results = self.model.evaluate_generator(self.test_generator)
        print("Test results: {}".format(accuracy_results))
        self.test_results = self.model.predict_generator(self.test_generator)

    def get_results_validation_generator(self):
        self.model.reset_states()
        accuracy_results = self.model.evaluate_generator(self.validate_generator)
        print("Validation results: {}".format(accuracy_results))
        self.validation_results = self.model.predict_generator(self.validate_generator)

    def get_summary_values(self):
        self.scaled_rmse = sqrt(
            mean_squared_error(self.test_results, self.test_y.reshape((len(self.test_y), 1))[-len(self.test_results):]))
        print('Test RMSE minmax: %.3f' % self.scaled_rmse)

        scaled_rmse_val = sqrt(
            mean_squared_error(self.validation_results, self.test_y.reshape((len(self.test_y), 1))[-len(self.validation_results):]))
        print('Validation RMSE minmax: %.3f' % scaled_rmse_val)

        # invert scaling for forecast
        self.results_with_test_x = concatenate(
            (self.test_results, self.test_X[:len(self.test_results), -self.number_features + 1:]), axis=1)
        self.results_with_test_x = self.scaler.inverse_transform(self.results_with_test_x)
        self.results_with_test_x = self.results_with_test_x[:, 0]

        # invert scaling for validation
        self.results_with_validation = concatenate(
            (self.validation_results, self.test_X[:len(self.validation_results), -self.number_features + 1:]),
            axis=1)
        self.results_with_validation = self.scaler.inverse_transform(self.results_with_validation)
        self.results_with_validation = self.results_with_validation[:, 0]

        # invert scaling for actual
        self.test_y = self.test_y.reshape((len(self.test_y), 1))
        self.inv_y = concatenate((self.test_y, self.test_X[:, -self.number_features + 1:]), axis=1)
        self.inv_y = self.scaler.inverse_transform(self.inv_y)
        self.inv_y = self.inv_y[-len(self.test_results):, 0]

    def plot_initial_graphs(self):
        groups = [j for j in range(0, self.number_features)]
        i = 1
        pyplot.figure()
        for group in groups:
            pyplot.subplot(len(groups), 1, i)
            pyplot.plot(self.values[:, group])
            pyplot.title(self.dataset.columns[group], y=0.5, loc='right')
            i += 1
        pyplot.show()

    def plot_loss_values(self):
        pyplot.plot(self.history.history['loss'])
        pyplot.plot(self.history.history['val_loss'])
        pyplot.title('model loss vs validation_loss')
        pyplot.ylabel('loss')
        pyplot.xlabel('epoch')
        pyplot.legend(['train', 'test'], loc='upper left')
        pyplot.show()

    def plot_prediction(self):
        series = self.dataset[self.col_to_predict].values
        full_x_values = list(range(len(series)))
        pred_x_values = list(
            range(self.train_X.shape[0] + self.look_back_period, self.train_X.shape[0] + self.test_X.shape[0]))
        validation_pred_x_values = list(range(int(len(series) - len(self.results_with_validation)), len(series)))

        pyplot.plot(full_x_values, self.dataset[self.col_to_predict].values, label='True')
        pyplot.plot(pred_x_values, self.results_with_test_x, label='Test prediction')
        pyplot.plot(validation_pred_x_values, self.results_with_validation, label='Out of sample prediction')

        pyplot.xlabel("Time units (months)")
        pyplot.ylabel(self.col_to_predict)
        pyplot.title("{} prediction vs real data".format(self.col_to_predict))
        pyplot.legend()
        pyplot.show()

    def start_prediction(self):
        self.validate_file()
        if self.display_initial_graphs:
            self.plot_initial_graphs()
        self.scale_dataset()
        self.split_train_test()
        self.create_train_generator()
        self.create_test_generator()
        self.create_validation_generator()
        self.create_network()
        if self.display_loss_graphs:
            self.plot_loss_values()
        self.get_results_test_generator()
        self.get_results_validation_generator()
        self.get_summary_values()
        if self.display_prediction_graph:
            self.plot_prediction()
        return self.rmse


def get_column_to_predict(dataset_keys):
    window = Tk()
    window.title("Please choose column to predict")

    selected = StringVar()
    rad_array = []
    for col in dataset_keys:
        rad_array.append(Radiobutton(window, text=col, value=col, variable=selected))

    def clicked():
        global column_to_predict
        column_to_predict = selected.get()
        window.destroy()
        print(column_to_predict)

    btn = Button(window, text="Choose", command=clicked)
    for i in range(len(rad_array)):
        rad_array[i].grid(column=0, row=i)

    btn.grid(column=1, row=len(rad_array))
    window.mainloop()
    return column_to_predict


if __name__ == "__main__":
    global column_to_predict
    column_to_predict = 0
    input_dataset = read_csv('../data/full_data_1987_onwards.csv', header=0, index_col=0)
    dataset_keys = list(input_dataset.keys())
    # dataset_keys.remove('Month')
    get_column_to_predict(dataset_keys)
    print("Column to predict: ", column_to_predict)
    final_result = LSTMPrediction(
        dataset=input_dataset,
        col_to_predict=column_to_predict,
        display_initial_graphs=False,
        display_loss_graphs=True,
        display_prediction_graph=True
    ).start_prediction()
    print(final_result)