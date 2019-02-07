from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers import LSTM


class LSTMPrediction:
    def __init__(self, dataset, col_to_predict, cols_to_use_for_prediction, look_back_period=3, training_split=0.8,
                 loss_function='mean_squared_error', optimizer='adam', num_epochs=200, steps_per_epoch=20, batch_size=4,
                 lstm_activation='tanh', lstm_units=50, display_initial_graphs=True, display_prediction_graph=True,
                 display_loss_graphs=True):
        self.dataset = dataset
        self.col_to_predict = col_to_predict
        self.cols_to_use_for_prediction = cols_to_use_for_prediction
        self.look_back_period = look_back_period
        self.training_split = training_split
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.LSTM_activation = lstm_activation
        self.LSTM_units = lstm_units
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
        self.generator = None
        self.lstm_activation = None
        self.model = None
        self.history = None
        self.test_generator = None
        self.results = None
        self.scaler = None
        self.scaled_rmse = None
        self.results_with_test_x = None
        self.rmse = None
        self.inv_y = None

    def validate_file(self):
        # specify the number of lag hours
        self.values = self.dataset.values
        self.number_features = self.values.shape[1]
        encoder = LabelEncoder()
        self.values[:, 4] = encoder.fit_transform(self.values[:, 4])
        self.values = self.values.astype('float32')

    def scale_dataset(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.values = self.scaler.fit_transform(self.values)

    def split_train_test(self):
        n_train_points = int(self.training_split * self.values.shape[0])
        train = self.values[:n_train_points, :]
        test = self.values[n_train_points:, :]

        # split into input and outputs
        n_obs = self.look_back_period * self.number_features
        self.train_X, self.train_y = train[:, :n_obs], train[:, -self.number_features]
        self.test_X, self.test_y = test[:, :n_obs], test[:, -self.number_features]
        print("Total number of rows: {}\nRows used for Training: {}\nRows used for testing: {}\n"
              "Number of variables used: {} ".format(self.values.shape[0], self.train_X.shape[0],
                                                     self.test_X.shape[0], self.train_X.shape[1]))

    def create_train_generator(self):
        self.generator = TimeseriesGenerator(self.train_X, self.train_y, self.look_back_period,
                                             batch_size=self.batch_size)

    def create_network(self):
        layers = [
            LSTM(100, activation=self.lstm_activation, input_shape=(self.look_back_period, self.train_X.shape[1])),
            Dropout(0.1),
            Dense(1, activation='relu')
        ]

        self.model = Sequential(layers)

        self.model.compile(loss=self.loss_function, optimizer=self.optimizer)
        self.history = self.model.fit_generator(generator=self.generator, epochs=self.num_epochs, verbose=2,
                                                steps_per_epoch=self.steps_per_epoch)

    def create_test_generator(self):
        self.test_generator = TimeseriesGenerator(self.test_X, self.test_y, self.look_back_period,
                                                  batch_size=self.batch_size)

        accuracy_results = self.model.evaluate_generator(self.test_generator)
        print("Results: {}".format(accuracy_results))

        self.results = self.model.predict_generator(self.test_generator)

    def get_summary_values(self):
        self.scaled_rmse = sqrt(
            mean_squared_error(self.results, self.test_y.reshape((len(self.test_y), 1))[-len(self.results):]))
        print('Test RMSE minmax scaled: %.3f' % self.scaled_rmse)

        # invert scaling for forecast
        self.results_with_test_x = concatenate((self.results, self.test_X[-len(self.results):, -self.number_features + 1:]),
                                               axis=1)
        self.results_with_test_x = self.scaler.inverse_transform(self.results_with_test_x)
        self.results_with_test_x = self.results_with_test_x[:, 0]

        # invert scaling for actual
        test_y = self.test_y.reshape((len(self.test_y), 1))
        self.inv_y = concatenate((test_y, self.test_X[:, -self.number_features + 1:]), axis=1)
        self.inv_y = self.scaler.inverse_transform(self.inv_y)
        self.inv_y = self.inv_y[-len(self.results):, 0]

        # calculate RMSE
        self.rmse = sqrt(mean_squared_error(self.inv_y, self.results_with_test_x))
        print('Test RMSE original scale: %.3f' % self.rmse)

    def plot_initial_graphs(self):
        groups = [j for j in range(0, self.number_features)]
        i = 1
        # plot each column
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
        pyplot.title('model accuracy')
        pyplot.ylabel('accuracy')
        pyplot.xlabel('epoch')
        pyplot.legend(['train', 'test'], loc='upper left')
        pyplot.show()

    def plot_prediction(self):
        pyplot.plot(self.results_with_test_x, label='Prediction')
        pyplot.plot(self.inv_y, label='True')
        pyplot.xlabel("Time units (months)")
        pyplot.ylabel(self.dataset.keys()[0])
        pyplot.title("{} prediction vs real data".format(self.dataset.keys()[0]))
        pyplot.legend()
        pyplot.show()

    def start_prediction(self):
        self.validate_file()
        if self.display_initial_graphs:
            self.plot_initial_graphs()
        self.scale_dataset()
        self.split_train_test()
        self.create_train_generator()
        self.create_network()
        self.create_test_generator()
        self.get_summary_values()
        if self.display_loss_graphs:
            self.plot_loss_values()
        if self.display_prediction_graph:
            self.plot_prediction()
        return self.rmse


final_result = LSTMPrediction(
    dataset=read_csv('../data/full_data_1981_onwards_no_nan.csv', header=0, index_col=0),
    col_to_predict=int(os.environ.get('COLUMN_TO_PREDICT')),
    cols_to_use_for_prediction=[int(x) for x in os.environ.get('COLUMNS_FOR_PREDICTION').split(',')],
    look_back_period=int(os.environ.get('LOOK_BACK_PERIOD')),
    training_split=float(os.environ.get('TRAINING_SPLIT')),
    loss_function=str(os.environ.get('LOSS_FUNCTION')),
    optimizer=str(os.environ.get('OPTIMIZER')),
    num_epochs=int(os.environ.get('NUM_EPOCHS')),
    steps_per_epoch=int(os.environ.get('STEPS_PER_EPOCH')),
    batch_size=int(os.environ.get('BATCH_SIZE')),
    lstm_activation=str(os.environ.get('LSTM_ACTIVATION')),
    lstm_units=int(os.environ.get('LSTM_UNITS')),
    display_initial_graphs=False,
    display_loss_graphs=False,
    display_prediction_graph=True
).start_prediction()
print(final_result)
