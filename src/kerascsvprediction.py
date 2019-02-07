import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import logging
import os
from pandas import read_csv, DataFrame, concat, Series
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM

# Settings from environment variables
col_to_predict = str(os.environ.get('COLUMN_TO_PREDICT'))
cols_to_use_for_prediction = [int(x) for x in os.environ.get('COLUMNS_FOR_PREDICTION').split(',')]
layer_setup = [int(x) for x in os.environ.get('LAYER_SETUP').split(',')]
look_back_period = int(os.environ.get('LOOK_BACK_PERIOD'))
training_split = float(os.environ.get('TRAINING_SPLIT'))
loss_function = str(os.environ.get('LOSS_FUNCTION'))
optimizer = str(os.environ.get('OPTIMIZER'))
num_epochs = int(os.environ.get('NUM_EPOCHS'))
batch_size = int(os.environ.get('BATCH_SIZE'))
sequence_length = int(os.environ.get('SEQUENCE_LENGTH'))
prediction_length = int(os.environ.get('PREDICTION_LENGTH'))


def series_to_supervised(data):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(look_back_period, 0, -1):
        cols.append(df.shift(-i))
        print(df.columns.values)
        names += [('{}(t-{})'.format(df.columns.values[j], i)) for j in range(n_vars)]
    cols.append(df[col_to_predict])
    names.append(col_to_predict)
    agg = concat(cols, axis=1)
    agg.columns = names
    agg.dropna(inplace=True)
    return agg


df = read_csv('../data/uk_data.csv', usecols=cols_to_use_for_prediction, engine='python')
df = series_to_supervised(df)
logging.info(df.shape)
logging.info(df.shape)
df.head(3)

x_data = DataFrame(df).iloc[:, :df.shape[1] - 1]
logging.info(x_data.shape)

y_data = DataFrame(df).iloc[:, -1:]
logging.info(y_data.shape)

logging.info(x_data.head(1))
logging.info(y_data.head(1))

num_data = len(x_data)

num_train = int(training_split * num_data)
logging.info(num_train)

num_test = num_data - num_train
logging.info(num_test)

x_train = x_data[0:num_train]
x_test = x_data[num_train:]
logging.info(len(x_train) + len(x_test))

y_train = y_data[0:num_train]
y_test = y_data[num_train:]
len(y_train) + len(y_test)

num_x_signals = x_data.shape[1]
logging.info(num_x_signals)

num_y_signals = y_data.shape[1]
logging.info(num_y_signals)

logging.info("Min:", np.min(x_train))
logging.info("Max:", np.max(x_train))

x_scaler = MinMaxScaler()

x_train_scaled = x_scaler.fit_transform(x_train)

logging.info("Min:", np.min(x_train_scaled))
logging.info("Max:", np.max(x_train_scaled))

x_test_scaled = x_scaler.transform(x_test)

y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

print(x_train_scaled.shape)
logging.info(y_train_scaled.shape)


def batch_generator(batch_size, sequence_length):
    """
    Generator function for creating random batches of training-data.
    """
    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(num_train - sequence_length)

            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train_scaled[idx:idx + sequence_length]
            y_batch[i] = y_train_scaled[idx:idx + sequence_length]

        yield (x_batch, y_batch)

generator = batch_generator(batch_size=batch_size,
                            sequence_length=sequence_length)

x_batch, y_batch = next(generator)

logging.info(x_batch.shape)
logging.info(y_batch.shape)

validation_data = (np.expand_dims(x_test_scaled, axis=0),
                   np.expand_dims(y_test_scaled, axis=0))

model = Sequential()
model.add(LSTM(units=batch_size,
               return_sequences=True,
               input_shape=(None, num_x_signals,)))
model.add(Dense(num_y_signals, activation='softmax'))
# model.add(Dense(num_y_signals, activation='tanh'))
model.compile(loss=loss_function, optimizer=optimizer)
model.summary()

model.fit_generator(generator=generator,
                    epochs=num_epochs,
                    steps_per_epoch=int(num_x_signals/batch_size),
                    validation_data=validation_data)

result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
                        y=np.expand_dims(y_test_scaled, axis=0))


def plot_comparison(target_names, length=100):
    """
    Plot the predicted and true output-signals.
    
    :param target_names: Name of column you are predicting
    :param length: Sequence-length to process and plot.
    """
    # Use test-data.
    x = x_test_scaled
    y_true = y_test

    # Select the sequences from the given start-index and
    # of the given length.
    x = x[-length:]
    y_true = y_true[-length:]

    # Input-signals for the model.
    x = np.expand_dims(x, axis=0)

    # Use the model to predict the output-signals.
    y_pred = model.predict(x)

    # The output of the model is between 0 and 1.
    # Do an inverse map to get it back to the scale
    # of the original data-set.
    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])

    # For each output-signal.
    logging.info("Predicting {}".format(target_names))
    # Get the output-signal predicted by the model.
    signal_pred = y_pred_rescaled[:, 0]

    # Get the true output-signal from the data-set.
    logging.info(y_true.shape)
    signal_true = DataFrame(data=y_true).iloc[:, 0]
    signal_true = list(signal_true.items())
    # Make the plotting-canvas bigger.
    # plt.figure(figsize=(15, 5))

    signal_true = Series([ind[1] for ind in signal_true], index=list(range(len(signal_true))))

    index_list_y_data = list(range(len(y_data)))
    index_list_y_data_minus_signal_true = list(range(len(y_data)-len(signal_true), len(y_data)))
    index_list_y_data_minus_signal_pred = list(range(len(y_data) - len(signal_pred), len(y_data)))

    plt.plot(index_list_y_data, y_data)
    plt.plot(index_list_y_data_minus_signal_true, signal_true)
    plt.plot(index_list_y_data_minus_signal_pred, signal_pred)
    # plt.plot(l, label='true')
    # plt.plot(signal_pred, label='pred')

    # Plot grey box for warmup-period.
    plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)

    rmse = sqrt(mean_squared_error(y_pred_rescaled, y_true))
    print('Test RMSE: %.3f' % rmse)
    # Plot labels etc.
    # plt.ylabel(target_names)
    # plt.xlabel('Time units')
    # plt.legend()
    # plt.title(target_names)
    plt.show()


warmup_steps = 7
plot_comparison(target_names=col_to_predict, length=prediction_length)
