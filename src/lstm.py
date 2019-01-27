import numpy
import os
import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

# Settings from environment variables
col_to_predict = int(os.environ.get('COLUMN_TO_PREDICT'))
cols_to_use_for_prediction = [int(x) for x in os.environ.get('COLUMNS_FOR_PREDICTION').split(',')]
layer_setup = [int(x) for x in os.environ.get('LAYER_SETUP').split(',')]
look_back_period = int(os.environ.get('LOOK_BACK_PERIOD'))
training_split = float(os.environ.get('TRAINING_SPLIT'))
loss_function = str(os.environ.get('LOSS_FUNCTION'))
optimizer = str(os.environ.get('OPTIMIZER'))
num_epochs = int(os.environ.get('NUM_EPOCHS'))
batch_size = int(os.environ.get('BATCH_SIZE'))

column_dict = {1: 'GDP',
               2: 'Unemployment',
               3: 'CPI',
               4: 'EconomicallyInactive',
               5: 'NetInvest',
               6: 'RPI',
               7: 'Manufacturing'}


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back, predicting_col):
    shift = 1
    dataY = dataset[predicting_col].shift(shift)
    for col in dataset:
        for i in range(1, look_back + 1):
            dataset["{}-{}".format(dataset[col].name, i)] = dataset[col].shift(-i)
    # dataset.to_csv('../data/uk_small_updated.csv')
    dt = dataset.drop(columns=[predicting_col])
    return numpy.array(dt)[:-look_back], numpy.array(dataY)[shift:-look_back + 1]

# fix random seed for reproducibility
numpy.random.seed(7)

# Read data from csv
dataframe = read_csv('../data/uk_data.csv', usecols=cols_to_use_for_prediction, engine='python')
dataset = preprocessing.scale(dataframe)
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * training_split)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

trainX, trainY = create_dataset(DataFrame(train), look_back_period, col_to_predict)
testX, testY = create_dataset(DataFrame(test), look_back_period, col_to_predict)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(trainX.shape[1], 1)))
model.add(Dense(1))
model.compile(loss=loss_function, optimizer=optimizer)
model.fit(trainX, trainY, epochs=num_epochs, batch_size=batch_size, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back_period:len(trainPredict) + look_back_period, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict) + (look_back_period * 2) + 1:len(dataset) - 1, :] = testPredict
# plot baseline and predictions
dt = plt.plot(scaler.inverse_transform(dataset), label='Dataset')
trd = plt.plot(trainPredictPlot, label='Training Data')
tstd = plt.plot(testPredictPlot, label='Testing Data')
plt.title('Prediction of {}'.format(column_dict[col_to_predict]))
plt.ylabel('{} (Scaled)'.format(column_dict[col_to_predict]))
plt.xlabel('Time Units (1 month)')
plt.legend(bbox_to_anchor=(1, 1), loc=6, borderaxespad=0.)
plt.show()
