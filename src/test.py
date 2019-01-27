import numpy
import time
from pandas import read_csv
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler


def create_dataset(dataset, look_back, predicting_col):
    shift = 1
    dataY = dataset[predicting_col].shift(shift)
    for col in dataset:
        for i in range(1, look_back+1):
            dataset["{}-{}".format(dataset[col].name, i)] = dataset[col].shift(-i)
        dataset.to_csv('../data/uk_small_updated.csv')
    dt = dataset.drop(columns=[predicting_col])
    return numpy.array(dt)[:-look_back], numpy.array(dataY)[shift:-look_back+1]


dataframe = read_csv('../data/uk_data_small.csv', usecols=[1, 2, 3, 4, 5, 6, 7], engine='python')
trainX, trainY = create_dataset(dataframe, 3, 'Unemployment')
print(trainX)
print(trainY)



#
# convert an array of values into a dataset matrix
# def create_dataset(dataset, look_back=1):
#     dataX, dataY = [], []
#     for i in range(len(dataset) - look_back - 1):
#         a = dataset[i:(i + look_back), 0]
#         dataX.append(a)
#         dataY.append(dataset[i + look_back, 0])
#     return numpy.array(dataX), numpy.array(dataY)
#
#
# # fix random seed for reproducibility
# numpy.random.seed(7)
# # load the dataset
# dataframe = read_csv('../data/uk_data.csv', usecols=[1, 2, 3, 4, 5], engine='python')
# data_label = dataframe.keys()[0]
# dataset = dataframe.values
# dataset = dataset.astype('float32')
# # normalize the dataset
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)
# # reshape into X=t and Y=t+1
# look_back = 3
# trainX, trainY = create_dataset(dataset, look_back)
# print(list(trainX))
# print(list(trainY))
