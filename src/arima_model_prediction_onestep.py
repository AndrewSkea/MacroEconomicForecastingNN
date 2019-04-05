from pandas import read_csv
from tkinter import *
from tkinter.ttk import *
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from math import sqrt


class ARIMAModel:
    def __init__(self, series, col_to_predict, train_test_split=0.85, validation_size=0.15, time_lags=5):
        self.series = series
        self.train_test_split = train_test_split
        self.col_to_predict = col_to_predict
        self.validation_split = validation_size
        self.time_lags = time_lags
        self.train = None
        self.test = None
        self.predictions = list()
        self.test_predictions = list()
        self.history = None
        self.rmse = 0

    def split_data(self):
        train_length = int(self.train_test_split * len(self.series))
        self.train = self.series[0:-36]
        self.test = self.series[-36:]
        print("Total length: {}\nTrain length: {}\nTest total: {}".format(
            len(self.series), len(self.train), len(self.test)
        ))

    def train_and_predict(self):
        self.history = list(self.train)
        true_future_values = []
        self.predictions = list()
        test_length = len(self.test)

        # walk-forward validation
        for t in range(test_length):
            # fit model
            model = ARIMA(self.history, order=(self.time_lags, 2, 1))
            model_fit = model.fit(disp=False)
            yhat = model_fit.forecast()[0]
            # store forecast and ob
            self.predictions.append(yhat)
            self.history.append(self.test[t])
            print("{}/{}".format(t, test_length))

        self.rmse = sqrt(mean_squared_error(self.predictions, self.test))

    def summarise(self):
        # evaluate forecasts
        print('Test RMSE: %.3f' % self.rmse)

        full_x_values = list(range(len(self.series)))
        pyplot.plot(full_x_values, self.series, label='values')

        pyplot.plot(full_x_values[-len(self.predictions)-1:-1], self.predictions, 'g', label='t+1 line')

        pyplot.xlabel("Time units (months)")
        pyplot.ylabel(self.col_to_predict)
        pyplot.title("{} prediction vs real data".format(self.col_to_predict))
        pyplot.legend()
        pyplot.show()

    def start(self):
        self.split_data()
        self.train_and_predict()
        self.summarise()
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

    btn = Button(window, text="Choose", command=clicked)
    for i in range(len(rad_array)):
        rad_array[i].grid(column=0, row=i)

    btn.grid(column=1, row=len(rad_array))
    window.mainloop()
    return column_to_predict


if __name__ == "__main__":
    global column_to_predict
    column_to_predict = 0
    input_dataset = read_csv('../data/FinalDataset.csv', header=0, index_col=0)
    dataset_key_list = list(input_dataset.keys())
    #dataset_key_list.remove('Month')
    get_column_to_predict(dataset_key_list)
    print("Column to predict: ", column_to_predict)

    data_series = list(input_dataset[column_to_predict].values)
    rmse_result = ARIMAModel(data_series, train_test_split=0.8, col_to_predict=column_to_predict).start()
