from pandas import read_csv
from tkinter import *
from tkinter.ttk import *
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from math import sqrt


class ARIMAModel:
    def __init__(self, series, col_to_predict, train_test_split=0.7, validation_size=0.11, time_lags=3):
        self.series = series
        self.train_test_split = train_test_split
        self.col_to_predict = col_to_predict
        self.validation_split = validation_size
        self.time_lags = time_lags
        self.train = None
        self.test = None
        self.validation = None
        self.predictions = list()
        self.validation_predictions = list()
        self.history = None
        self.rmse = 0

    def split_data(self):
        validation_length = int(len(self.series) * self.validation_split)
        train_length = int(self.train_test_split * (len(self.series) - validation_length))
        self.train = self.series[0:train_length]
        self.test = self.series[train_length:-validation_length]
        self.validation = self.series[-validation_length:]
        print("Total length: {}\nTrain length: {}\nTest total: {}\nValidation length: {}".format(
            len(self.series), len(self.train), len(self.test), len(self.validation)
        ))
        self.history = [x for x in self.train]

    def train_and_predict(self):
        self.predictions = list()
        test_length = len(self.test)
        # walk-forward validation
        for t in range(test_length):
            # fit model
            model = ARIMA(self.history, order=(self.time_lags, 1, 0))
            model_fit = model.fit(disp=False)
            # one step forecast
            yhat = model_fit.forecast()[0]
            # store forecast and ob
            self.predictions.append(yhat)
            self.history.append(self.test[t])
            print("{}/{}".format(t, test_length))

        # for i in range(len(self.validation)):
        self.validation_predictions = model_fit.forecast(len(self.validation))[0]

    def summarise(self):
        # evaluate forecasts
        self.rmse = sqrt(mean_squared_error(self.test, self.predictions))
        print('Test RMSE: %.3f' % self.rmse)

        full_x_values = list(range(len(self.series)))
        pred_x_values = list(range(len(self.train), len(self.train)+len(self.test)))
        validation_pred_x_values = list(range(int(len(self.series) - len(self.validation)), len(self.series)))

        pyplot.plot(full_x_values, self.series, label='True')
        pyplot.plot(pred_x_values, self.predictions, label='Test predictions (one-step ahead)')
        pyplot.plot(validation_pred_x_values, self.validation_predictions, label='Test predictions (out-of-sample)')

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
    input_dataset = read_csv('../data/final_data.csv', header=0, index_col=0)
    dataset_key_list = list(input_dataset.keys())
    # dataset_key_list.remove('Month')
    get_column_to_predict(dataset_key_list)
    print("Column to predict: ", column_to_predict)

    data_series = list(input_dataset[column_to_predict].values)
    # data_series = data_series[int(len(data_series)*0.7):]
    rmse_result = ARIMAModel(data_series, train_test_split=0.9, col_to_predict=column_to_predict).start()
