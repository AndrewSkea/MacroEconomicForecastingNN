from pandas import read_csv
from tkinter import *
from tkinter.ttk import *
from statsmodels.tsa.stattools import adfuller
from numpy import log
from matplotlib import pyplot as plt


class ADFullerTest:
    def __init__(self, series, do_log=False):
        self.series = series
        self.do_log = do_log

    def adfuller_test(self):
        if self.do_log:
            self.series = log(self.series)
        plt.plot(self.series)
        plt.show()
        result = adfuller(self.series)
        adf_value = result[0]
        print('ADF Statistic: %f' % adf_value)
        print('p-value: %f' % result[1])
        print('Critical Values:')
        pass_stationary = False
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
            if adf_value < value:
                pass_stationary = True
        return pass_stationary


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
    input_dataset = read_csv('../data/full_data_1981_onwards_no_nan.csv', header=0, index_col=0)
    dataset_key_list = list(input_dataset.keys())
    dataset_key_list.remove('Month')
    get_column_to_predict(dataset_key_list)
    print("Column to predict: ", column_to_predict)

    data_series = list(input_dataset[column_to_predict].values)
    is_stationary = ADFullerTest(data_series, do_log=True).adfuller_test()
    print(is_stationary)
