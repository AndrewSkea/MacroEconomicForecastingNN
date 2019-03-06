from statsmodels.tsa.stattools import adfuller
from numpy import log
from matplotlib import pyplot as plt


class ADFullerTest:
    def __init__(self, series, do_log=False):
        self.series = series
        self.do_log = do_log

    def adfuller_test(self):
        # plt.plot(list(range(len(self.series))), self.series)
        # plt.show()
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