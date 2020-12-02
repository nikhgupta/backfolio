from statsmodels.tsa.stattools import adfuller


def adf(src):
    result = adfuller(src.ffill().dropna().values)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
