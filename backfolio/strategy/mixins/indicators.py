import numpy as np
import talib as ta
import pandas as pd
from ...functions import averaged_indicator, alpharank


class IndicatorsMixin(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._indicator_periods = [2,3,5,8,13,21,34,55,89,144] # default
        self._indicator_cache = {}

    def list_available_indicators(self):
        return [f for f in dir(self) if '__indicator' in f]

    def set_indicator_periods(self, *args):
        self._indicator_periods = args

    def reset_indicator_cache(self):
        self._indicator_cache = {}

    def __get_averages(self, src):
        data = {}
        data[1] = src
        for d in self._indicator_periods:
            data[d] = src.rolling(d).mean()
        self._indicator_cache['averages'] = pd.Panel(data)
        return self._indicator_cache['averages']

    def indicator__volatility(self, src):
        averages = self.__get_averages(src)
        return averages.max(axis=0)/averages.min(axis=0)*100 - 100

    def indicator__trend(self, src):
        trend = 0
        averages = self.__get_averages(src)
        for idx, d in enumerate(self._indicator_periods):
            trend += averages[averages.axes[0] < d].sum(axis=0)/averages[d] - idx - 1
        return trend / len(self._indicator_periods) * 100

    def indicator__drop(self, src, pr=-1, **kwargs):
        return averaged_indicator(src,
            func=lambda x,p: -x.pct_change(p), pr=pr, **kwargs)

    def indicator__revertness(self, src, pr=-1, **kwargs):
        return averaged_indicator(src,
            func=lambda x,p: x.rolling(p).mean()/x-1, pr=pr, **kwargs)

    def indicator__dca_loss(self, src, pr=-1, **kwargs):
        return averaged_indicator(src,
            func=lambda x,p: -x*(1/x).rolling(p).sum(), pr=pr, **kwargs)
