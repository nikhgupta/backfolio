import requests
import numpy as np
import pandas as pd
from copy import deepcopy
from abc import ABCMeta, abstractmethod
from os import environ
from os.path import join, isfile
from .core.utils import make_path
from .indicator import rebase
import json, hashlib


class BaseBenchmark(object):
    """
    BaseBenchmark is a base class that describes the interface required for all
    benchmark objects.

    A benchmark is used to compare the results of a strategy against returns
    from it. Data for the benchmark is available to the strategy via `BM.daily`
    or `BM.timeline` properties, and can be used to formulate strategies, as
    well.

    A benchmark object must implement `_returns_data()` method, wherein you
    must return a Pandas dataframe with a DateTimeIndex and a `returns` field
    containing decimal returns for the benchmark per tick.

    Subsequent calls to the same benchmark are cached, if a `cache_name` is
    specified when initializing the benchmark object. This is overridden when
    `set_refresh_history(True)` is specified for trading session.
    """
    __metaclass__ = ABCMeta

    def __init__(self, name, cache_name=None):
        self.name = name
        self._cache_name = cache_name
        self._result = None
        self.context = None
        self.session_fields = ['data']

    def reset(self, context=None):
        self.context = context
        self._result = None
        if context:
            self.data_dir = context.root_dir
        else:
            self.data_dir = join(environ['HOME'], '.backfolio')
        return self

    @property
    def data_dir(self):
        return self._data_dir

    @data_dir.setter
    def data_dir(self, val):
        self._data_dir = join(val, "benchmarks")
        make_path(self.data_dir)

        self._cache = join(self.data_dir, "%s.csv" % self._cache_name)
        if self._cache_name is None:
            self._cache = None

    @property
    def account(self):
        return self.context.account

    @property
    def broker(self):
        return self.context.broker

    @property
    def strategy(self):
        return self.context.strategy

    @property
    def portfolio(self):
        return self.context.portfolio

    @property
    def reporters(self):
        return self.context.reporters

    @property
    def benchmarks(self):
        return self.context.benchmarks

    @property
    def datacenter(self):
        return self.context.datacenter

    @property
    def events(self):
        return self.context.events

    @property
    def data(self):
        return self.timeline

    @property
    def timeline(self):
        if self.context is None:
            self.reset(None)
        data = self._fetch_data()
        return self._sanitize_benchmark_data(data, group_daily=False)

    @property
    def daily(self):
        if self.context is None:
            self.reset(None)
        data = self._fetch_data()
        return self._sanitize_benchmark_data(data, group_daily=True)

    def _sanitize_benchmark_data(self, data, group_daily=False):
        data.index = pd.to_datetime(data.index)
        data.index.name = 'time'
        if group_daily:
            data = data.groupby(pd.Grouper(freq='D')).last()

        if 'open' in data:
            data['returns'] = data['open'] / data['open'].shift(1) - 1
        if 'equity' in data:
            data['returns'] = data['equity'] / data['equity'].shift(1) - 1

        data['returns'] = data['returns'].fillna(0)
        data['cum_returns'] = (1 + data['returns']).cumprod()

        if self.context is not None:
            if self.context.start_time:
                data = data[self.context.start_time:]
            if self.context.end_time:
                data = data[:self.context.end_time]

        if len(data['cum_returns']) > 0:
            data['cum_returns'] = (data['cum_returns'] /
                                   data['cum_returns'].iloc[0])
        return data

    def _fetch_data(self):
        if self._result is None:
            if (self._cache and isfile(self._cache) and
                (not self.context or not self.context.refresh_history)):
                data = pd.read_csv(self._cache, index_col=0)
            else:
                data = self._returns_data()
                data.to_csv(self._cache, index=True)
            self._result = data
            if self.context:
                data = self.datacenter.history
                start_time = data.major_axis[0].strftime("%Y-%m-%d %H:%M")
                end_time = data.major_axis[-1].strftime("%Y-%m-%d %H:%M")
                self._result = self._result[start_time:end_time]
        return self._result

    @abstractmethod
    def _returns_data(self):
        raise NotImplementedError("Benchmark must implement `_returns_data()`")


class SymbolAsBenchmark(BaseBenchmark):
    """
    Use a single symbol as a benchmark.

    Effectively, we want to compare with buy and hold of a single coin.
    """
    def __init__(self, symbol='BTC/USDT', cache_name=None):
        if not cache_name:
            cache_name = symbol.replace("/", "_")
        super().__init__(symbol, cache_name)
        self.symbol = symbol

    def _returns_data(self):
        return self.datacenter.refresh_history_for_symbol(self.symbol)


class PortfolioAsBenchmark(BaseBenchmark):
    """
    Use a list of symbols as a benchmark.

    Effectively, we want to compare with buy and hold of multiple coins.

    Weights of individual coins can be specified, e.g. a `mapping` with:

        `{"BTC/USDT": 1, "BNB/USDT": 2}`

    means that the portfolio has 67% equity in BNB and 33% equity in BTC.
    """
    def __init__(self, mapping={}, cache_name=None):
        if not cache_name:
            j = json.dumps(mapping, sort_keys=True).encode('utf-8')
            cache_name = 'portfolio-%s' % hashlib.md5(j).hexdigest()[:12]
        super().__init__(cache_name, cache_name)
        self.mapping = mapping

    def _returns_data(self):
        syms = list(self.mapping.keys())
        if len(syms) > 0:
            ch = self.datacenter.history[syms, :, 'close'].pct_change()
        else:
            ch = self.datacenter.history[:, :, 'close'].pct_change()
        if self.context.start_time:
            ch = ch[self.context.start_time:]
        if self.context.end_time:
            ch = ch[:self.context.end_time]

        pf, wt = 0, 0
        for symbol in ch.columns:
            val = 1 if symbol not in self.mapping else self.mapping[symbol]
            pf += val * (1 + ch[symbol]).cumprod()
            wt += val
        df = pd.DataFrame()
        df['equity'] = pf / wt
        return df


class StrategyAsBenchmark(BaseBenchmark):
    def __init__(self, strategy, cache_name=None):
        # TODO: Implement strategy as benchmark
        raise ValueError("StrategyAsBenchmark is currently not supported.")
        super().__init__(strategy.__class__.__name__, cache_name)
        self.strategy = deepcopy(strategy)

    def reset(self, context=None):
        super().reset(context)
        if not self.context.backtesting():
            raise ValueError("StrategyAsBenchmark is not supported in \
                             paper/live trading mode.")

    def _returns_data(self):
        current_strat = self.context.strategy
        self.context.set_strategy(self.strategy)
        self.context.run(report=False, main=False)
        res = self.context.portfolio.daily
        self.context.strategy = current_strat
        self.context.strategy.reset(self.context)
        return res


class CSVAsBenchmark(BaseBenchmark):
    """
    Use a saved portfolio for benchmark.

    You can save a backtest run as benchmark using:

        `pf.portfolio.save_as_benchmark(name)`
    """
    def __init__(self, file_name, name=None):
        if not name:
            name = file_name
        super().__init__(name, file_name)

    def _returns_data(self):
        if not isfile(self._cache):
            raise ValueError("No such benchmark: %s" % self._cache)
        else:
            return pd.read_csv(self._cache, index_col=0)


class CryptoMarketCapAsBenchmark(BaseBenchmark):
    """
    Use crypto market capital as a benchmark.

    Effectively, this is equivalent to buy and hold of all crypto assets.
    You can, seletively, enable or disable BTC market capital.

    You can also set `volume` to true to experiment with volume based data.
    """
    def __init__(self, include_btc=False, volume=False, cache_name=None):
        self.volume = volume
        self.graph = '' if include_btc else '_altcoin'
        name = "Total" if include_btc else 'Alt'
        name += '24hVol' if self.volume else 'MCap'
        if not cache_name:
            cache_name = name
        super().__init__(name, cache_name)

    def _returns_data(self):
        if self.context:
            data = self.datacenter.history
            start_date = int(data.major_axis[0].timestamp()) * 1000
            end_date = int(data.major_axis[-1].timestamp()) * 1000
        else:
            start_date = int(pd.to_datetime("20170101").timestamp()) * 1000
            end_date = int(pd.to_datetime("now").timestamp()) * 1000

        url = 'https://web-api.coinmarketcap.com/v1.1/global-metrics/quotes/historical'
        url += '?format=chart%s&interval=1d&time_start=%d&time_end=%d'
        url = url % (self.graph, start_date, end_date)
        dic_t = requests.get(url).json()

        df = pd.DataFrame.from_dict(dic_t['data']).transpose()
        df.index = pd.to_datetime(df.index)
        df = df.reset_index()
        df.columns = ['time', 'market_cap', '24h_vol']

        df['time'] = pd.to_datetime(df['time'], unit='ms').dt.floor('1d')
        df = df.drop_duplicates(subset=['time'], keep='last').set_index('time')
        df = df.reset_index()
        index = 'index' if 'index' in df.columns else 'time'
        df = df.rename(columns={index: "time"}).set_index('time')
        df['price'] = np.log(
            df['24h_vol']) if self.volume else df['market_cap']
        df['returns'] = df['price'] / df['price'].shift() - 1

        return df


class BitcoinMarketCapAsBenchmark(BaseBenchmark):
    """
    Use Bitcoin market capital as a benchmark.

    Effectively, this is roughly equivalent to buy and hold of Bitcoin.

    You can also set `volume` to true to experiment with volume based data.
    """
    def __init__(self, volume=False, cache_name=None):
        self.volume = volume
        name = 'BTC24hVol' if self.volume else 'BTCMCap'
        if not cache_name:
            cache_name = name
        super().__init__(name, cache_name)

    def _returns_data(self):
        tot = CryptoMarketCapAsBenchmark(include_btc=True, volume=self.volume)
        alt = CryptoMarketCapAsBenchmark(include_btc=False, volume=self.volume)

        tot = tot.reset(self.context).timeline
        alt = alt.reset(self.context).timeline
        tot['market_cap'] -= alt['market_cap']
        tot['24h_vol'] -= alt['24h_vol']

        tot['price'] = np.log(
            tot['24h_vol']) if self.volume else tot['market_cap']
        tot['returns'] = tot['price'] / tot['price'].shift() - 1
        tot = tot.drop(columns=['cum_returns'])

        return tot
