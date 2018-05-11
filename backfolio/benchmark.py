import requests
import pandas as pd
from copy import deepcopy
from abc import ABCMeta, abstractmethod
from os.path import join, isfile
from .core.utils import make_path


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
        self.session_fields = ['data']

    def reset(self, context=None):
        self.context = context
        self._result = None
        self.data_dir = join(context.root_dir, "benchmarks")
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
        return self._fetch_data(group_daily=False)

    @property
    def daily(self):
        return self._fetch_data(group_daily=True)

    def _sanitize_benchmark_data(self, data, group_daily=True):
        data.index = pd.to_datetime(data.index)
        data.index.name = 'time'
        if group_daily:
            data = data.groupby(pd.Grouper(freq='D')).first()

        if 'returns' not in data:
            data['returns'] = data['open']/data['open'].shift(1) - 1
        data['returns'] = data['returns'].fillna(0)
        if 'cum_returns' not in data:
            data['cum_returns'] = (1+data['returns']).cumprod()
        if self.context.start_time:
            data = data[self.context.start_time:]
        if self.context.end_time:
            data = data[:self.context.end_time]
        data['cum_returns'] = data['cum_returns']/data['cum_returns'].iloc[0]
        return data

    def _fetch_data(self, group_daily=False):
        if self._result is None:
            if (self._cache and isfile(self._cache) and
                    not self.context.refresh_history):
                data = pd.read_csv(self._cache, index_col=0)
            else:
                data = self._returns_data()
                data.to_csv(self._cache, index=True)
            self._result = self._sanitize_benchmark_data(data, group_daily)
        return self._result

    @abstractmethod
    def _returns_data(self):
        raise NotImplementedError("Benchmark must implement `_returns_data()`")


class SymbolAsBenchmark(BaseBenchmark):
    def __init__(self, symbol='BTC/USDT', cache_name=None):
        if not cache_name:
            cache_name = symbol.replace("/", "_")
        super().__init__(symbol, cache_name)
        self.symbol = symbol

    def _returns_data(self):
        return self.datacenter.refresh_history_for_symbol(self.symbol)


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
    pass


class CryptoMarketCapAsBenchmark(BaseBenchmark):
    def __init__(self, include_btc=False, cache_name=None):
        self.graph = 'total' if include_btc else 'altcoin'
        name = "TotalMCap" if include_btc else 'AltMCap'
        if not cache_name:
            cache_name = name
        super().__init__(name, cache_name)

    def _returns_data(self):
        data = self.datacenter.history
        start_date = int(data.major_axis[0].timestamp())*1000
        end_date = int(data.major_axis[-1].timestamp())*1000
        url = 'https://graphs2.coinmarketcap.com/global/marketcap-%s/%d/%d'
        dic_t = requests.get(url % (self.graph, start_date, end_date)).json()
        dic_t = dic_t['market_cap_by_available_supply']
        df = pd.DataFrame.from_dict(dic_t)
        df.columns = ["time", "price"]

        df['time'] = pd.to_datetime(df['time'], unit='ms').dt.ceil('1d')
        df = df.drop_duplicates(subset=['time'], keep='last').set_index('time')
        df = df.reset_index()
        index = 'index' if 'index' in df.columns else 'time'
        df = df.rename(columns={index: "time"}).set_index('time')

        df['returns'] = df['price']/df['price'].shift(1) - 1
        return df
