"""
Various implementations for obtaining historical as well as live data from
an exchange/datacenter.
"""

import os
import glob
import ccxt
import json
import pyprind
import requests
import numpy as np
import pandas as pd

try:
    import quandl
except ModuleNotFoundError:
    print("QuandlDatacenter will not work. Library missing!")

try:
    from nsetools import Nse
    from yahoofinancials import YahooFinancials
except:
    print("NseDatacenter will not work. Library missing!")

from zipfile import ZipFile
from abc import ABCMeta, abstractmethod
from datetime import datetime
from os.path import join, basename, isfile

from .core.object import Tick
from .core.utils import make_path
from .core.event import TickUpdateEvent


MAXINT = 2**32 - 1


class BaseDatacenter(object):
    """
    BaseDatacenter is an abstract base class providing an interface for
    all subsequent (inherited) data centers (both live and historical).

    The goal of a (derived) BaseDatacenter object is to output a generated
    dataframe containing set of bars (OLHCV) for each asset requested.

    Datacenters generate a new TickUpdateEvent upon every heartbeat of the
    system, i.e. on each tick of the underlying exchange.

    This will replicate how a live strategy would function as current
    market data would be sent "down the pipe". Thus a historical and live
    system will be treated identically by the rest of the backtesting suite.
    """

    __metaclass__ = ABCMeta

    def __init__(self, timeframe='1d', fill=True, realign=True, resample=None):
        self._name = 'base_datacenter'
        self._selected_symbols = []
        self.timeframe = timeframe
        self.session_fields = []
        self.fill = fill
        self.resample = resample
        self.realign = realign

    def reset(self, context=None, root_dir=None):
        """ Routine to run when trading session is resetted. """
        self.context = context
        if hasattr(context, "root_dir") and not root_dir:
            root_dir = context.root_dir

        if not root_dir:
            raise ValueError("You must specify `root_dir` for `reset()`.")

        self._data_dir = join(root_dir, "data", self.name)
        make_path(self._data_dir)

        self._data_seen = []
        self._all_data = None
        self._generator = None
        self._current_real = None
        self._continue_backtest = True
        self.markets = self._selected_symbols.copy()
        if context:
            self.reload_history(refresh=self.context.refresh_history)
            if (not self.context.refresh_history and
                    self.history is None):
                raise ValueError("You must run with refresh=True")
        return self

    @property
    def refresh_history(self):
        return self.context.refresh_history if self.context else True

    @property
    def name(self):
        return self._name

    @property
    def data_dir(self):
        return self._data_dir

    @property
    def symbols(self):
        return self.markets

    @symbols.setter
    def symbols(self, arr=[]):
        self._selected_symbols = arr
        self.markets = arr

    @property
    def timeframe(self):
        return self._timeframe

    @timeframe.setter
    def timeframe(self, value):
        self._timeframe = value

    @property
    def timeframe_delta(self):
        return pd.to_timedelta(self.timeframe)

    @property
    def history(self):
        return self._all_data

    def replace_history(self, data):
        self._all_data = data

    @property
    def account(self):
        return self.context.account

    @property
    def broker(self):
        return self.context.broker

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
    def strategy(self):
        return self.context.strategy

    @property
    def events(self):
        return self.context.events

    @property
    def current_time(self):
        return self.context.current_time

    def log(self, message):
        if self.context:
            return self.context.log(message)
        else:
            print(message)

    @abstractmethod
    def assets_to_symbol(self, *_args):
        raise NotImplementedError('Datacenter should allow conversion from ' +
                                  'asset names to symbol.')

    @abstractmethod
    def symbol_to_assets(self, _symbol):
        raise NotImplementedError('Datacenter should allow conversion from ' +
                                  'symbol to assets.')

    @abstractmethod
    def all_symbols(self):
        """
        Returns a list of all symbols/assets supported by this datacenter.
        """
        raise NotImplementedError('Datacenter should allow fetching a list of \
                                   all symbols.')

    @abstractmethod
    def refresh_history_for_symbol(self, symbol, existing=None):
        """
        Reload/refresh history data for a particular symbol.
        """
        raise NotImplementedError('Datacenter should allow reloading/refreshing \
                                  historical data for a particular asset.')

    def _next_tick_generator(self):
        """
        Returns the snapshot of price data for all symbols at Nth tick,
        since the start of the backtest.
        """
        history = self.history
        if self.context.start_time:
            history = history[:, self.context.start_time:, :]
        if self.context.end_time:
            history = history[:, :self.context.end_time, :]

        self._prev_tick = None
        for ts in history.major_axis:
            self._prev_tick = self._current_real
            self._current_real = history.loc[:, ts, :].T.dropna(how='all')
            if self._prev_tick is not None:
                yield((ts, self._prev_tick))

    def recent_frames(self, n=1):
        """
        Returns the snapshot of price data for all symbols for the recent-most
        N ticks. We, specifically, avoid look-ahead bias here.
        """
        data = self._data_seen[-n:]
        data = pd.Panel(dict((tick.time, tick.history) for tick in data))
        data = data.transpose(1, 0, 2)
        return data

    def next_tick(self):
        """
        Pushes the latest tick data to visited history for backtesting mode.
        """
        if not self._generator:
            self._generator = self._next_tick_generator()

        try:
            time, data = next(self._generator)
        except StopIteration:
            self._continue_backtest = False
        else:
            tick = Tick(time, data)
            self._data_seen.append(tick)
            self.events.put(TickUpdateEvent(tick))
            return tick

    def last_tick(self):
        """
        Pushes the penultimate tick data to visited history for paper/live
        trading modes. Most cryptocurrency exchange provide a last incomplete
        tick for the current hour.
        """
        history = self.history
        self._prev_tick = history[:, history.major_axis[-2], :].T.dropna()
        self._current_real = history[:, history.major_axis[-1], :].T.dropna()

        time = history.major_axis[-1]
        tick = Tick(time, self._prev_tick)
        self._data_seen.append(tick)
        self.events.put(TickUpdateEvent(tick))
        return time

    def _sanitize_index(self, df):
        """ Ensure that the index of an asset's history is set to `date` """
        if df.index.name:
            df.index.name = 'time'
        elif 'date' in df.columns:
            df.set_index('time', inplace=True)
        elif 'time' in df.columns:
            df.set_index('time', inplace=True)
            df.index.name = 'time'
        return df

    def _sanitize_ohlcv(self, df):
        if df.empty:
            return df

        if self.realign:
            freq = self.timeframe.replace('m', 'T')
            df = df.groupby(pd.Grouper(freq=freq)).last()
        if self.fill:
            df['volume'] = df['volume'].fillna(0.0)
            df['close'] = df['close'].fillna(method='pad')
            df['open'] = df['open'].fillna(df['close'])
            df['low'] = df['low'].fillna(df['close'])
            df['high'] = df['high'].fillna(df['close'])
            if 'realclose' in df.columns:
                df['realclose'] = df['realclose'].fillna(method='pad')
        df['dividend'] = 0
        df['split'] = 1
        df['volume'] = np.where(df['volume'] > MAXINT, MAXINT, df['volume'])
        if self.resample:
            df = df.resample(self.resample).agg({
                "open": 'first', 'high': 'max',
                "low": "min", "close": "last",
                "volume": 'sum', "dividend": 'mean', "split": 'mean'
            })
        return df

    def _cleanup_and_save_symbol_data(self, symbol, df):
        df = self._sanitize_index(df)
        # save newly reloaded data to disk
        path = join(self.data_dir, "%s.csv" % symbol.replace("/", "_"))
        df.dropna().reset_index().to_csv(path, index=False)
        return df

    def reload_history(self, refresh=True):
        """ Reload/refresh history for all symbols from disk/exchange. """
        histories = {}
        freq = self.timeframe.replace('m', 'T')

        # load existing data from disk
        for filename in glob.iglob(join(self.data_dir, "*.csv")):
            data = pd.read_csv(filename, index_col=0)
            name = basename(filename)[:-4].replace("_", "/")
            if self._selected_symbols and name not in self._selected_symbols:
                continue

            if (self.to_sym is not None and self.to_sym.strip() and
                    name[-len(self.to_sym):] != self.to_sym):
                continue

            data = self._sanitize_index(data)
            data.index = pd.to_datetime(data.index).floor(freq)
            data = data[~data.index.duplicated(keep='last')]
            histories[name] = data.sort_index(ascending=1)
            if name not in self.markets:
                self.markets.append(name)

        # in case, we do not have data locally, fetch all symbols for next step
        if self.markets and not refresh:
            self.log("Loaded history data from local cache.")
        else:
            self.all_symbols()
            if not refresh:
                self.log("No history data found on this system.")

        # download/refresh data for symbols, if required
        bar = pyprind.ProgPercent(len(self.markets))
        for symbol in self.markets:
            bar.update(item_id="%12s - %4s" % (symbol, self.timeframe))

            has_data = histories and symbol in histories
            if has_data and not refresh:
                continue

            cdf = histories[symbol] if has_data else None
            to_time = pd.to_datetime(datetime.utcnow()).floor(freq)
            if cdf is not None and not cdf.empty and cdf.index[-1] >= to_time:
                continue

            try:
                cdf = self.refresh_history_for_symbol(symbol, cdf)
            except Exception as e:
                print(e)
                self.log("Encountered error when downloading data \
                                  for symbol %s:\n%s" % (symbol, str(e)))
                cdf = None

            if cdf is not None:
                histories[symbol] = cdf

        # finally, save the data so obtained as a panel for quick ref.
        self._all_data = {}
        for sym, df in histories.items():
            if symbol not in self.markets:
                continue
            self._all_data[sym] = self._sanitize_ohlcv(df)
        df = pd.Panel(self._all_data)
        if self.fill:
            df.loc[:, :, 'volume'] = df[:, :, 'volume'].fillna(0.0)
            df.loc[:, :, 'close'] = df[:, :, 'close'].fillna(method='pad')
            df.loc[:, :, 'open'] = df[:, :, 'open'].fillna(df[:, :, 'close'])
            df.loc[:, :, 'low'] = df[:, :, 'low'].fillna(df[:, :, 'close'])
            df.loc[:, :, 'high'] = df[:, :, 'high'].fillna(df[:, :, 'close'])
            if 'realclose' in df.axes[2]:
                df.loc[:, :, 'realclose'] = df[:, :,'realclose'].fillna(method='pad')

        self._all_data = df


class CryptocurrencyDatacenter(BaseDatacenter):
    def __init__(self, exchange, *args,
                 to_sym='BTC', limit=10000, params={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.to_sym = to_sym
        self.exchange = getattr(ccxt, exchange)(params)
        self.history_limit = limit
        self._market_data = {}

        if not self.exchange.has['fetchOHLCV']:
            raise NotImplementedError(
                'Exchange does not support fetching past market data.')

    @property
    def name(self):
        return "crypto/%s/%s" % (self.exchange.name, self.timeframe)

    def load_markets(self):
        if self.refresh_history:
            self._market_data = self.exchange.load_markets(True)
        elif self.history is not None:
            self._market_data = dict(
                [(k, {}) for k in self.history.axes[0]])
        else:
            raise ValueError("You must run with refresh=True to load markets")
        return self._market_data

    def all_symbols(self):
        """ Fetch all symbols supported by this exchange as a list """
        if self._selected_symbols:
            return self._selected_symbols

        self.load_markets()

        if self.to_sym is not None and self.to_sym.strip():
            self.markets = [key for key, _v in self._market_data.items()
                            if key[-len(self.to_sym):] == self.to_sym]
        else:
            if not self._market_data:
                self.load_markets()
            self.markets = list(self._market_data.keys())
        return self.markets

    def symbol_to_assets(self, symbol):
        return symbol.split("/")

    def assets_to_symbol(self, fsym, tsym=None):
        if tsym is None:
            tsym = self.context.base_currency
        return "%s/%s" % (fsym, tsym)

    def refresh_history_for_symbol(self, symbol, cdf=None):
        """ Refresh history for a given asset from exchange """
        plen = 0
        last_timestamp = None
        col_list = ['time', 'open', 'high', 'low', 'close', 'volume']
        if cdf is None:
            cdf = pd.DataFrame(columns=col_list).set_index('time')

        while True:
            data = self.exchange.fetch_ohlcv(
                symbol, timeframe=self.timeframe,
                since=last_timestamp, limit=self.history_limit)

            df = pd.DataFrame(data, columns=col_list)
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df = df.sort_values(by='time', ascending=1).set_index('time')
            cdf = cdf.append(df)
            cdf = cdf[~cdf.index.duplicated(keep='last')]
            cdf = cdf.sort_index(ascending=1)
            if (df.empty or len(cdf) == plen or self.history_limit is None or
                    len(cdf) >= self.history_limit or
                    (self.context and not self.context.backtesting())):
                break

            plen = len(cdf)
            last_timestamp = int((cdf.index[0] - pd.to_timedelta(
                 len(data) * self.timeframe)).timestamp())*1000

        return self._cleanup_and_save_symbol_data(symbol, cdf)


class CryptocompareDatacenter(BaseDatacenter):
    def __init__(self, exchange, *args,
                 to_sym='BTC', limit=10000, params={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.to_sym = to_sym
        self.exchange = getattr(ccxt, exchange)(params)
        self.history_limit = limit
        self._market_data = {}

    @property
    def name(self):
        return "crypto/%s/%s" % (self.exchange.name, self.timeframe)

    def load_markets(self):
        if self.refresh_history:
            self._market_data = self.exchange.load_markets(True)
        elif self.history is not None:
            self._market_data = dict(
                [(k, {}) for k in self.history.axes[0]])
        else:
            raise ValueError("You must run with refresh=True to load markets")
        return self._market_data

    def all_symbols(self):
        """ Fetch all symbols supported by this exchange as a list """
        if self._selected_symbols:
            return self._selected_symbols

        self.load_markets()

        if self.to_sym is not None and self.to_sym.strip():
            self.markets = [key for key, _v in self._market_data.items()
                            if key[-len(self.to_sym):] == self.to_sym]
        else:
            if not self._market_data:
                self.load_markets()
            self.markets = list(self._market_data.keys())
        return self.markets

    def symbol_to_assets(self, symbol):
        return symbol.split("/")

    def assets_to_symbol(self, fsym, tsym=None):
        if tsym is None:
            tsym = self.context.base_currency
        return "%s/%s" % (fsym, tsym)

    def refresh_history_for_symbol(self, symbol, cdf=None):
        """ Refresh history for a given asset from exchange """
        plen = 0
        last_timestamp = None
        col_list = ['time', 'open', 'high', 'low', 'close', 'volume']
        if cdf is None:
            cdf = pd.DataFrame(columns=col_list).set_index('time')

        while True:
            fsym, tsym = self.symbol_to_assets(symbol)
            endpoint = 'histohour' if self.timeframe == '1h' else 'histoday'
            url = "https://min-api.cryptocompare.com/data/%s?fsym=%s&tsym=%s&limit=2000&e=%s"
            url = url % (endpoint, fsym, tsym, self.exchange.name)
            if last_timestamp:
                url += "&toTs=%s" % last_timestamp
            url += "&api_key=%s" % os.environ['CRYPTOCOMPARE_API_KEY']
            data = requests.get(url)
            data = json.loads(data.text)
            if data['Response'] == 'Error':
                break
            data = data["Data"]
            df = pd.DataFrame.from_records(data)
            df['volume'] = df['volumefrom']
            df = df.drop(['volumefrom', 'volumeto'], axis=1)
            df['time'] = pd.to_datetime(df['time']*1000, unit='ms')
            df = df.sort_values(by='time', ascending=1).set_index('time')
            cdf = cdf.append(df)
            cdf = cdf[~cdf.index.duplicated(keep='last')]
            cdf = cdf.sort_index(ascending=1)
            if (df.empty or len(cdf) == plen or self.history_limit is None or
                    len(cdf) >= self.history_limit or df.volume.sum() == 0 or
                    (self.context and not self.context.backtesting())):
                break

            plen = len(cdf)
            last_timestamp = int(cdf.index[0].timestamp())

        if cdf.empty:
            return cdf
        return self._cleanup_and_save_symbol_data(symbol, cdf)


class QuandlDatacenter(BaseDatacenter):
    def __init__(self, table, *args, api_key=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._timeframe = '1d'
        self.to_sym = 'USD'
        self.table = table
        if api_key:
            self.api_key = api_key
            quandl.ApiConfig.api_key = api_key

    @property
    def name(self):
        return "quandl/%s" % self.table.lower()

    def load_markets(self):
        if not self._market_data:
            # download list of symbols from quandl
            path = join(self.data_dir, "codes.zip")
            if self.refresh_history or not isfile(path):
                url = "https://www.quandl.com/api/v3/databases/%s/codes"
                resp = requests.get(url % self.table, data={
                    "api_key": self.api_key}, stream=True)
                with open(path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024):
                        f.write(chunk)
                        f.flush()

            # parse list of symbols from zip file so downloaded
            with ZipFile(path) as qzip:
                with qzip.open('%s-datasets-codes.csv' % self.table, "r") as f:
                    text = f.read().decode('utf-8').splitlines()
                    self._market_data = [f.split(',')[0] for f in text]
        return self._market_data

    def all_symbols(self):
        """ Fetch all symbols supported by this exchange as a list """
        if self._selected_symbols:
            return self._selected_symbols

        if not self.markets:
            self.load_markets()
            self.markets = self._market_data
        return self.markets

    def symbol_to_assets(self, symbol):
        return "%s/%s" % (symbol.split("/")[1], "USD")

    def assets_to_symbol(self, fsym, tsym='USD'):
        return "%s/%s" % (self.table, fsym)

    def refresh_history_for_symbol(self, symbol, cdf=None):
        """ Refresh history for a given asset from exchange """
        cdf = quandl.get(symbol)

        # cleanup data (while ensuring compatibility with zipline)
        cdf = cdf.drop(["Open", "High", "Low", "Volume", "Close"], axis=1)
        cdf = cdf.rename(columns={"Ex-Dividend": "dividend",
                                  "Split Ratio": "split",
                                  "Adj. Close": "close", "Adj. Open": "open",
                                  "Adj. High": "high", "Adj. Low": "low",
                                  "Adj. Volume": "volume"})

        return self._cleanup_and_save_symbol_data(symbol, cdf)




class NseDatacenter(BaseDatacenter):
    def __init__(self, *args, start="2000-01-01", params={}, **kwargs):
        defaults = {"realign": False, "fill": True}
        super().__init__(*args, **{**defaults, **kwargs})
        self.exchange = Nse()
        self.to_sym='INR'
        self._market_data = {}
        self.start_date = start

    @property
    def name(self):
        return "stocks/nse/1d"

    def load_markets(self):
        if self.refresh_history:
            stocks = self.exchange.get_stock_codes(cached=False)
            stocks = set(stocks) - set(['SYMBOL'])
            self._market_data = {self.assets_to_symbol(k): {} for k in stocks}
        elif self.history is not None:
            self._market_data = dict(
                [(k, {}) for k in self.history.axes[0]])
        elif self.history is not None:
            raise ValueError("You must run with refresh=True to load markets")
        return self._market_data

    def all_symbols(self):
        """ Fetch all symbols supported by this exchange as a list """
        if self._selected_symbols:
            return self._selected_symbols

        self.markets = [key for key, val in self.load_markets().items()]
        return self.markets

    def symbol_to_assets(self, symbol):
        return symbol.split("/")

    def assets_to_symbol(self, fsym, tsym=None):
        tsym = tsym if tsym else self.to_sym
        return "%s/%s" % (fsym, tsym)

    def refresh_history_for_symbol(self, symbol, cdf=None, exact=False):
        """ Refresh history for a given asset from exchange """
        today = datetime.now().strftime("%Y-%m-%d")
        fsym, tsym = self.symbol_to_assets(symbol)
        stock = fsym if exact else "%s.NS" % fsym

        df = YahooFinancials(stock)
        df = df.get_historical_price_data(self.start_date, today, "daily")
        df = pd.DataFrame.from_records(df[stock]['prices'])

        if df.empty:
            return df

        df = df.drop(['date'], axis=1)
        df = df.rename(columns={"dividend_amount": "dividend", "split_coefficient": "split",
                        "close": "realclose", "adjclose": "close", "formatted_date": "time"})

        if 'time' not in df.columns or 'volume' not in df.columns:
            from IPython import embed; embed()

        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values(by='time', ascending=1).set_index('time')
        df = df[~np.isnat(df.index)]
        df.loc[df['volume']==0, :] = np.nan

        if df.empty:
            return df

        return self._cleanup_and_save_symbol_data(symbol, df)
