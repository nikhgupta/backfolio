import os, re, glob, operator

from .common import print_emphasized

from ..__init__ import __version__
from ..api import ccxt_backtest
from ..benchmark import CSVAsBenchmark


def bittrex_backtest(*args, **kwargs):
    return ccxt_backtest(*args, **kwargs)


def binance_backtest(*args, **kwargs):
    defaults = {
        "commission": (0.075, 'BNB'),
        "balance": {
            "BTC": 1,
            "BNB": 20
        },
        "exchange": 'binance'
    }
    kwargs = {**defaults, **kwargs}
    return ccxt_backtest(*args, **kwargs)


def add_sibling_benchmarks(pf, strat, key):
    """
    Add saved equity curves from strategies with same name prefix
    as benchmarks into the current backtest. e.g. if `strat` class is
    `MR05`, it will look for saved equity curves from `MR*` strategies
    with the same `key`.

    A `key` must be specified to only include equity curves corresponding
    to that key (a strategy can have multiple equity curves stored on disk)
    """
    bms = []
    name = re.sub("\d+$", "", strat.__class__.__name__)
    hits = glob.glob(
        os.path.join(os.path.expanduser("~"), ".backfolio", "benchmarks",
                     __version__, "%s*" % name, "*", "%s.csv" % key))
    for path in hits:
        _name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        if name == re.sub("\d+$", "",
                          _name) and _name != strat.__class__.__name__:
            path = re.sub("\.csv$", "", path)
            bms.append(CSVAsBenchmark(path,
                                      "%s-%s" % (_name, key[-1].upper())))

    for bm in sorted(bms, key=operator.attrgetter('name'))[-4:]:
        pf.benchmarks.append(bm)
