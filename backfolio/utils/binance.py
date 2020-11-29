import datetime
import pandas as pd

from .common import print_emphasized, strat_name
from ..benchmark import CryptoMarketCapAsBenchmark
from .backtest import binance_backtest, add_sibling_benchmarks


def binance_backtest_in_time_period(strat, key, dur, refresh=False, **kwargs):
    """
    Run binance backtest for a given time period. Saves the equity curve (portfolio)
    as a benchmark with key `key` for use later (with `add_sibling_benchmarks`)

    `kwargs` can be specified which will be passed to `binance_backtest`
    utility function from the api.


    """
    kwargs = {
        **{
            "log_axis": True,
            "plots": True,
            "refresh": refresh
        },
        **kwargs
    }
    mapping = {
        "tpS": "sideways",
        "tpR": "rising",
        "tpF": "falling",
        "tpV": "validation",
        "tpD": "validation",
        "tpT": "validation",
        "tpN": "last_30"
    }

    def before_run(pf):
        pf.benchmarks = [CryptoMarketCapAsBenchmark()]
        add_sibling_benchmarks(pf, strat, key)

    print_emphasized("=> Type:     %s" % mapping[key])
    print_emphasized("=> Duration: %s" % " - ".join(dur))
    print_emphasized("=> Strategy: %s" % strat_name(strat))
    print("=" * 100)

    if 'before_run' not in kwargs:
        kwargs['before_run'] = before_run

    pf = None
    if len(dur) == 2:
        pf = binance_backtest(strat, dur[0], dur[1], **kwargs)
    elif len(dur) == 1:
        pf = binance_backtest(strat, dur[0], **kwargs)
    else:
        pf = binance_backtest(strat, **kwargs)

    name = strat_name(strat)
    pf.portfolio.save_as_benchmark(name, key)
    print("\n\n\n\n")
    return pf


def binance_backtest_for_each_month(strat,
                                    start="2017-10-01",
                                    end="2100-01-01",
                                    shift=0,
                                    **kwargs):
    """
    Run a strategy for every month in the calendar
    starting from binance inception.
    """
    ds = pd.date_range(start=start, end=end, freq='MS')
    de = pd.date_range(start=start, end=end, freq='M')

    ranges = []
    for i in range(0, len(de)):
        if ds[i] < datetime.datetime.now():
            s, e, fmt = ds[i], de[i] + datetime.timedelta(
                days=1), "%Y-%m-%d %H:%M"
            if shift != 0:
                tf = kwargs['timeframe'] if 'timeframe' in kwargs else '1h'
                delta = shift * pd.to_timedelta(tf)
                s, e = s - delta, e - delta
            ranges.append((s.strftime(fmt), e.strftime(fmt)))

    defaults = {
        "timeframe": '1h',
        "refresh": False,
        "each_tick": False,
        "plots": False,
        "debug": False,
        "doprint": False
    }
    kwargs = {**defaults, **kwargs}

    runs = {}
    for dr in ranges:
        print("Performance for [%s - %s]: " % dr, end="")
        start = datetime.datetime.now()
        pf = binance_backtest(strat, dr[0], dr[1], **kwargs)
        time_taken = datetime.datetime.now() - start
        eq = pf.portfolio.timeline.equity
        perf = pf.reporters[1].data['portfolio'].to_dict()
        data = (eq.iloc[-1] / eq.iloc[0], perf['max_drawdown'] * 100,
                perf['sharpe_ratio'], perf['sortino_ratio'],
                time_taken.total_seconds())
        print(
            "%9.4fx with DDN: %7.2f%%, Sharpe: %5.2f, Sortino: %6.2f [%4ds]" %
            data)
        pf, runs[dr[0]] = None, {"performance": perf, "equity": eq}

    equity1 = {k: v['equity'] for k, v in runs.items()}
    equitya = (1 + pd.DataFrame.from_records(equity1).pct_change().mean(axis=1)
               ).cumprod()
    perf = pd.DataFrame.from_records(
        {k: pd.Series(v['performance'])
         for k, v in runs.items()})
    data = {"performance": perf.T, "overall": equitya, "individual": equity1}
    equitya.plot()
    return data
