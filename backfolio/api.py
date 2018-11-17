import os
import math
import random
import pandas as pd
import empyrical as ep
from tabulate import tabulate
import matplotlib.pyplot as plt

from .portfolio import BasePortfolio
from .account import SimulatedAccount, CcxtExchangeAccount
from .trading_session import BacktestSession, LiveTradingSession
from .datacenter import CryptocurrencyDatacenter as CryptoDC
from .broker import SimulatedBroker, CcxtExchangeBroker
from .notifier import FileLogger, SlackNotifier
from .benchmark import SymbolAsBenchmark, CryptoMarketCapAsBenchmark
from .core.utils import fast_xs
from .reporter import (
    BaseReporter,
    CashAndEquityReporter,
    OrdersReporter
)


def ccxt_backtest(strat, start_time=None, end_time=None,
                  timeframe='1h', exchange='bittrex', resample=None,
                  refresh=False, slippage=True, run=True,
                  balance={"BTC": 1}, initial_capital=None, commission=0.25,
                  benchmarks=False, debug=True, doprint=True, plots=True,
                  before_run=None, log_axis=False, each_tick=False):
    pf = BacktestSession()
    random.seed(1)
    pf.debug = debug
    pf.refresh_history = refresh

    pf.commission = commission
    pf.datacenter = CryptoDC(exchange, timeframe, resample=resample)
    pf.portfolio = BasePortfolio()
    pf.broker = SimulatedBroker()
    pf.account = SimulatedAccount(
            initial_balance=balance, initial_capital=initial_capital)

    pf.reporters = [
        OrdersReporter(),
        CashAndEquityReporter(bounds=False, mean=True, plot=plots, period=24*7,
                              log_axis=False, each_tick=each_tick),
        BaseReporter(log_axis=log_axis, daily=False,
                     plot=plots, doprint=False)]

    if benchmarks:
        pf.benchmarks = [
            SymbolAsBenchmark(),
            CryptoMarketCapAsBenchmark(),
            CryptoMarketCapAsBenchmark(include_btc=True)
        ]

    if slippage:
        pf.slippage = lambda: 0.15 + random.random() * 0.5

    pf.strategy = strat
    pf.start_time = start_time
    pf.end_time = end_time

    if before_run:
        before_run(pf)

    if run:
        pf.run()

        if doprint:
            print(tabulate(
                pf.reporters[2].data, headers='keys', tablefmt="orgtbl"))

    return pf


def binance_backtest(*args, **kwargs):
    defaults = {"commission": (0.075, 'BNB'),
                "balance": {"BTC": 1, "BNB": 20},
                "exchange": 'binance'}
    kwargs = {**defaults, **kwargs}
    return ccxt_backtest(*args, **kwargs)


def bittrex_backtest(*args, **kwargs):
    return ccxt_backtest(*args, **kwargs)


def ccxt_live(name, session, strat, cred, slack_url,
              timeframe='1h', exchange='bittrex', poll_frequency=None,
              debug=True, slippage=True, commission=0.25, report=True):
    """
    Run trading bot in live mode with a given name and session.
    """
    pf = LiveTradingSession()
    pf.debug = debug
    pf.poll_frequency = poll_frequency
    pf.commission = commission

    pf.portfolio = BasePortfolio()
    pf.datacenter = CryptoDC(exchange, timeframe)

    if cred is None:
        cred = {'apiKey': os.environ['%s_API_KEY' % exchange.upper()],
                'secret': os.environ['%s_SECRET' % exchange.upper()]}

    if slack_url is None:
        slack_url = os.environ['BACKFOLIO_SLACK_URL']

    opts = {**cred, **{'adjustForTimeDifference': True}}
    pf.broker = CcxtExchangeBroker(exchange, params=opts)
    pf.account = CcxtExchangeAccount(exchange, params=opts)

    pf.reporters = [
        OrdersReporter(),
        CashAndEquityReporter(bounds=False, mean=True, plot=False,
                              period=24*7, log_axis=False, each_tick=True)
    ]

    pf.notifiers = [FileLogger(name), SlackNotifier(name, slack_url)]
    pf.strategy = strat
    pf.session = session
    pf.run(report=report)
    return pf


def quick_bt(history, lookup, top=None, bottom=None, ft=None, tt=None,
             fee=0.05, annualization=365, rebalance=None, plots=True):
    """
    Quick vectorized backtesting to explore worthy strategies.
    """
    if ft:
        history = history[:, ft:]
    if tt:
        history = history[:, :tt]

    close = history[:, :, 'close']
    if rebalance:
        close = close.resample(rebalance).last()
    ret = close.pct_change()

    pidx, bar, war = None, {}, {}
    for idx, row in ret.iterrows():
        if pidx:
            scorers = lookup.loc[pidx].dropna().sort_values(ascending=0).index
            bar[idx] = math.fsum([fast_xs(ret, idx)[asset] - 2*fee/100
                                  for asset in scorers[:top]])
            war[idx] = math.fsum([fast_xs(ret, idx)[asset] - 2*fee/100
                                  for asset in scorers[-bottom:]])
        pidx = idx

    df = pd.DataFrame()
    df['market'] = (ret.sum(axis=1)/ret.count(axis=1)).cumsum()
    df['top'] = (pd.Series(bar)/top).cumsum()
    df['bottom'] = (pd.Series(war)/bottom).cumsum()

    md, td, bd = df['market'].diff(), df['top'].diff(), df['bottom'].diff()
    msh = ep.sharpe_ratio(md, annualization=annualization)
    tsh = ep.sharpe_ratio(td, annualization=annualization)
    bsh = ep.sharpe_ratio(bd, annualization=annualization)
    mso = ep.sortino_ratio(md, annualization=annualization)
    tso = ep.sortino_ratio(td, annualization=annualization)
    bso = ep.sortino_ratio(bd, annualization=annualization)
    mdd = ep.max_drawdown(md)
    tdd = ep.max_drawdown(td)
    bdd = ep.max_drawdown(bd)
    tcorr = df['market'].corr(df['top'])
    tcov = df['market'].cov(df['top'])
    bcorr = df['market'].corr(df['bottom'])
    bcov = df['market'].cov(df['bottom'])

    common = "returns: %7.2fx, sharpe: %7.2f, sortino: %7.2f, drawdn: %5.2f%%"
    print(("market %s" % common) % (
        df.market.iloc[-1], msh, mso, mdd))
    print(("bottom %s, corr: %%5.2f, cov: %%7.2f" % common) % (
        df.bottom.iloc[-1], bsh, bso, bdd, bcorr, bcov))
    print(("   top %s, corr: %%5.2f, cov: %%7.2f" % common) % (
        df.top.iloc[-1], tsh, tso, tdd, tcorr, tcov))

    if plots:
        fig, axs = plt.subplots(1, 3)
        fig.set_size_inches(16, 16/3)
        fig.suptitle('Assets Score Performance - (Log Scale)', fontsize=20)
        df[['market', 'top']].plot(
            kind='area', stacked=False, ax=axs[0], linewidth=0,
            color=['b', 'g'], title='Top Scorer Performance')
        df[['market', 'bottom']].plot(
            kind='area', stacked=False, ax=axs[1], linewidth=0,
            color=['b', 'r'], title='Bottom Scorer Performance')

        linear = df['market'].apply(lambda _: 1.025**(1/24) - 1).cumsum()
        P = df['top'].rolling(24).cov(linear).plot(
            kind='area', stacked=False, linewidth=0,
            color='g', title='Portfolio vs. Market', yticks=None)
        df['bottom'].rolling(24).cov(linear).plot(
            kind='area', stacked=False, linewidth=0,
            color='r', alpha=0.1, title='Portfolio vs. Market', yticks=None)
        P.axes.get_yaxis().set_visible(False)

    return df
