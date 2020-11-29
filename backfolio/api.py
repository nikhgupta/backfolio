import os
import re
import math
import random
import requests
import pandas as pd
import empyrical as ep
from bs4 import BeautifulSoup
from tabulate import tabulate
import matplotlib.pyplot as plt
from os.path import join, expanduser

from .portfolio import BasePortfolio
from .account import SimulatedAccount, CcxtExchangeAccount
from .trading_session import BacktestSession, LiveTradingSession
from .datacenter import *
from .datacenter import BinanceDatacenter as BinanceDC
from .datacenter import CryptocurrencyDatacenter as CryptoDC
from .broker import SimulatedBroker, CcxtExchangeBroker
from .notifier import FileLogger, SlackNotifier
from .benchmark import *
from .core.utils import fast_xs
from .reporter import (BaseReporter, CashAndEquityReporter, OrdersReporter)


def ccxt_backtest(strat,
                  start_time=None,
                  end_time=None,
                  timeframe='1h',
                  exchange='bittrex',
                  resample=None,
                  refresh=False,
                  slippage=True,
                  run=True,
                  prefer_ccxt=True,
                  balance={"BTC": 1},
                  initial_capital=None,
                  commission=0.25,
                  benchmarks=False,
                  debug=True,
                  doprint=True,
                  plots=True,
                  before_run=None,
                  log_axis=False,
                  each_tick=False,
                  base_currency='BTC'):
    pf = BacktestSession(base_currency=base_currency)
    random.seed(1)
    pf.debug = debug
    pf.refresh_history = refresh

    pf.commission = commission
    pf.datacenter = CryptoDC(exchange,
                             timeframe,
                             resample=resample,
                             to_sym=base_currency)

    if exchange == 'binance' and base_currency == 'USDT':
        pf.datacenter = BinanceUsdtDatacenter(timeframe=timeframe,
                                              resample=resample)
    elif exchange == 'binance' and not prefer_ccxt:
        pf.datacenter = BinanceDC(timeframe=timeframe, resample=resample)

    pf.portfolio = BasePortfolio()
    pf.broker = SimulatedBroker()
    pf.account = SimulatedAccount(initial_balance=balance,
                                  initial_capital=initial_capital)

    pf.reporters = [
        CashAndEquityReporter(bounds=False,
                              mean=True,
                              plot=plots,
                              period=24 * 7,
                              log_axis=False,
                              each_tick=each_tick),
        BaseReporter(log_axis=log_axis, plot=plots, doprint=False)
    ]
    if doprint:
        pf.reporters.append(OrdersReporter())

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

        if doprint and len(pf.reporters) > 1:
            print(
                tabulate(pf.reporters[1].data,
                         headers='keys',
                         tablefmt="orgtbl"))

    return pf


def ccxt_live(name,
              session,
              strat,
              cred,
              slack_url,
              prefer_ccxt=True,
              timeframe='1h',
              exchange='bittrex',
              poll_frequency=None,
              resample=None,
              debug=True,
              slippage=True,
              commission=0.25,
              report=True,
              cancel_pending=False):
    """
    Run trading bot in live mode with a given name and session.
    """
    pf = LiveTradingSession()
    pf.debug = debug
    pf.poll_frequency = poll_frequency
    pf.commission = commission

    pf.portfolio = BasePortfolio()
    pf.datacenter = CryptoDC(exchange, timeframe, resample=resample)
    if exchange == 'binance' and not prefer_ccxt:
        pf.datacenter = BinanceDC(timeframe=timeframe, resample=resample)

    if cred is None:
        cred = {
            'apiKey': os.environ['%s_API_KEY' % exchange.upper()],
            'secret': os.environ['%s_SECRET' % exchange.upper()]
        }

    if slack_url is None:
        slack_url = os.environ['BACKFOLIO_SLACK_URL']

    opts = {**cred, **{'adjustForTimeDifference': True}}
    pf.broker = CcxtExchangeBroker(exchange, params=opts)
    pf.account = CcxtExchangeAccount(exchange, params=opts)

    pf.reporters = [
        OrdersReporter(),
        CashAndEquityReporter(bounds=False,
                              mean=True,
                              plot=False,
                              period=24 * 7,
                              log_axis=False,
                              each_tick=True)
    ]

    pf.notifiers = [FileLogger(name), SlackNotifier(name, slack_url)]
    pf.strategy = strat
    pf.session = session
    pf.cancel_pending = cancel_pending
    print("Current session: %s" % pf.session)

    pf.run(report=report)
    return pf


def quick_bt(history,
             lookup,
             top=None,
             bottom=None,
             ft=None,
             tt=None,
             fee=0.05,
             rebalance=None,
             plots=True):
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
            scorers = lookup.loc[pidx].dropna().sort_values(
                ascending=False).index
            bar[idx] = math.fsum([
                fast_xs(ret, idx)[asset] - 2 * fee / 100
                for asset in scorers[:top]
            ])
            war[idx] = math.fsum([
                fast_xs(ret, idx)[asset] - 2 * fee / 100
                for asset in scorers[-bottom:]
            ])
        pidx = idx

    df = pd.DataFrame()
    df['market'] = (ret.sum(axis=1) / ret.count(axis=1)).cumsum()
    df['top'] = (pd.Series(bar) / top).cumsum()
    df['bottom'] = (pd.Series(war) / bottom).cumsum()

    md, td, bd = df['market'].diff(), df['top'].diff(), df['bottom'].diff()
    msh = ep.sharpe_ratio(md)
    tsh = ep.sharpe_ratio(td)
    bsh = ep.sharpe_ratio(bd)
    mso = ep.sortino_ratio(md)
    tso = ep.sortino_ratio(td)
    bso = ep.sortino_ratio(bd)
    mdd = ep.max_drawdown(md)
    tdd = ep.max_drawdown(td)
    bdd = ep.max_drawdown(bd)
    tcorr = df['market'].corr(df['top'])
    tcov = df['market'].cov(df['top'])
    bcorr = df['market'].corr(df['bottom'])
    bcov = df['market'].cov(df['bottom'])

    common = "returns: %7.2fx, sharpe: %7.2f, sortino: %7.2f, drawdn: %5.2f%%"
    print(("market %s" % common) % (df.market.iloc[-1], msh, mso, mdd))
    print(("bottom %s, corr: %%5.2f, cov: %%7.2f" % common) %
          (df.bottom.iloc[-1], bsh, bso, bdd, bcorr, bcov))
    print(("   top %s, corr: %%5.2f, cov: %%7.2f" % common) %
          (df.top.iloc[-1], tsh, tso, tdd, tcorr, tcov))

    if plots:
        fig, axs = plt.subplots(1, 3)
        fig.set_size_inches(16, 16 / 3)
        fig.suptitle('Assets Score Performance - (Log Scale)', fontsize=20)
        df[['market', 'top']].plot(kind='area',
                                   stacked=False,
                                   ax=axs[0],
                                   linewidth=0,
                                   color=['b', 'g'],
                                   title='Top Scorer Performance')
        df[['market', 'bottom']].plot(kind='area',
                                      stacked=False,
                                      ax=axs[1],
                                      linewidth=0,
                                      color=['b', 'r'],
                                      title='Bottom Scorer Performance')

        linear = df['market'].apply(lambda _: 1.025**(1 / 24) - 1).cumsum()
        P = df['top'].rolling(24).cov(linear).plot(
            kind='area',
            stacked=False,
            linewidth=0,
            color='g',
            title='Portfolio vs. Market',
            yticks=None)
        df['bottom'].rolling(24).cov(linear).plot(kind='area',
                                                  stacked=False,
                                                  linewidth=0,
                                                  color='r',
                                                  alpha=0.1,
                                                  title='Portfolio vs. Market',
                                                  yticks=None)
        P.axes.get_yaxis().set_visible(False)
    return df
