import os
from random import seed, random
from tabulate import tabulate

from .portfolio import BasePortfolio
from .account import SimulatedAccount, CcxtExchangeAccount
from .trading_session import BacktestSession, LiveTradingSession
from .datacenter import CryptocurrencyDatacenter as CryptoDC
from .broker import SimulatedBroker, CcxtExchangeBroker
from .notifier import FileLogger, SlackNotifier
from .benchmark import SymbolAsBenchmark, CryptoMarketCapAsBenchmark
from .reporter import (
    BaseReporter,
    CashAndEquityReporter,
    OrdersReporter
)


def ccxt_backtest(strat, start_time=None, end_time=None,
                  timeframe='1h', exchange='bittrex',
                  refresh=False, slippage=True,
                  balance={"BTC": 1}, initial_capital=None, commission=0.25,
                  benchmarks=False, debug=True, doprint=True, plots=True,
                  before_run=None, log_axis=False):
    pf = BacktestSession()
    pf.debug = debug
    pf.refresh_history = refresh

    pf.commission = commission
    pf.datacenter = CryptoDC(exchange, timeframe)
    pf.portfolio = BasePortfolio()
    pf.broker = SimulatedBroker()
    pf.account = SimulatedAccount(
            initial_balance=balance, initial_capital=initial_capital)

    pf.reporters = [
        OrdersReporter(),
        CashAndEquityReporter(bounds=False, mean=True, plot=plots, period=24*7,
                              log_axis=False, each_tick=False),
        BaseReporter(log_axis=log_axis, daily=False, plot=plots)]

    if benchmarks:
        pf.benchmarks = [
            SymbolAsBenchmark(),
            CryptoMarketCapAsBenchmark(),
            CryptoMarketCapAsBenchmark(include_btc=True)
        ]

    if slippage:
        seed(1)
        pf.slippage = lambda: 0.15 + random() * 0.5

    pf.strategy = strat
    pf.start_time = start_time
    pf.end_time = end_time

    if before_run:
        before_run(pf)

    pf.run()

    if doprint:
        print(tabulate(
            pf.reporters[2].data, headers='keys', tablefmt="orgtbl"))

    return pf


def binance_backtest(*args, **kwargs):
    defaults = {"commission": (0.05, 'BNB'),
                "balance": {"BTC": 1, "BNB": 20},
                "exchange": 'binance'}
    kwargs = {**defaults, **kwargs}
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
    pf.broker = CcxtExchangeBroker(exchange, opts)
    pf.account = CcxtExchangeAccount(exchange, opts)

    pf.reporters = [
        OrdersReporter(),
        CashAndEquityReporter(bounds=False, mean=True, plot=False,
                              period=24*7, log_axis=False, each_tick=True),
        BaseReporter(log_axis=False, daily=False, plot=False)
    ]

    pf.notifiers = [FileLogger(name), SlackNotifier(name, slack_url)]
    pf.strategy = strat
    pf.session = session
    pf.run(report=report)
    return pf
