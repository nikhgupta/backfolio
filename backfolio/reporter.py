import pandas as pd
import empyrical as ep
import matplotlib.pyplot as plt
from .core.utils import fast_xs


class AbstractReporter(object):
    def __init__(self, name=''):
        self._name = name
        self._data = None
        self.session_fields = []

    def reset(self, context=None):
        self.context = context

    def generate_tick_report(self):
        """ Routine to run after trading for each tick is complete. At this
        stage most of the pandas dataframes are available as an array of dict
        records, except `account.balance` which is available as a dict.
        """
        pass

    def generate_summary_report(self):
        """ Routine to run after trading session has ended. The return value of
        this function call is returned as output from the main
        `cryptofolio.run` call """
        pass

    @property
    def name(self):
        return self._name

    @property
    def data(self):
        return self._data

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


class BaseReporter(AbstractReporter):
    def __init__(self, annualization=365, daily=True,
                 log_axis=True, plot=True):
        super().__init__(name='base-reporter')
        self.annualization = annualization
        self.log_axis = log_axis
        self.daily = daily
        self.plot = plot

    def generate_summary_report(self):
        if len(self.portfolio.timeline) < 2:
            return
        ticks = self.annualization
        benchmarks = self.benchmarks + [self.portfolio]
        df = pd.DataFrame(columns=[bm.name for bm in benchmarks])
        df.loc['invested'] = 1
        for bm in benchmarks:
            name, series = (bm.name, bm.daily)

            # regarding returns obtained from the strategy
            df.loc['final_return', name] = series.cum_returns.iloc[-1]
            df.loc['daily_return', name] = (series.cum_returns.iloc[-1]
                                            ** (1/len(series))) - 1
            df.loc['annual_return', name] = ep.annual_return(
                series.returns, period='daily', annualization=ticks)

            if self.account._extra_capital:
                if name == 'portfolio':
                    equity = series.equity + self.account._extra_capital
                    ret = (equity/equity.shift(1) - 1).fillna(0)
                    cumret = (1+ret).cumprod()
                else:
                    equity, ret, cumret = (
                        series.equity, series.returns, series.cum_returns)
                df.loc['overall_growth', name] = cumret.iloc[-1]
                df.loc['daily_growth', name] = (cumret.iloc[-1]
                                                ** (1/len(series))) - 1
                df.loc['annual_growth', name] = ep.annual_return(
                    ret, period='daily', annualization=ticks)

            # risk assessment of the strategy
            df.loc['max_drawdown', name] = ep.max_drawdown(series.returns)
            df.loc["annual_volatility", name] = ep.annual_volatility(
                series.returns, alpha=2.0, annualization=ticks)
            df.loc["downside_risk", name] = ep.downside_risk(
                series.returns, annualization=ticks)
            df.loc['stability', name] = ep.stability_of_timeseries(
                series.returns)

            # various ratio and other stats about strategy
            df.loc['sharpe_ratio', name] = ep.sharpe_ratio(
                series.returns, period='daily', annualization=ticks)
            df.loc['calmar_ratio', name] = ep.calmar_ratio(
                series.returns, period='daily', annualization=ticks)
            df.loc['sortino_ratio', name] = ep.sortino_ratio(
                series.returns, period='daily', annualization=ticks)
            df.loc['tail_ratio', name] = ep.tail_ratio(series.returns)
            df.loc['alpha', name], df.loc['beta', name] = ep.alpha_beta(
                self.portfolio.daily.returns, series.returns,
                period='daily', annualization=ticks)

        if self.plot:
            fig, ax = plt.subplots()
            ax.set_title('Performance Report')
            if self.log_axis:
                ax.set_yscale('log')

        for bm in benchmarks:
            name, series = (bm.name, bm.daily)
            returns = fast_xs(df, 'final_return')
            df.loc['performance', name] = (
                returns[self.portfolio.name]/returns[name])
            if self.plot:
                curve = bm.daily if self.daily else bm.timeline
                ax.plot(curve.cum_returns, label=bm.name)

        if self.plot:
            fig.legend()
            plt.show()

        self._data = df
        return df


class CashAndEquityReporter(AbstractReporter):
    def __init__(self, log_axis=False, mean=True, bounds=False,
                 each_tick=False, period=7, plot=True):
        super().__init__(name='cash-and-equity-reporter')
        self.log_axis = log_axis
        self.each_tick = each_tick
        self.period = period
        self.bounds = bounds
        self.mean = mean
        self.plot = plot

    def generate_tick_report(self):
        if not self.each_tick:
            return
        data = self.portfolio.timeline[-1]
        base = self.context.base_currency
        comm = self.context.commission_asset
        message = "Equity: %.8f %s, Cash: %.8f %s"
        message %= (data['equity'], base, data['cash'], base)
        if 'commission_paid' in data:
            message += ", CommPaid: %.8f %s"
            message %= (data['commission_paid'], comm)
        self.context.notify(message, formatted=True, now=data['time'])

    def _plot_with_averages(self, axis, name, data):
        axis.plot(data, label=name)
        rolling = data.rolling(self.period)
        if self.bounds:
            axis.plot(rolling.min(), label="Min %s" % name)
            axis.plot(rolling.max(), label="Max %s" % name)
        if self.mean:
            axis.plot(rolling.mean(), label="Avg %s" % name)

    def generate_summary_report(self):
        cash = self.portfolio.timeline.cash
        equity = self.portfolio.timeline.equity
        if len(equity) < 2 or not self.plot:
            return
        fig, ax = plt.subplots()
        ax.set_title('Cash vs Equity')
        if self.log_axis:
            ax.set_yscale('log')
        self._plot_with_averages(ax, 'Cash', cash)
        self._plot_with_averages(ax, 'Equity', equity)
        fig.legend()
        plt.show()


class OrdersReporter(AbstractReporter):
    def __init__(self):
        super().__init__(name='orders-reporter')

    def generate_summary_report(self):
        total = len(self.portfolio.orders)
        if total:
            filled = len(self.portfolio.filled_orders)/total*100
            rejected = len(self.portfolio.rejected_orders)/total*100
            unfilled = len(self.portfolio.unfilled_orders)/total*100
            ignored = 100 - (filled + rejected + unfilled)
            each_tick = total / len(self.portfolio.timeline)

            print("Order Placement Summary")
            print("=======================")
            print(("Total: %d orders, Per Tick: %.2f orders\n" +
                   "Filled: %.2f%%, Unfilled: %.2f%%, " +
                   "Rejected: %.2f%%, Ignored: %.2f%%\n") % (
                      total, each_tick, filled, unfilled, rejected, ignored))
