import numpy as np
import pandas as pd
import empyrical as ep
import matplotlib.pyplot as plt


class AbstractReporter(object):
    def __init__(self, name=''):
        self._name = name
        self._data = None
        self.session_fields = []

    def reset(self, context=None):
        self.context = context
        return self

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

    @property
    def current_time(self):
        return self.context.current_time


class BaseReporter(AbstractReporter):
    def __init__(self, annualization=None, log_axis=True,
            plot=True, doprint=True):
        super().__init__(name='base-reporter')
        self.annualization = annualization
        self.log_axis = log_axis
        self.plot = plot
        self.doprint = doprint

    def generate_summary_report(self):
        if len(self.portfolio.timeline) < 2:
            return
        ticks = self.annualization
        if not ticks:
            eq = self.portfolio.timeline.returns
            freq = pd.tseries.frequencies.to_offset(pd.infer_freq(eq.index))
            ticks = int((365*24*60*60)/pd.to_timedelta(freq).total_seconds())

        benchmarks = self.benchmarks + [self.portfolio]
        df = pd.DataFrame(columns=[bm.name for bm in benchmarks])
        df.loc['invested'] = 1
        for bm in benchmarks:
            name, series, dseries = (bm.name, bm.timeline, bm.daily)
            pofret = self.portfolio.timeline.returns

            if series.empty:
                continue

            # regarding returns obtained from the strategy
            df.loc['final_return', name] = series.cum_returns.iloc[-1]
            df.loc['daily_return', name] = (series.cum_returns.iloc[-1]
                                            ** (ticks/365/len(series))) - 1
            df.loc['annual_return', name] = ep.annual_return(series.returns,
                    period='daily', annualization=ticks)
            df.loc['performance', name] = (
                series.cum_returns.iloc[-1]
                / benchmarks[-1].timeline.cum_returns.iloc[-1])

            if self.account._extra_capital:
                if name == 'portfolio':
                    equity = series.equity + self.account._extra_capital
                    ret = (equity/equity.shift(1) - 1).fillna(0)
                    cumret = (1+ret).cumprod()
                else:
                    equity, ret, cumret = (
                        series.equity, series.returns, series.cum_returns)
                    dequity, dret, dcumret = (
                        dseries.equity, dseries.returns, dseries.cum_returns)
                df.loc['overall_growth', name] = cumret.iloc[-1]
                df.loc['daily_growth', name] = (cumret.iloc[-1]
                                                ** (ticks/365/len(series))) - 1
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
            df.loc['sharpe_ratio', name] = ep.sharpe_ratio( series.returns, annualization=ticks)
            df.loc['calmar_ratio', name] = ep.calmar_ratio(
                series.returns, annualization=ticks)
            df.loc['sortino_ratio', name] = ep.sortino_ratio(
                series.returns, annualization=ticks)
            df.loc['tail_ratio', name] = ep.tail_ratio(series.returns)
            df.loc['alpha', name], df.loc['beta', name] = ep.alpha_beta(
                self.portfolio.timeline.returns,
                series.returns, annualization=ticks)

            df.loc['correlation', name] = series.returns.corr(pofret)
            df.loc['covariance', name] = (pofret.cov(series.returns)
                                          / pofret.cov(pofret))

        if self.plot:
            fig, ax = plt.subplots()
            ax.set_title('Performance Report')
            if self.log_axis:
                ax.set_yscale('log')

        for bm in benchmarks:
            if self.plot:
                curve = bm.timeline
                if bm.name == 'portfolio':
                    ax.plot(curve.cum_returns, label=bm.name,
                            color='black', linewidth=3)
                else:
                    ax.plot(curve.cum_returns, label=bm.name)

        if self.plot:
            fig.legend()
            plt.show()

        self._data = df
        if self.doprint:
            print(df)
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
        free = self.account.cash
        base = self.context.base_currency
        comm = self.context.commission_asset
        message = "Equity: %.8f %s, Cash: %.8f %s"
        message %= (data['equity'], base, free, base)
        if 'commission_paid' in data and not np.isnan(data['commission_paid']):
            message += ", CommPaid: %.8f %s"
            message %= (data['commission_paid'], comm)
        self.context.notify(message, formatted=True, now=data['time'], publish=False)

    def _plot_with_averages(self, axis, name, data, **kwargs):
        axis.plot(data, label=name)
        rolling = data.rolling(self.period)
        if self.bounds:
            axis.plot(rolling.min(), label="Min %s" % name)
            axis.plot(rolling.max(), label="Max %s" % name)
        if self.mean:
            axis.plot(rolling.mean(), label="Avg %s" % name, **kwargs)

    def generate_summary_report(self):
        cash = self.portfolio.timeline.cash
        equity = self.portfolio.timeline.equity
        if len(equity) < 2 or not self.plot:
            return
        fig, ax = plt.subplots()
        ax.set_title('Cash vs Equity')
        if self.log_axis:
            ax.set_yscale('log')
        self._plot_with_averages(ax, 'Cash', cash, color='black')
        self._plot_with_averages(ax, 'Equity', equity, color='r')
        fig.legend()
        plt.show()


class OrdersReporter(AbstractReporter):
    def __init__(self):
        super().__init__(name='orders-reporter')

    def generate_summary_report(self):
        og = self.portfolio.order_groups
        if og.empty:
            return
        cog = pd.DataFrame(og[og.status == 'CLOSED'])
        if cog.empty:
            return

        print("Order Placement Summary")
        cog['duration'] = (cog['ended_at'].astype(int)
                           - cog['started_at'].astype(int))/3600/1e9
        wins = cog[cog.total_profits > 0]
        lost = cog[cog.total_profits < 0]
        wr = len(wins)/len(cog)*100
        lost_money = abs(lost.total_profits.sum())
        won_money = wins.total_profits.sum()
        pf = won_money/lost_money if lost_money > 0 else np.nan
        gog = og.groupby('asset').sum()
        gog['buy_price'] = gog['buy_cost']/gog['buy_quantity']
        gog['sell_price'] = gog['sell_cost']/gog['sell_quantity']

        summary = self.portfolio.last_positions_and_equities_at_tick_close()
        gog['remaining_equity'] = pd.Series(summary['asset_equity'])
        gog['commission_deducted'] = self.portfolio.closed_orders.groupby(
            'commission_asset').sum().commission_cost
        gog['commission_deducted'] = gog['commission_deducted'].fillna(0)
        gog['net_profits'] = (gog['total_profits'] + gog['remaining_equity']
                              + gog['commission_deducted'])
        gog['profit%'] = gog['net_profits']/gog['buy_cost']*100
        actual = gog.net_profits.sum() + self.portfolio.timeline.iloc[0].equity
        reported = self.portfolio.timeline.iloc[-1].equity
        error = actual/reported*100-100
        lt, wt = lost.duration.mean(), wins.duration.mean()
        self._data = gog.sort_values(by='profit%', ascending=0)

        total = len(self.portfolio.orders)
        if total:
            def hr(sep, size=75): print("+"+sep*size+"+")
            filled = len(self.portfolio.filled_orders)/total*100
            rejected = len(self.portfolio.rejected_orders)/total*100
            unfilled = len(self.portfolio.unfilled_orders)/total*100
            open = len(self.portfolio.open_orders)/total*100
            ignored = 100 - (filled + rejected + unfilled + open)
            each_tick = total / len(self.portfolio.timeline)

            hr("=")
            print(("| WinR:  %7.2f%% | PF: %7.2fx | WD: %7.2fT " +
                   "| LD: %.2fT | Err: %6.2f%%  |") % (wr, pf, wt, lt, error))
            hr('-')
            print(("| ClosedTrades: %5d | TotalTrades: %5d " +
                   "| OrdersInClosedTrades: %5d    |") % (
                      len(cog), len(og), cog.num_orders.sum()))
            hr("=")
            print(("| TotalOrders: %6d | PerTick: %7.2f | Closed: %6.2f%% " +
                   "| Open: %5.2f%%   |") % (total, each_tick, filled, open))
            hr("-")
            print(("| Rejected:    %5.2f%% | Unfilled: %5.2f%% " +
                   "| Ignored: %5.2f%%                  |") % (
                      rejected, unfilled, ignored))
            hr("=")
        return self.data


class MonteCarloAnalysis(AbstractReporter):
    def __init__(self, simulations=2000, histogram=True, log_axis=True,
                 bins=20):
        super().__init__(name='monte-carlo')
        self.simulations = simulations
        self.histogram = histogram
        self.log_axis = log_axis
        self.bins = bins

    @staticmethod
    def max_dd(returns):
        """Assumes returns is a pandas Series"""
        r = returns.add(1).cumprod()
        dd = r.div(r.cummax()).sub(1)
        mdd = dd.min()
        end = dd.argmin()
        start = r.loc[:end].argmax()
        return mdd, start, end

    def generate_summary_report(self):
        result_wr, result_wor = [], []
        ret = 1 + self.portfolio.timeline.returns

        fig, axs = plt.subplots(3, 2, figsize=(16, 16))
        for i in range(self.simulations):
            vals = ret.values
            np.random.shuffle(vals)
            price_list_wr = [1]
            price_list_wor = [1]
            for wor in vals:
                wr = vals[np.random.randint(len(vals))]
                price_list_wr.append(price_list_wr[-1]*wr)
                price_list_wor.append(price_list_wor[-1]*wor)
            axs[0][0].plot(price_list_wor, color='gray', alpha=0.1)
            axs[0][1].plot(price_list_wr, color='gray', alpha=0.1)
            result_wr.append(price_list_wr)
            result_wor.append(price_list_wor)

        fig.suptitle('Monte Carlo Equity Curve Analysis %s' % (
            ' (Logarithmic)' if self.log_axis else ''))
        axs[0][0].set_title('Equity Without replacement')
        axs[0][0].plot(ret.cumprod().values, color='r', linewidth=3)
        axs[0][1].set_title('Equity With replacement')
        axs[0][1].plot(ret.cumprod().values, color='r', linewidth=3)
        if self.log_axis:
            axs[0][0].set_yscale('log')
            axs[0][1].set_yscale('log')

        ret = [x[-1] for x in result_wr]
        sharpe = [np.mean(x)/np.std(x) for x in result_wr]

        dds_wr = [self.max_dd(pd.Series(i)-1) for i in result_wr]
        dds_wor = [self.max_dd(pd.Series(i)-1) for i in result_wor]
        axs[1][0].hist([-min(x)*100 for x in dds_wor], bins=self.bins)
        axs[1][0].set_title('Max Drawdown Histogram (without replacement)')
        axs[1][1].hist([-min(x)*100 for x in dds_wr], bins=self.bins)
        axs[1][1].set_title('Max Drawdown Histogram (with replacement)')

        axs[2][0].hist(ret, bins=self.bins)
        axs[2][0].axvline(np.mean(ret), color='r', linewidth=2)
        axs[2][0].axvline(np.percentile(ret, 5), color='r',
                          linestyle='dashed', linewidth=2)
        axs[2][0].axvline(np.percentile(ret, 95), color='r',
                          linestyle='dashed', linewidth=2)
        axs[2][0].set_title('Final Equity Histogram (with replacement)')
        axs[2][1].hist(sharpe, bins=self.bins)
        axs[2][1].axvline(np.mean(sharpe), color='r', linewidth=2)
        axs[2][1].axvline(np.percentile(sharpe, 5), color='r',
                          linestyle='dashed', linewidth=2)
        axs[2][1].axvline(np.percentile(sharpe, 95), color='r',
                          linestyle='dashed', linewidth=2)
        axs[2][1].set_title('Mean/Stdev Histogram (with replacement)')

        plt.show()
