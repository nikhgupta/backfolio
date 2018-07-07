import time
import traceback
from datetime import datetime
from copy import deepcopy
from os.path import join, expanduser

from .core.utils import make_path, load_df, save_df
from .core.queue import (EventQueue, EventQueueEmpty)
from .core.event import (
    TickUpdateEvent,
    StrategyAdviceEvent,
    OrderRequestedEvent,
    OrderFilledEvent,
    OrderUnfilledEvent,
    OrderRejectedEvent,
    OrderCreatedEvent,
    OrderPendingEvent
)


class TradingSession:
    def __init__(self, debug=False, base_currency='BTC'):
        self._mode = None
        self.debug = debug
        self.base_currency = base_currency
        self.commission = (0, base_currency)
        self.events = EventQueue()
        self.session_fields = []
        self._account = None
        self._broker = None
        self._strategy = None
        self._datacenter = None
        self._portfolio = None
        self._reporters = []
        self._benchmarks = []
        self._notifiers = []
        self._root_dir = None
        self._slippage = lambda: 0
        self._refresh_history = True
        self._start_time = None
        self._end_time = None
        self._current_time = None

    @property
    def mode(self):
        return self._mode

    @property
    def debug(self):
        return self._debug

    @debug.setter
    def debug(self, value):
        self._debug = value

    @property
    def refresh_history(self):
        return self._refresh_history

    @refresh_history.setter
    def refresh_history(self, value):
        self._refresh_history = value

    @property
    def start_time(self):
        return self._start_time

    @start_time.setter
    def start_time(self, value):
        self._start_time = value

    @property
    def end_time(self):
        return self._end_time

    @end_time.setter
    def end_time(self, value):
        self._end_time = value

    @property
    def root_dir(self):
        if self._root_dir:
            return self._root_dir
        else:
            return join(expanduser("~"), ".backfolio")

    @root_dir.setter
    def root_dir(self, value):
        make_path(value)
        self._root_dir = value

    @property
    def base_currency(self):
        return self._base_currency

    @base_currency.setter
    def base_currency(self, value):
        self._base_currency = value

    @property
    def commission(self):
        return self._commission

    @property
    def commission_asset(self):
        return self._commission_asset

    @commission.setter
    def commission(self, value):
        if hasattr(value, '__len__') and len(value) == 2:
            self._commission, self._commission_asset = value
        else:
            self._commission = value
            self._commission_asset = self.base_currency

    @property
    def slippage(self):
        return self._slippage

    @slippage.setter
    def slippage(self, slippage_fn):
        self._slippage = slippage_fn

    @property
    def broker(self):
        return self._broker

    @broker.setter
    def broker(self, inst):
        self._broker = inst

    @property
    def account(self):
        return self._account

    @account.setter
    def account(self, inst):
        self._account = inst

    @property
    def datacenter(self):
        return self._datacenter

    @datacenter.setter
    def datacenter(self, inst):
        self._datacenter = inst

    @property
    def portfolio(self):
        return self._portfolio

    @portfolio.setter
    def portfolio(self, inst):
        self._portfolio = inst

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, inst):
        self._strategy = deepcopy(inst)

    @property
    def reporters(self):
        if self._reporters is None:
            self._reporters = []
        return self._reporters

    @reporters.setter
    def reporters(self, arr=[]):
        self._reporters = arr

    def add_reporter(self, reporter):
        self._reporters.append(reporter)

    def run_report(self, reporter):
        return reporter.reset(self).generate_summary_report()

    @property
    def benchmarks(self):
        if self._benchmarks is None:
            self._benchmarks = []
        return self._benchmarks

    @benchmarks.setter
    def benchmarks(self, arr=[]):
        self._benchmarks = arr

    def add_benchmark(self, benchmark):
        self._benchmarks.append(benchmark)

    @property
    def notifiers(self):
        if self._notifiers is None:
            self._notifiers = []
        return self._notifiers

    @notifiers.setter
    def notifiers(self, arr=[]):
        self._notifiers = arr

    def add_notifier(self, notifier):
        self._notifiers.append(notifier)

    @property
    def current_time(self):
        return self._current_time

    def log(self, message):
        if self.debug:
            print(message)

    def notify(self, message, formatted=True, now=None):
        if now is None:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.log("[%s]: %s" % (now, message))
        method = 'formatted_notify' if formatted else 'notify'
        for notifier in self.notifiers:
            if hasattr(notifier, method):
                getattr(notifier, method)(message, now)

    def notify_error(self, err):
        str = "[%s] - %s" % (err.__class__.__name__, err.__str__())
        str += "\n" + traceback.format_exc()
        self.notify(str, formatted=True)

    def backtesting(self):
        return self.mode == 'backtest'

    def paper_trading(self):
        return self.mode == 'paper'

    def live_trading(self):
        return self.mode == 'live'

    def has_session(self):
        return (hasattr(self, 'session') and
                self.session is not None and
                self.session.strip())

    def _with_each_component(self, fn):
        fn('datacenter', self.datacenter)
        fn('account', self.account)
        fn('portfolio', self.portfolio)
        fn('broker', self.broker)
        for component in self.reporters:
            fn('reporter', component)
        for component in self.benchmarks:
            fn('benchmark', component)
        for component in self.notifiers:
            fn('notifier', component)
        fn('strategy', self.strategy)

    def _load_state_from_session_for_component(self, name, component):
        if not hasattr(component, 'session_fields'):
            return
        for field in component.session_fields:
            name = component.name if hasattr(component, "name") else name
            name = "%s-%s" % (name, field)
            load_df(self.session_dir, name, component, field)

    def _load_state_from_session(self, fields=[]):
        if not self.has_session():
            return
        self._load_state_from_session_for_component('trading-session', self)
        self._with_each_component(self._load_state_from_session_for_component)

    def _save_state_in_session_for_component(self, name, component):
        if not hasattr(component, 'session_fields'):
            return
        for field in component.session_fields:
            value = getattr(component, field)
            name = component.name if hasattr(component, "name") else name
            name = "%s-%s" % (name, field)
            save_df(self.session_dir, name, value)

    def _save_state_in_session(self, fields=[]):
        if not self.has_session():
            return
        self._save_state_in_session_for_component('trading-session', self)
        self._with_each_component(self._save_state_in_session_for_component)

    def _run_hook(self, name, *args):
        if hasattr(self.strategy, name):
            return getattr(self.strategy, name)(*args)

    def mutate_datacenter_history(self, refresh=True):
        if refresh:
            self.datacenter.reload_history(refresh=refresh)
        history = self.datacenter.history
        if hasattr(self.strategy, 'transform_history'):
            history = self.strategy.transform_history(history)
        self.datacenter.replace_history(history)

    def _run_benchmarks(self):
        self._run_hook('before_benchmarks')
        for benchmark in self.benchmarks:
            benchmark.daily  # trigger run of benchmark
        self._run_hook('after_benchmarks')

    def _before_trading_start(self):
        pass

    def _before_each_tick(self):
        pass

    def _on_each_tick(self):
        self.broker.check_order_statuses()

    def _should_reloop(self):
        return True

    def _after_each_tick(self):
        pass

    def _after_trading_end(self):
        pass

    def _get_next_tick(self):
        ts = self.datacenter.last_tick()
        if datetime.utcnow() > ts + self.datacenter.timeframe_delta:
            self.events.get(block=False)  # remove existing tick event
            self.mutate_datacenter_history(refresh=self.refresh_history)
            ts = self.datacenter.last_tick()
        return ts

    def reset_component(self, name, component):
        if component and hasattr(component, 'reset'):
            component.reset(self)
        elif not component:
            raise ValueError("You must set component: %s" % name)
        else:
            raise ValueError("Component %s does not have attr `reset`" % name)

    def reset(self):
        self._with_each_component(self.reset_component)
        return self

    def last_run_timestamp(self):
        if len(self.portfolio.timeline) > 0:
            return self.portfolio.timeline[-1]['time']

    def run(self, report=True):
        self.reset()
        self.mutate_datacenter_history(refresh=False)
        self._run_benchmarks()

        self.loop_index = 0
        self._before_trading_start()
        self.portfolio._load_order_groups()
        self._run_hook('before_trading_start')

        while True:
            self.loop_index += 1
            self._load_state_from_session()
            self._before_each_tick()

            ts = self._get_next_tick()
            self._on_each_tick()
            if not ts:
                break

            if (not self.last_run_timestamp() or
                    ts != self.last_run_timestamp()):
                self._run_hook('at_each_tick_start')

            while True:
                while True:
                    try:
                        event = self.events.get(block=False)
                    except EventQueueEmpty:
                        break
                    else:
                        if event is not None:
                            self._process_event(event)

                end = self._run_hook('at_each_tick_end')
                if end is None or end:
                    break

            self.portfolio.trading_session_tick_complete()
            self._save_state_in_session()
            if report:
                self._run_hook('before_tick_report')
                for reporter in self.reporters:
                    reporter.generate_tick_report()
            self._after_each_tick()
            if not self._should_reloop():
                break

        self.portfolio.trading_session_complete()

        if report:
            self._run_hook('before_summary_report')
            for reporter in self.reporters:
                reporter.generate_summary_report()
        self._run_hook('after_trading_end')
        self._after_trading_end()

    def _process_event(self, event):
        self._run_hook('after_any_event', event)

        if type(event) == TickUpdateEvent:
            if event.item.time == self.last_run_timestamp():
                return

            data = event.item.history
            self._current_time = event.item.time
            self.strategy.data = data
            self.strategy.tick = event.item

            self._run_hook('after_tick_update', event)
            self.portfolio.update_portfolio_value_at_tick(event)
            self.strategy.advice_investments_at_tick(event)
            self._run_hook('after_tick_update_done', event)

        elif type(event) == StrategyAdviceEvent:
            self._run_hook('after_strategy_advice', event)
            self.portfolio.record_advice_from_strategy(event)
            self.portfolio.place_order_after_advice(event)
            self._run_hook('after_strategy_advice_done', event)

        elif type(event) == OrderRequestedEvent:
            self._run_hook('after_order_placed', event)
            self.broker.create_order_after_placement(event)
            self._run_hook('after_order_placed_done', event)

        elif type(event) == OrderCreatedEvent:
            self._run_hook('after_order_created', event)
            order = self.broker.execute_order_after_creation(event)
            if order:
                self.portfolio.record_created_order(event, order)
                self.account.lock_cash_for_order_if_required(event, order)
            self._run_hook('after_order_created_done', event)

        elif type(event) == OrderPendingEvent:
            self._run_hook('after_order_pending', event)
            self.broker.execute_order_after_creation(event)
            self._run_hook('after_order_pending_done', event)

        elif type(event) == OrderFilledEvent:
            self._run_hook('after_order_filled', event)
            self.portfolio.record_filled_order(event)
            self.account.update_after_order_filled(event)
            self.portfolio.update_commission_paid(event)
            self._run_hook('after_order_filled_done', event)

        elif type(event) == OrderUnfilledEvent:
            self._run_hook('after_order_unfilled', event)
            self.account.update_after_order_unfilled(event)
            self.portfolio.record_unfilled_order(event)
            self._run_hook('after_order_unfilled_done', event)

        elif type(event) == OrderRejectedEvent:
            self._run_hook('after_order_rejected', event)
            self.account.update_after_order_rejected(event)
            self.portfolio.record_rejected_order(event)
            self._run_hook('after_order_rejected_done', event)
        self._run_hook('after_any_event_done')
        self.events.task_done()


class BacktestSession(TradingSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mode = 'backtest'

    def _get_next_tick(self):
        ts = None
        if self.datacenter._continue_backtest:
            ts = self.datacenter.next_tick()
        return ts


class PaperTradingSession(TradingSession):
    def __init__(self, *args, session=None, poll_frequency=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._mode = 'paper'
        self.session = session
        self.poll_frequency = poll_frequency

    @property
    def current_time(self):
        return datetime.now()

    @property
    def refresh_history(self):
        return self._refresh_history

    @refresh_history.setter
    def refresh_history(self, _value):
        raise ValueError("History is always refreshed in paper/live mode.")

    @property
    def start_time(self):
        return self._start_time

    @start_time.setter
    def start_time(self, value):
        raise ValueError("Cannot set start time in paper/live mode.")

    @property
    def end_time(self):
        return self._end_time

    @end_time.setter
    def end_time(self, value):
        raise ValueError("Cannot set end time in paper/live mode.")

    @property
    def session(self):
        return self._session

    @session.setter
    def session(self, value):
        self._session = value
        if self._session and self._session.strip():
            self._session_dir = join(
                self.root_dir, 'sessions', self._session)
            make_path(self._session_dir)

    @property
    def session_dir(self):
        return self._session_dir

    @property
    def poll_frequency(self):
        return self._poll_frequency

    @poll_frequency.setter
    def poll_frequency(self, value):
        self._poll_frequency = value

    def _after_each_tick(self):
        if self.poll_frequency:
            time.sleep(self.poll_frequency)  # sleep before re-looping

    def _should_reloop(self):
        return self.poll_frequency is not None

    def run(self, *args, **kwargs):
        try:
            super().run(*args, **kwargs)
        except Exception as e:
            self.notify_error(e)


class LiveTradingSession(PaperTradingSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mode = 'live'
        self._slippage = lambda: 0  # slippage is zero when live trading

    @property
    def slippage(self):
        return self._slippage

    @slippage.setter
    def slippage(self, slippage_fn):
        raise ValueError("Slippage is calculated inherently in live trading " +
                         "session. You do not need to set it manually.")
