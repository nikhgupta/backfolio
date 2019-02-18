import pandas as pd


class RebalancingScheduleMixin(object):
    """
    Mixin that allows a strategy to add rebalancing based schedule, s.t.
    strategy is only asked for advice after every specified frequency.

    The actual rebalancing of assets is not done here.
    """
    def __init__(self, *args, rebalance=1, **kwargs):
        """
        :param rebalance: number of ticks in a rebalancing period
        """
        if not hasattr(self, '_state_fields'):
            raise ValueError("StateMixin is required.")

        super().__init__(*args, **kwargs)
        self._state_fields += ['last_rebalance']
        self.rebalance = rebalance

    def rebalance_required(self, data):
        """
        Check whether rebalancing is required at this tick or not.
        If `rebalance` is set as None, we will rebalance (trade) on
        each tick.

        Specifying `rebalance` as None means that we are not doing
        a time-based rebalancing. Assets are, instead, rebalanced
        based on Signals.

        Pending orders are cancelled at the start of each rebalancing
        tick.

        As a side effect, at the moment, you SHOULD cancel your open
        orders for an asset yourself if you are not doing time-based
        rebalancing.

        # FIXME: ensure Strategy is able to work with Signals as well,
        instead of just time-based rebalancing.
        """
        if not self.rebalance:
            return False
        time = self.tick.time
        timediff = pd.to_timedelta(self.datacenter.timeframe*self.rebalance)
        return (not self._last_rebalance or
                time >= self._last_rebalance + timediff)

    def before_strategy_advice_at_tick(self):
        halted = super().before_strategy_advice_at_tick()
        if not halted:
            if self.rebalance_required(self.data):
                self.broker.cancel_pending_orders()
            elif self.rebalance:
                halted = True
        return halted

    def after_strategy_advice_at_tick(self):
        super().after_strategy_advice_at_tick()
        self.set_state_for("last_rebalance", self.tick.time, save={})
