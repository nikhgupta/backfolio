from ...core.utils import fast_xs


class OrderSizeMixin(object):
    """
    Mixin that imposes various conditions on the strategy, so as to make the
    backtest behave more closely to real-life practical applications.

    Order size is heavily limited based on average trading volume of asset,
    as well as on other basis.
    """
    def __init__(self, *args, max_order_size=0, min_order_size=0.001,
                 flow_period=1, flow_multiplier=2.5,
                 assets=3, max_assets=13, **kwargs):
        """
        :param min_order_size: Minimum allowed order size in base asset.
        :param max_order_size: Maximum allowed order size in base asset
            (surpassed if flow is more than this amount). Set to `None`
            to not limit order size using this. To completely, disable
            order size limiting, also set `flow_multiplier` to `0`.
        :param flow_period: How many ticks to consider for flow based
            max order sizing?
        :param flow_multiplier: (in %) %age of flow used. Flow is the
            average BTC flow for that asset in given period. Set to `0`
            to disable flow based order size limiting.
        :param assets: Number of assets strategy should buy.
        :param max_assets: If cash allows, try increasing assets
            to buy gradually uptil this value. This will only be
            done, when order size is limited using `max_order_size`
            or flow control.
        """
        super().__init__(*args, **kwargs)
        self.assets = assets
        self.max_assets = max_assets
        self.min_order_size = min_order_size
        self.max_order_size = max_order_size
        self.flow_period = flow_period
        self.flow_multiplier = flow_multiplier

    def transform_history(self, panel):
        """
        Calculate Flow for each point in time for each asset.
        """
        panel = super().transform_history(panel)
        panel.loc[:, :, 'flow'] = (
            panel[:, :, 'close'] * panel[:, :, 'volume']).rolling(
            self.flow_period).mean()
        return panel

    def order_percent(self, symbol, amount, price=None, ignore_size=False,
                      max_cost=None, side=None, relative=None):
        """
        Place a MARKET/LIMIT order for a symbol for a given percent of
        available account capital.

        We calculate the flow of BTC in last few ticks for that asset.
        This combined with overall max_order_size places an upper bound
        on the order cost.

        If a price is specified, a LIMIT order is issued, otherwise MARKET.
        If `max_cost` is specified, order cost is, further, limited by that
        amount.
        """
        if symbol in self.data.index:
            max_flow = fast_xs(self.data, symbol)['flow']*self.flow_multiplier/100
        else:
            max_flow = 1
        if self.max_order_size:
            max_flow = max(max_flow, self.max_order_size)
        max_cost = min(max_cost, max_flow) if max_cost else max_flow
        max_cost = None if ignore_size else max_cost
        args = ('MARKET', None, max_cost, side)
        if price:
            args = ('LIMIT', price, max_cost, side)
        if relative:
            self.order_relative_target_percent(symbol, amount, *args)
        else:
            self.order_target_percent(symbol, amount, *args)

    def before_strategy_advice_at_tick(self):
        """
        Increase number of assets being traded if we have surplus cash,
        gradually. This function is called at each tick before any
        advices are given out to ensure we are using max available
        assets on each tick.

        OPTIMIZE: (old legacy code) This can be optimized further.
        """
        halted = super().before_strategy_advice_at_tick()
        if not halted:
            if self.assets:
                limited = self.max_order_size or self.flow_multiplier
                if limited and self.assets < self.max_assets:
                    equity = self.account.equity
                    if (equity > self.assets**(self.assets**0.2)):
                        self.assets = min(self.max_assets, self.assets+1)
        return halted

    def before_trading_start(self):
        """
        Ensure that broker rejects any orders with cost
        less than the `min_order_size` specified for this
        strategy.

        You can, similarily, set `max_order_size` for the
        broker, which sets a hard limit for the max order
        size. However, we will be using a soft limit on that
        and use flow based max order sizing. The reason for
        using a max order size is that for shorter TFs such
        as `1h`, an abruptly large order will remain unfilled
        in live mode, while does get filled when backtesting,
        resulting in abnormally high returns of strategy.

        You can, also, limit the max_position_held for an asset
        in a similar way. This ensures that buy orders are
        limited such that the new order will not increase the
        equity of the asset beyond max_position_held, at the
        current prices.

        Finally, NOTE that order sizing is applicable on both BUY
        and SELL orders.

        For example, binance rejects any order worth less
        than 0.001BTC. This will be useful to reject such
        orders beforehand in backtest as well as live mode.
        """
        super().before_trading_start()
        self.broker.min_order_size = self.min_order_size
