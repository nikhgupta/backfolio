class MoneyManagementMixin(object):
    def __init__(self, *args, reserved_cash=0,
            min_commission_asset_equity=3, **kwargs):
        """
        :param min_commission_asset_equity: (in %) capital to keep in
            commission asset
        :param reserved_cash: (in %) amount of cash wrt total account equity
            to keep in reserve at all times.
        """
        super().__init__(*args, **kwargs)
        self.min_commission_asset_equity = min_commission_asset_equity
        self.reserved_cash = reserved_cash

    def replenish_commission_asset_equity(self, asset_eq, at=1):
        account_eq = self.account.equity
        comm_eq = asset_eq[self.context.commission_asset]
        comm_sym = self._symbols[self.context.commission_asset]
        min_comm = self.min_commission_asset_equity
        if comm_eq < account_eq/100*min_comm*at:
            self.order_percent(comm_sym, min_comm, side='BUY')

    def set_required_equity_for_each_asset(self):
        multiplier = self.account.equity
        multiplier *= (1 - self.min_commission_asset_equity/100)
        if self.reserved_cash:
            multiplier *= (1 - self.reserved_cash/100)
        return multiplier * super().set_required_equity_for_each_asset()

    def transform_order_calculation(self, advice, cost, quantity, price):
        """
        Hook into broker, before it creates an order for execution,
        to override the final order cost, quantity or price.
        Supplied `price` is limit price for LIMIT orders, else None.

        You should return a tuple consisting of new order cost,
        quantity and LIMIT price. If cost, quantity and price are all changed,
        it is the quantity that will be recalculated to fit cost and price.

        Do NOT set price for MARKET orders.
        You can return `(0, 0, 0)` to not place this order.
        """
        n = (1-self.min_commission_asset_equity/100)
        if cost > 0 and cost > self.account.free[advice.base] * n:
            cost = self.account.free[advice.base] * n

        # if the equity vs quantity calculation, messes up our
        # ordering side, ensure that we still rebalance, but
        # we use the correct LIMIT price this time, but
        # we do this only when the cost of order is less than
        # 3% of our account equity.
        if advice.is_limit and advice.asset != self.context.commission_asset:
            th = self.account.equity*self.min_commission_asset_equity/100
            if advice.is_buy and cost < 0 and abs(cost) < th:
                advice.side = "SELL"
                if price:
                    price = self.selling_prices(
                        advice.symbol(self), {"price": advice.last_price})[0]

            elif advice.is_sell and cost > 0 and abs(cost) < th:
                advice.side = "BUY"
                if price:
                    price = self.buying_prices(
                        advice.symbol(self), {"price": advice.last_price})[0]
        return (cost, quantity, price)
