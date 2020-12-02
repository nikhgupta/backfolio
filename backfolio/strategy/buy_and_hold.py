from .base import BaseStrategy


class BuyAndHoldStrategy(BaseStrategy):
    """
    A simple buy and hold strategy that buys a new asset whenever a new
    asset is encountered in the tick data. We rebalance our portfolio on this
    tick to include this new asset, such that all assets always have equal
    weights in our portfolio.

    Assets that have been delisted have no impact on portfolio (other than
    zero returns from them) as per the strategy logic.

    We ensure that we, also, keep an equal weight of cash/primary currency.

    This portfolio strategy is different than a UCRP strategy in that rebalance
    is only done when a new asset is introduced on the exchange, which may take
    weeks/months, as opposed to UCRP where rebalance is done daily, regardless.
    """
    def __init__(self, min_comm_asset=3):
        super().__init__()
        self.min_comm_asset = min_comm_asset

    def advice_investments_at_tick(self, tick_event):
        """
        We sell our assets first so that we do have sufficient liquidity.
        Afterwards, we will issue a buy order.
        """

        n = (100.0 - self.min_comm_asset) / (1 + len(self.data))
        rebalanced = self.account.equity * n / 100
        equity = self.portfolio.equity_per_asset
        # print(rebalanced, self.account.equity)

        current = [
            self.datacenter.symbol_to_assets(k)[0] for k in self.data.index
        ]

        bought = [k for k, v in equity.items() if v > 0]
        new = [k for k in current if k not in bought]

        if not new:
            return

        # sell assets that have higher equity first
        # once we have cash available, buy assets that have lower equity
        for asset, asset_equity in equity.items():
            symbol = self.datacenter.assets_to_symbol(asset)
            if symbol not in self.data.index or asset_equity < rebalanced * 1.05:
                continue

            m = n
            if asset == self.context.commission_asset:
                m += self.min_comm_asset
            self.order_target_percent(symbol, m, 'MARKET', side='SELL')

        for asset, asset_equity in equity.items():
            symbol = self.datacenter.assets_to_symbol(asset)
            if symbol not in self.data.index or asset_equity > rebalanced * 0.95:
                continue

            m = n
            if asset == self.context.commission_asset:
                m += self.min_comm_asset
            self.order_target_percent(symbol, m, 'MARKET', side='BUY')
