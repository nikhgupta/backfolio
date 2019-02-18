import pandas as pd


class BlacklistedMixin(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blacklisted = None

    def before_trading_start(self):
        super().before_trading_start()
        self.set_blacklisted()

    def set_blacklisted(self, df=None):
        """
        Hook to set blacklisted symbols.

        Blacklisted symbols are exited with a single Market order which ignores
        volume size as soon, as they are first encountered.

        Note that backtest does not account for this gracefully, as in
        real world scenario a blacklisted asset (e.g. asset delisting)
        will often be accompanied with a large sell-off - accompanied with our
        single large market order - it will result in high slippage,
        but backtest considers normal slippage.

        Must return a dataframe with:
        - Time (as column) when a symbol was blacklisted
        - Asset (as index) which was blacklisted
        """
        df = pd.DataFrame(columns=['time', 'asset']) if df is None else df
        if df is not None:
            df['symbol'] = df['asset'].apply(
                lambda x: self.datacenter.assets_to_symbol(x))
            df = df.set_index('symbol')
        self.blacklisted = df

    def before_strategy_advice_at_tick(self):
        halted = super().before_strategy_advice_at_tick()
        # symbols that are banned at this moment onwards due to being blacklisted
        self.banned = self.blacklisted[self.blacklisted['time']<=self.tick.time]
        return halted

    def selected_assets(self, data):
        data = data[~data.index.isin(self.banned.index)]
        data = super().selected_assets(data)
        return data

    def sell_blacklisted_assets(self, equity):
        for asset, asset_equity in equity.items():
            symbol = self._symbols[asset]
            if (symbol in self.banned.index and symbol in self.data.index and
                    asset_equity >= 1e-3):
                asset_data = fast_xs(self.data, symbol)
                self.order_percent(symbol, 0, side='SELL', ignore_size=True)
