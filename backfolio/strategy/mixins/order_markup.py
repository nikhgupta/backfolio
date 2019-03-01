class OrderMarkupMixin(object):
    """
    Mixin that allows a strategy to buy/sell assets at a price preferrable
    than the last tick's closing price.

    This mixin, also, sets functions that adjust markup for aggressive mode.
    """
    def __init__(self, *args,
                 markup_sell=[1.1,0.9,1.0], markdn_buy=[0.9,1.1,1.0],
                 markdn_buy_func=None, markup_sell_func=None, **kwargs):
        """
        :param markdn_buy: (in %) %ages of price to decrease buy orders by.
        :param markup_sell: (in %) %ages of price to increase sell orders by.
        :param markdn_buy_func: function that adjusts markdn_buy based on
            current iteration for aggressive mode
        :param markup_sell_func: function that adjusts markup_sell based on
            current iteration for aggressive mode
        """
        super().__init__(*args, **kwargs)
        self.markdn_buy = markdn_buy
        self.markup_sell = markup_sell

        if markdn_buy_func is not None:
            self.markdn_buy_func  = markdn_buy_func
        else:
            self.markdn_buy_func  = lambda curr, i: i*curr

        if markup_sell_func is not None:
            self.markup_sell_func = markup_sell_func
        else:
            self.markup_sell_func = lambda curr, i: (i-1)*0.5+curr

    def selling_prices(self, _symbol, data):
        """
        Array of limit prices at which an asset should be sold.
        MARKET order is used if returned prices are None.
        """
        if self.markup_sell is not None:
            price = data['price'] if 'price' in data else data['close']
            return [price*(1+x/100) for x in self.markup_sell]

    def buying_prices(self, _symbol, data):
        """
        Array of limit prices at which an asset should be bought.
        MARKET order is used if returned prices are None.
        """
        if self.markdn_buy is not None:
            price = data['price'] if 'price' in data else data['close']
            return [price*(1-x/100) for x in self.markdn_buy]

    def modify_order(self, asset, asset_data, direction='sell',
            amount=0, percent=100, iteration=1, relative=False):
        if hasattr(self, "modify_order_placement"):
            amount, percent = self.modify_order_placement(asset, asset_data,
                amount, percent, direction, iteration, relative=relative)
        return (amount, percent)

    def sell_asset(self, asset, asset_equity, asset_data=None, symbol=None,
                   amount=0, percent=100, orders=None, when=True,
                   relative=False, markup_multiplier=1):
        if not when:
            return

        if hasattr(self, "min_commission_asset_equity"):
            if asset == self.context.commission_asset:
                amount = max(self.min_commission_asset_equity, amount)

        symbol = symbol if symbol else self.datacenter.assets_to_symbol(asset)
        asset_data = asset_data if asset_data else fast_xs(self.data, symbol)

        N = asset_equity/self.account.equity*100
        amount = N - (N-amount)*(percent/100)

        orig = self.markup_sell
        if orig and markup_multiplier > 1 and hasattr(self, "markup_sell_func"):
            self.markup_sell = [self.markup_sell_func(curr, markup_multiplier)
                                for curr in orig]
        prices = self.selling_prices(symbol, asset_data)
        prices = prices[-orders:] if orders is not None else prices

        if prices:
            for price in prices:
                x = (amount-(0 if relative else N))/len(prices)
                self.order_percent(symbol, x, price, relative=True, side='SELL')
        else:
            self.order_percent(symbol, amount, side='SELL')
        self.markup_sell = orig


    def buy_asset(self, asset, asset_equity, asset_data=None, symbol=None,
                  amount=0, percent=100, when=True, relative=False,
                  markdn_multiplier=1):
        if not when or not amount:
            return

        amount = amount*percent/100
        symbol = symbol if symbol else self.datacenter.assets_to_symbol(asset)
        asset_data = asset_data if asset_data else fast_xs(self.data, symbol)

        N = asset_equity/self.account.equity*100
        diff = amount*self.account.equity/100 - asset_equity

        orig = self.markdn_buy
        if orig and markdn_multiplier > 1 and hasattr(self, 'markdn_buy_func'):
            self.markdn_buy = [self.markdn_buy_func(curr, markdn_multiplier)
                            for curr in orig]
        prices = self.buying_prices(symbol, asset_data)
        orders = None if diff > 1e-3 else 1
        prices = prices[-orders:] if orders is not None else prices
        self.markdn_buy = orig

        if hasattr(self, "min_commission_asset_equity"):
            min_comm = self.min_commission_asset_equity
            if asset == self.context.commission_asset:
                if asset_equity < min_comm*self.account.equity/100:
                    self.order_percent(symbol, min_comm, side='BUY')
                    amount -= min_comm

        if prices:
            for price in prices:
                x = (amount-(0 if relative else N))/len(prices)
                self.order_percent(symbol, x, price, side='BUY', relative=True)
        else:
            self.order_percent(symbol, amount, side='BUY')
