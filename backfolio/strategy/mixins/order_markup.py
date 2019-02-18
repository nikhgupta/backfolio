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
