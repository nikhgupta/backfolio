from .rebalance_on_score import RebalanceOnScoreStrategy


# TODO: buy commission asset if we get low on it
class EwUCRPStrategy(RebalanceOnScoreStrategy):
    """
    A simple strategy which invests our capital equally amongst all assets
    available on an exchange, and rebalances it on each tick.

    In principle, a EwUCRP (Equally-weighted Uniformly Constant Rebalanced
    Portfolio) is hard to beat, which makes this strategy an excellent
    benchmark for our use cases.

    Assets that have been delisted are removed from our holdings.

    This portfolio strategy is different than the Buy and Hold strategy above,
    in that rebalance is done daily here as opposted to when a new asset is
    introduced on the exchange.
    """
    def calculate_scores(self, panel):
        panel.loc[:, :, 'score'] = 1
        return panel.loc[:, :, 'score']

    def selected_assets(self, data):
        return data[data['score'] == 1]

    def rejected_assets(self, data):
        return data[data['score'] == 0]
