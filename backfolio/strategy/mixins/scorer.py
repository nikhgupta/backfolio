import numpy as np


class ScoringMixin(object):
    def __init__(self, *args, weighted=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.weighted = weighted

        # can be set externally/manually, e.g. via hedged strategies
        self.score_col = 'score'
        self.weight_col = 'weight'

    def transform_history(self, panel):
        """
        Ask the child strategy for an assigned score, and/or
        weight for each asset in new portfolio.

        `score` and `weight` are mutually exclusive, and `weight`
        takes preference.

        Therefore, if `weight`s are provided, `score` will be ignored,
        and the strategy will try to buy any asset with a positive weight,
        if the current equity allocation is less than the specified
        weight in the portfolio. Weights can be non-normalized.

        If `score` is provided (and not weights), assets are weighted
        equally. If `assets` is None, strategy will try to buy all assets
        with positive score, otherwise top N assets will be bought. In
        this scenario, weights are equally distributed.

        Look into `transform_tick_data` for an additional approach to
        specify `score` and `weight` at each tick.
        """
        panel = super().transform_history(panel)
        if hasattr(self, 'calculate_scores'):
            panel.loc[:, :, self.score_col] = self.calculate_scores(panel)
        if self.weighted and hasattr(self, 'calculate_weights'):
            panel.loc[:, :, self.weight_col] = self.calculate_weights(panel)
        return panel

    def transform_tick_data(self, data):
        """
        Allow strategy to transform tick data to provide score and/or weight
        for assets at each tick.
        """
        if hasattr(self, 'calculate_scores_at_each_tick'):
            scores = self.calculate_scores_at_each_tick(data)
            if scores is not None:
                data.loc[:, self.score_col] = scores

        if self.weighted:
            if hasattr(self, 'calculate_weights_at_each_tick'):
                weights = self.calculate_weights_at_each_tick(data)
                if weights is not None:
                    data.loc[:, self.weight_col] = weights

        data = super().transform_tick_data(data)
        return data

    def calculate_weights_at_each_tick(self, data):
        """
        Default implementation for assigning weights
        to positive/top scoring assets.

        This allows to easily switch a strategy from
        equal weights to weights based on score
        by adding `weighted=True` strategy parameter.
        """
        if self.weight_col in data.columns:
            return
        weights = data[self.score_col].copy()
        weights[weights < 0] = 0
        if self.assets:
            weights = weights.sort_values(ascending=False)
            weights.iloc[self.assets:] = 0
        return weights

    def sorted_data(self, data):
        """
        Sort tick data based on score, volume and closing price of assets.

        Child strategy can use this method to implement their own sorting.
        The top N assets for used for placing buy orders if `weight`
        column is missing from the assigned data.
        """
        if self.score_col in data.columns:
            return data.sort_values(
                by=[self.score_col, 'volume', 'close'],
                ascending=[False, False, True])
        else:
            return data

    def selected_assets(self, data):
        """
        By default:
            - Select all assets with positive weights,
              if asset weights are specified
            - Select all assets with positive score,
              if `assets` is not None
            - Select top N assets, if `assets` is specified.
        """
        data = super().selected_assets(data)
        if self.weighted and "weight" in data.columns:
            data = data[np.isfinite(data[self.weight_col])]
            data = data[data[self.weight_col] > 0]
        elif self.assets:
            data = self.sorted_data(data)
            data = data[np.isfinite(data[self.score_col])]
            data = data[data[self.score_col] > 0].head(self.assets)
        else:
            data = data[np.isfinite(data[self.score_col])]
            data = data[data[self.score_col] > 0]
        return data

    def rejected_assets(self, data, selected=None):
        """
        By default:
            - Reject all assets with zero,
              if asset weights are specified
            - Reject all assets with negative score,
              if `assets` is not None
            - Reject all but top N assets, if `assets` is specified.

        Be careful not to specify negative weights.
        """
        if self.weighted and "weight" in data.columns:
            data = data[data[self.weight_col] == 0]
        elif self.assets:
            data = self.sorted_data(data)
            data = data.tail(len(data) - self.assets)
        else:
            data = data[data[self.score_col] < 0]
        return super().rejected_assets(data)

    def before_strategy_advice_at_tick(self):
        halted = super().before_strategy_advice_at_tick()
        if not halted:
            if (self.score_col not in self.data.columns and
                    self.weight_col not in self.data.columns):
                return True
        return halted

    def set_required_equity_for_each_asset(self):
        if self.weight_col not in self.data.columns:
            self.data[self.weight_col] = 0
            self.data.loc[self.selected.index, self.weight_col] = 1
        self.data[self.weight_col] /= self.data[self.weight_col].sum()
        return self.data[self.weight_col]
