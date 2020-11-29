import math
from .compute import stdev, rolling, rebase


def beta(src, mkt, tp, min_periods=None):
    reta = src.pct_change()
    retm = mkt.pct_change()
    deva = stdev(reta, tp, min_periods)
    devm = stdev(retm, tp, min_periods)
    beta = rolling(reta, tp, min_periods).corr(retm)
    beta = beta * deva.div(devm, axis='index')
    return beta


def alpha(src, mkt, atp, btp):
    b = beta(src, mkt, btp)
    s = src.pct_change(atp)
    m = mkt.pct_change(atp)
    a = s - b.mul(m, axis='index')
    return a


def market_index(panel):
    close = panel[:, :, 'close']
    high = panel[:, :, 'high']
    low = panel[:, :, 'low']
    volume = panel[:, :, 'volume']
    multiplier = ((close - low) - (high - close)) / (high - low)
    mkt = ((multiplier * volume).fillna(0).cumsum() * close).sum(axis=1)
    mkt = rebase(mkt - mkt.min())
    return mkt


def alpharank(*dfs, weights=None):
    score = 0
    if not weights:
        weights = [1.0 for df in dfs]
    for idx, df in enumerate(dfs):
        score += weights[idx] * df.rank(axis=1, pct=True)
    return score / math.fsum(weights)


def averaged_indicator(src,
                       func=None,
                       periods=[2, 3, 5, 8, 13, 21, 34, 55, 89, 144],
                       freq=1,
                       pr=1.0,
                       include_one=False):
    '''
    A function to get average value of an indicator over time.
    For focussing on recent values of indicator, use negative `pr` values.
    For focussing on long term values of indicator, use positive `pr` values.
    For equally weighting indicator values over time, use `pr` value of 0.

    Common use case:
        for MR:
            averaged_indicator(close, lambda x,p: x.rolling(p).mean()/x - 1,
                freq=2, pr=-1.0)
    '''
    if not func:
        raise ValueError("You must provide indicator function.")
    if include_one:
        periods = [1] + periods
    ind, wgt, periods = 0, 0, set([int(round(freq * x, 0)) for x in periods])
    for pd in periods:
        ind += func(src, pd) * (pd**pr)
        wgt += pd**pr
    return ind / wgt
