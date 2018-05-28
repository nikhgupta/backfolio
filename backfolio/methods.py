def rolling(src, tp, min_periods=None):
    if not min_periods:
        min_periods = tp
    return src.rolling(window=tp, min_periods=min_periods)


def stdev(src, tp, min_periods=None):
    return rolling(src, tp, min_periods).std()


def rebase(series):
    return series/series.dropna().iloc[0]


def rank(src):
    return src.rank(pct=True, axis=0)


def ts_rank(src, tp, min_periods=None):
    def rollrank(data):
        return (1+data.argsort().argsort()[-1])/len(data)
    return rolling(src, tp, min_periods).apply(rollrank)


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
    multiplier = ((close-low)-(high-close))/(high-low)
    mkt = ((multiplier*volume).fillna(0).cumsum()*close).sum(axis=1)
    mkt = rebase(mkt - mkt.min())
    return mkt
