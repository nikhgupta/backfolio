import math


def zscore(df, period, maximum=None):
    z = (df - df.rolling(period).mean()) / df.rolling(period).std()
    if maximum:
        z[z > maximum] = maximum
        z[z < -maximum] = -maximum
    return z


def zscore_reversion(df, maximum=None):
    z, PD = 0, [2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

    for p in PD:
        z += zscore(df, p) / p
    z /= math.fsum([1 / x for x in PD])

    if maximum:
        z[z > maximum] = maximum
        z[z < -maximum] = -maximum
    return z


def zscore_trend(df, maximum=None):
    z, PD = 0, [2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

    for p in PD:
        z += zscore(df, p) * p
    z /= math.fsum(PD)

    if maximum:
        z[z > maximum] = maximum
        z[z < -maximum] = -maximum
    return z


def zscore_avg(df, maximum=None):
    z, w, PD = 0, 0, [2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

    for idx, p in enumerate(PD):
        m = min(p, PD[len(PD) - 1 - idx])
        z += zscore(df, p) * m
        w += m
    z /= w

    if maximum:
        z[z > maximum] = maximum
        z[z < -maximum] = -maximum
    return z
