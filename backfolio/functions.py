import re
import math
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from os.path import join, expanduser, isfile


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


def alpharank(*dfs, weights=None):
    score = 0
    if not weights:
        weights = [1.0 for df in dfs]
    for idx, df in enumerate(dfs):
        score += weights[idx]*df.rank(axis=1, pct=True)
    return score/math.fsum(weights)


def averaged_indicator(src, func=None, periods=[2,3,5,8,13,21,34,55,89,144],
        freq=2, pr=-1.0, include_one=False, dropna=False, curve='average'):
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

    ind, wgt, src = 0, 0, src.copy()
    periods = [1] + periods if include_one else periods
    periods = sorted(list(set([int(round(freq*x, 0)) for x in periods])))

    for idx, period in enumerate(periods):
        mul = period
        if curve == 'distribution':
            mul = period if idx < (len(periods)+1)//2 else periods[len(periods)-idx-1]

        mul = abs(mul)**pr*np.sign(mul)
        if dropna:
            ind += src.apply(lambda x: func(x.dropna(), period))*mul
        else:
            ind += func(src, period)*mul
        wgt += mul
    return ind/wgt


def get_binance_news(recent=False, fetch=True):
    path = join(expanduser("~"), ".backfolio", "data", "news", "binance.csv")
    articles = pd.DataFrame()
    articles = pd.read_csv(path, index_col=0) if isfile(path) else articles
    new_articles = []

    if fetch:
        all_links = []
        url = "https://support.binance.com/hc/en-us/sections/115000202591-Latest-News"
        pages = [1] if recent else range(1, 101)
        #pages = ["%s?page=%d" % (url, i) for i in pages]
        for page in pages:
            print("%s?page=%s" % (url, page))
            page = requests.get("%s?page=%s" % (url, page))
            soup = BeautifulSoup(page.text, 'html.parser')
            links = ["https://support.binance.com%s" % a.get('href')
                     for a in soup.find_all("a", class_="article-list-link")]
            if len(links) == 0:
                break
            all_links += links
        all_links = list(set(all_links))

        scraped = [] if articles is None or articles.empty else articles.url.tolist()
        remaining = [link for link in all_links if link not in scraped]

        for url in remaining:
            print(url)
            page = BeautifulSoup(requests.get(url).text, 'html.parser')
            timestamp = pd.to_datetime(page.find("time").get("datetime"))
            title = page.find("h1", class_="article-title").text.strip()
            content = page.find("div", class_="article-body")
            new_articles.append(dict(url=url, timestamp=timestamp, title=title, content=content))

    df = pd.DataFrame.from_records(new_articles)
    if not df.empty:
        df['time'] = pd.to_datetime(df['timestamp'])
        df = df.drop(['timestamp'], axis=1).set_index('time')
    df = pd.concat([articles, df])
    df.index = pd.to_datetime(df.index)
    df = df.sort_index(ascending=1)[['title', 'url', 'content']]

    delisted = df[df['title'].str.lower().str.contains('delist')]

    z = []
    for row in delisted.to_records():
        m1 = re.findall(r'\b[A-Z]+\b', row.title)
        m2 = [x for x in re.sub(r'(binance|will|delist|and|,)', ' ', row.title,
                flags=re.I).split(" ") if x]
        ma = list(set(m1).union(set(m2)))

        for coin in ma:
            z.append(dict(coin=coin, time=row.time, url=row.url, title=row.title))
    delisted = pd.DataFrame.from_records(z).set_index('coin').sort_index()

    df.to_csv(path, index=True)
    return (delisted, df)
