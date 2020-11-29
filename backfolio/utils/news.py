import re, requests
import pandas as pd
from bs4 import BeautifulSoup
from os import join, expanduser, isfile


def get_binance_news(recent=False):
    path = join(expanduser("~"), ".backfolio", "data", "news", "binance.csv")
    articles = pd.DataFrame()
    articles = pd.read_csv(path, index_col=0) if isfile(path) else articles

    all_links = []
    url = "https://support.binance.com/hc/en-us/sections/115000202591-Latest-News"
    pages = [1] if recent else range(1, 101)
    #pages = ["%s?page=%d" % (url, i) for i in pages]
    for page in pages:
        print("%s?page=%s" % (url, page))
        page = requests.get("%s?page=%s" % (url, page))
        soup = BeautifulSoup(page.text, 'html.parser')
        links = [
            "https://support.binance.com%s" % a.get('href')
            for a in soup.find_all("a", class_="article-list-link")
        ]
        if len(links) == 0:
            break
        all_links += links
    all_links = list(set(all_links))

    scraped = [] if articles is None or articles.empty else articles.url.tolist(
    )
    remaining = [link for link in all_links if link not in scraped]

    new_articles = []
    for url in remaining:
        page = requests.get(url)
        if page.status_code == 404:
            continue
        page = BeautifulSoup(page.text, 'html.parser')
        if not page.find("time"):
            continue
        timestamp = pd.to_datetime(page.find("time").get("datetime"))
        title = page.find("h1", class_="article-title").text.strip()
        content = page.find("div", class_="article-body")
        new_articles.append(
            dict(url=url, timestamp=timestamp, title=title, content=content))

    df = pd.DataFrame.from_records(new_articles)
    if not df.empty:
        df['time'] = pd.to_datetime(df['timestamp'])
        df = df.drop(['timestamp'], axis=1).set_index('time')
    df = pd.concat([articles, df])
    df = df.sort_index(ascending=1)[['title', 'url', 'content']]

    delisted = df[df['title'].str.lower().str.contains('delist')]
    # delisted['coins'] = df['title'].str.lower()
    # delisted['coins'] = delisted['coins'].str.replace("binance will delist", "").str.replace("and", ",")
    # delisted['coins'] = delisted['coins'].str.upper().str.replace(r'\s+', '')

    z = []
    for row in delisted.to_records():
        m1 = re.findall(r'\b[A-Z]+\b', row.title)
        m2 = [
            x for x in
            re.sub(r'(binance|will|delist|and|,)', ' ', row.title,
                   flags=re.I).split(" ") if x
        ]
        ma = list(set(m1).union(set(m2)))

        for coin in ma:
            z.append(
                dict(coin=coin, time=row.time, url=row.url, title=row.title))
    delisted = pd.DataFrame.from_records(z).set_index('coin').sort_index()

    df.to_csv(path, index=True)
    return (delisted, df)
