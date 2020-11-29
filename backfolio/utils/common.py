import json
import hashlib
import requests
import backfolio as bf
import datetime
from bs4 import BeautifulSoup


def print_emphasized(message, color=31):
    print("\x1b[%sm[%s]: %s\x1b[0m" %
          (color, datetime.datetime.now(), message))


def print_error(msg):
    print_emphasized("[ERROR]:   %s" % msg)


def print_warning(msg):
    print_emphasized("[WARNING]: %s" % msg, color=33)


def print_info(msg):
    print_emphasized("[INFO]:    %s" % msg, color=32)


def print_backfolio_version():
    print_emphasized("Backfolio version: %s" % bf.version)


def fetch_json(url):
    return json.loads(
        BeautifulSoup(requests.get(url).content, "html.parser").prettify())


def strat_name(strat):
    """
    A unique name for the strategy which takes into account the parameters
    being passed. Provide the `strat` object to this function.

    Useful when saving data about a given strategy in a file.
    """
    name = strat.__class__.__name__
    dict = {}
    for key, val in strat.__dict__.items():
        try:
            json.dumps(val)
            dict[key] = val
        except TypeError:
            if key == "markdn_buy_func" or key == "markup_sell_func":
                dict[key] = [val(i + 1, 2) for i in range(10)]
    h = hashlib.md5(json.dumps(dict,
                               sort_keys=True).encode('utf-8')).hexdigest()
    return "%s/%s/%s" % (bf.__version__, name, h)
