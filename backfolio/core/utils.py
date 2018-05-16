import string
import pathlib
import pandas as pd
from random import choice
from os.path import join, isfile


def fast_xs_values(df, key):
    """ Really fast implementation for accessing values of a row
    of a dataframe by its index """

    return df._data.fast_xs(df.index.get_loc(key)).tolist()


def fast_xs(df, key):
    """ Really fast implementation for accessing a row of a dataframe
    by its index and returning a dictionary for that row.

    This is faster (clocked at 16.4 µs ± 2.07 µs per loop) than doing:

        df.xs(ts).to_dict()  # 132 µs ± 16.4 µs per loop
        df.ix[ts].to_dict()  # 178 µs ± 22.2 µs per loop
        df.loc[ts].to_dict() # 248 µs ± 25.1 µs per loop

    """
    keys = df.columns
    vals = fast_xs_values(df, key)
    return dict([keys[idx], val] for idx, val in enumerate(vals))


def make_path(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def load_df(path, name, container, field, array=True):
    """ Load a dataframe of records into an array of dict from a given path,
    and set it as an attribute on container """
    path = join(path, "%s.csv" % name)
    if isfile(path):
        try:
            data = pd.read_csv(path, index_col=None)
            if data is not None:
                if 'time' in data.columns:
                    data['time'] = pd.to_datetime(data['time'])
                data = data[~data.duplicated(keep='last')]
                data = data.to_dict(orient='records')
                setattr(container, field, data if array else data[0])
        except pd.io.common.EmptyDataError:
            pass


def as_df(items, index=None, dupes='all'):
    """ Convert an array of dicts into a pandas dataframe """
    if len(items) == 0:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(items)
    if index and index in df.columns:
        df = df.set_index(index)
    elif index and 'index' in df.columns:
        df = df.set_index('index')
        df.index.name = index
    df = drop_duplicates(df, dupes == 'index')
    return df


def drop_duplicates(df, index=False):
    if index:
        return df[~df.index.duplicated(keep='last')]
    else:
        return df[~df.duplicated(keep='last')]


def save_df(path, name, data):
    """ Save an array of dicts into a csv file via pandas dataframe """
    if data is None:
        return
    path = join(path, '%s.csv' % name)
    as_df(data).to_csv(path, index=False)


def detect(items, pred):
    return next((i for i in items if pred(i)), None)


def detect_with_index(items, pred):
    return next((i for i in enumerate(items) if pred(i)), None)


def generate_id(name, inst, length=8):
    str = None
    set = string.ascii_uppercase + string.digits
    if not hasattr(inst, 'ids'):
        inst.__class__._ids = []
    while not str or str in inst._ids:
        str = ''.join([choice(set) for _ in range(length)])
    return str
