#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `backfolio` package."""

import os
import pytest
import random
import pandas as pd


@pytest.fixture(scope='session')
def asset_data(tmpdir_factory):
    def rand():
        return random.random() * (-1 if random.random() > 0.5 else 1)

    def rand_walk(seed, init=1000, len=1000):
        series = [init]
        random.seed(seed)
        for i in range(len):
            series.append(rand() + series[-1])
        return series

    def rand_stock_data(seed, st=None, et=None, freq='1h', init=1000):
        if not st:
            st = "2017-07-01"
        if not et:
            et = "2018-04-01"

        df = pd.DataFrame()
        df['time'] = pd.date_range(st, et, freq=freq)
        df['open'] = pd.Series(rand_walk(seed, init, len(df['time'])))
        df['close'] = df['open'].shift(-1)
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + rand()/1000)
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - rand()/1000)
        return df.set_index('time')[:-1]

    def asset_data_for_symbol(symbol, **kwargs):
        fn = tmpdir_factory.mktemp('data').join('%s.csv' % symbol)
        if os.path.isfile(fn):
            return pd.read_csv(fn, index=0)
        else:
            data = rand_stock_data(symbol, **kwargs)
            data.to_csv(fn, index=True)
        return data

    return asset_data_for_symbol
