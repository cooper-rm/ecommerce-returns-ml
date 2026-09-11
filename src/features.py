import pandas as pd


def past_rate(df, key, date, flag, prior_weight=50.0):
    """Past-only smoothed rate of ``flag`` within each ``key``, using strictly earlier dates.

    Within a key, the exclusive cumulative sum and count give the history before each row. Taking the
    first value of each ``(key, date)`` block freezes that history at the start of the day, so rows
    sharing a date never see each other. Empty history falls back to the global prior.

    Returns ``(rate, n)`` aligned to ``df.index``, where ``n`` is the number of prior observations.
    """
    d = df[[key, date, flag]].sort_values([key, date], kind="mergesort")
    g = d.groupby(key, sort=False)[flag]
    excl_sum = g.cumsum() - d[flag]
    excl_n = g.cumcount()
    day = [d[key], d[date]]
    hist_sum = excl_sum.groupby(day).transform("first")
    hist_n = excl_n.groupby(day).transform("first")
    prior = df[flag].mean()
    rate = (hist_sum + prior_weight * prior) / (hist_n + prior_weight)
    return rate.reindex(df.index), hist_n.reindex(df.index)
