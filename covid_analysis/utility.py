import logging

import numpy as np

logging.basicConfig(level=logging.DEBUG, filename="/Users/drskippy/logs/covid.log")


def doubling_time_in_days(total_by_day):
    """
    assumes daily data points, only leading zeros
    """
    non_zero_total_by_day = np.trim_zeros(total_by_day)
    n_non_zero = len(non_zero_total_by_day)
    log_total_by_day = np.log(non_zero_total_by_day)
    t = range(len(total_by_day))
    t_non_zero = t[-n_non_zero:]
    A = np.vstack([t_non_zero, np.ones(len(t_non_zero))]).T
    try:
        m, b = np.linalg.lstsq(A, log_total_by_day, rcond=None)[0]
        line = m * np.array(t) + b
    except np.linalg.LinAlgError:
        line = np.zeros((len(total_by_day)))
        m = 1e10
    return -np.log(.5)/m, line


def current_cases(new_by_day, resolve_time=10, hosp=.15, icu=0.04):
    if len(new_by_day) > resolve_time:
        s = np.sum(new_by_day[-resolve_time:])
    elif len(new_by_day) > 0:
        s = np.sum(new_by_day)
    else:
        s = 0
    try:
        res = (int(s), int(hosp*s), int(icu*s))
    except ValueError as e:
        res = (0,0,0)
    return res

def get_state_df(df, state, pos_key = "positive"):
    if state == "*":
        dfq = df.groupby('date', as_index=False)[["date", pos_key]].sum()
        dfq["lastUpdateEt"] =  max(df["lastUpdateEt"])
    else:
        dfq = df.loc[state == df["state"]].copy()
    last_update = dfq["lastUpdateEt"].values[0]
    dfq["daily_new_positive"] = dfq[pos_key].diff(1)
    return dfq, last_update

def get_state_doubling_df(df, state, zero_aligned=False, min_pos=10, pos_key="positive"):
    dfq, last_update = get_state_df(df, state, pos_key)
    y = dfq[pos_key].values
    dt, l = doubling_time_in_days(y)
    dfq["exp_fit_line"] = np.exp(l)
    if zero_aligned:
        dfq = dfq.loc[dfq[pos_key] >= min_pos].copy()
        dfq["log_positive"] = np.log(dfq[pos_key])
        dfq["days_since_{}".format(min_pos)] = range(len(dfq))
    return dfq, dt, last_update

def actual_model_1(df, state, asymptomatic=0.4):
    dfq, last_update = get_state_df(df, state)
    # testing all and only symptoms
    dfq["_model_1_daily_new_positive"] = dfq.daily_new_positive * (1 + asymptomatic)
    dfq["model_1_positive"] = dfq._model_1_daily_new_positive.cumsum()
    dfq["model_1_positive"] += dfq.positive[0]
    return dfq

