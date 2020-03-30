import pandas as pd
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

import logging

from covid_analysis.utility import *

logfile = "/Users/drskippy/logs/covid.log"
logging.basicConfig(level=logging.DEBUG, filename=logfile)

def _greater_than_and_changes(df, min_pos, pos_key, use_last_n_days=None):
    df = df.loc[df[pos_key] > min_pos].copy()
    if use_last_n_days is not None:
        df = df.iloc[-use_last_n_days:].copy()
    while df.positive.values[0] == df.positive.values[1]:
        df = df.iloc[1:].copy()
    return df

def get_state_doubling_df(df, state, zero_aligned=False, min_pos=10, pos_key="positive", use_last_n_days=None):
    df_res, last_update = get_state_df(df, state, pos_key)
    y = df_res[pos_key].values
    dt, l = doubling_time_in_days(y, use_last_n_days)
    df_res["exp_fit_line"] = np.exp(l)
    if zero_aligned:
        df_res = _greater_than_and_changes(df_res, min_pos, pos_key, use_last_n_days)
        df_res["log_positive"] = np.log(df_res[pos_key])
        initial_value = df_res.iloc[0]["log_positive"]
        df_res["log_positive"] = df_res["log_positive"] - initial_value
        df_res["days_since_{}".format(min_pos)] = range(len(df_res))
    return df_res, dt, last_update


def _fix_spurious_zeros(x):
    """
    NV as a zero after a nonzero value...
    assume leading zeros are removed
    """
    for i in range(len(x)):
        if x[i] == 0:
            logging.error("fixing spurious zero at index={}".format(i))
            x[i] = x[i-1]
    return x


def doubling_time_in_days(total_by_day, use_last_n_days=None):
    """
    assumes daily data points, only leading zeros
    """
    non_zero_total_by_day = np.trim_zeros(total_by_day)
    non_zero_total_by_day = _fix_spurious_zeros(non_zero_total_by_day)
    if use_last_n_days is not None:
        non_zero_total_by_day = non_zero_total_by_day[-int(use_last_n_days):]
    n_non_zero = len(non_zero_total_by_day)
    # model is linearized logs
    log_total_by_day = np.log(non_zero_total_by_day)

    # arbitrary time bases
    t = range(len(total_by_day))
    t_non_zero = t[-n_non_zero:]
    # linear fit and model line
    A = np.vstack([t_non_zero, np.ones(len(t_non_zero))]).T
    try:
        m, b = np.linalg.lstsq(A, log_total_by_day, rcond=None)[0]
        fit_line_points = m * np.array(t) + b
    except np.linalg.LinAlgError:
        fit_line_points = np.zeros((len(total_by_day)))
        m = 1e-10
    return -np.log(.5)/m, fit_line_points


def estimate_current_cases(new_by_day, resolution_time=10, hospitalized=.15, icu=0.04):
    if len(new_by_day) > resolution_time:
        s = np.sum(new_by_day[-resolution_time:])
    elif len(new_by_day) > 0:
        s = np.sum(new_by_day)
    else:
        s = 0
    try:
        res = (int(s), int(hospitalized * s), int(icu * s))
    except ValueError as e:
        res = (0,0,0)
    return res


def SIRModel(N=100, I0=1, R0=0, beta=0.3, gamma=0.1):
    # e.g. 5696 (thousand) for population of CO
    # print("R-nought={}".format(beta/gamma))
    # Total population, N.
    # Initial number of infected and recovered individuals, I0 and R0.
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0
    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
    # A grid of time points (in days)
    t = np.linspace(0, 160, 160)
    # The SIR model differential equations.
    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    # Initial conditions vector
    y0 = S0, I0, R0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T

    df = pd.DataFrame(data={"susceptible": S, "infected": I, "removed": R})
    return df


def SIRSSE(x, N, R0, c):
    beta, gamma, I0 = x
    dfp = SIRModel(N, I0, R0, beta, gamma)
    cp = dfp.infected.values + dfp.removed.values
    cp = cp[:len(c)]
    res = np.linalg.norm(c-cp)**2
    return res


def SIRFitter(c, N=350000,  x0=(0.3896, 0.08149, 0.1), R0=0):
    a = minimize(SIRSSE, x0, args=(N, R0, c), method="Powell")
    print(a)
    beta, gamma, I0 = a.x

    dfm = SIRModel(N=N, I0=I0, R0=R0, beta=beta, gamma=gamma)
    dfm.plot(figsize=[15, 6])

    actual_pos = np.zeros((len(dfm)))
    start_index = 0
    actual_pos[start_index: start_index + len(c)] += c

    dfm["actual_pos"] = actual_pos
    dfm["total_pos"] = dfm.infected + dfm.removed

    dfm.plot(y=["actual_pos", "total_pos"], figsize=[15, 6], ylim=[0, 1.2*np.max(c)])

