import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize

import logging
import datetime

from covid_analysis.utility import *

logfile = "/Users/drskippy/logs/covid.log"
logging.basicConfig(level=logging.DEBUG, filename=logfile)


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
    assumes daily data points, only trims zeros
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
        m, b = 1e-10, 0
    return -np.log(.5)/m, fit_line_points, (m, b)


def get_state_doubling_df(df, state, key="positive", use_last_n_days=None):
    df_res = get_state_df(df, state)
    return get_doubling_df(df_res, key, use_last_n_days)


def get_doubling_df(df_res, key="positive", use_last_n_days=None):
    y = df_res[key].values
    dt, l, params = doubling_time_in_days(y, use_last_n_days)
    df_res["exp_fit_line"] = np.exp(l)
    return df_res, dt, params


def _greater_than_and_changes(df, min_pos, pos_key, use_last_n_days=None):
    df = df.loc[df[pos_key] > min_pos].copy()
    if use_last_n_days is not None:
        df = df.iloc[-use_last_n_days:].copy()
    while df.positive.values[0] == df.positive.values[1]:
        df = df.iloc[1:].copy()
    return df

def get_zero_aligned_log_positive(df, min_pos=10, key="positive", use_last_n_days=None):
    df_res = _greater_than_and_changes(df, min_pos, key, use_last_n_days)
    df_res["log_{}".format(key)] = np.log(df_res[key])
    initial_value = df_res.iloc[0]["log_{}".format(key)]
    df_res["log_{}".format(key)] = df_res["log_{}".format(key)] - initial_value
    df_res["days_since_{}".format(min_pos)] = range(len(df_res))
    return df_res

class CurrentCases:
    def __init__(self):
        pass

    def add_positive_estimate(self, df, params=None, suffix="ident", base="positive"):
        if params is None:
            params = {"a": 1}
        key = "positive_{}".format(suffix)
        df[key] = df[base] * params["a"]
        return df

    def add_death_estimate(self, df, params=None, suffix="ident", base="death"):
        if params is None:
            params = {"a": 1}
        key = "death_{}".format(suffix)
        df[key] = df[base] * params["a"]
        return df

    def add_hostiptalized_estimate(self, df, params=None, suffix="ident", base="new_daily_positive"):
        if params is None:
            params = {"a": .15, "rt": 10}
        key = "hospitalized_{}".format(suffix)
        df[key] = df[base].rolling(params["rt"]).sum() * params["a"]
        return df

    def add_icu_estimate(self, df, params=None, suffix="_ident", base="new_daily_positive"):
        if params is None:
            params = {"a": .04, "rt": 10}
        key = "icu_{}".format(suffix)
        df[key] = df[base].rolling(params["rt"]).sum() * params["a"]
        return df


class CurrentCasesFromDeaths(CurrentCases):

    def add_positive_estimate(self, df, params=None, suffix="fromdeath", base="daily_new_death"):
        if params is None:
            params = {"a": 200, "shift": 10}
        key = "daily_new_positive_{}".format(suffix)
        df[key] = df[base] * params["a"]
        key2 = "positive_{}".format(suffix)
        df[key2] = df[key].cumsum()
        self.fit_series = df[key2].values.copy()
        self.shift = params["shift"]
        df[key2] = df[key2].shift(-params["shift"], fill_value=0)
        return df


class CurrentCasesUndercount(CurrentCases):

    def add_positive_estimate(self, df, params=None, suffix="undercount", base="daily_new_positive"):
        if params is None:
            params = {"a": 0.8}
        key = "daily_new_positive_{}".format(suffix)
        df[key] = df[base] * (1 + params["a"])
        key2 = "positive_{}".format(suffix)
        df[key2] = df[key].cumsum()
        self.fit_series = df[key2].values
        self.shift = 0
        return df


class ModelProjection:
    def __init__(self):
        self.d_day = pd.Timedelta(days=1)

    def project(self, df, days, params=None):
        if params is None:
            params = {"last_n_days": 10}
        # TODO finish this

        df_res, dt, params = get_doubling_df(df, key="positive", use_last_n_days=params["last_n_days"])
        dfex = 1
        df_res = df.append(dfex, ignore_index=True)
        return df_res


class SIR:
    def __init__(self):
        pass

    def SIRModel(self, N=100, I0=1, R0=0, beta=0.3, gamma=0.1):
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


    def SIRSSE(self, x, N, R0, c):
        beta, gamma, I0 = x
        dfp = self.SIRModel(N, I0, R0, beta, gamma)
        cp = dfp.infected.values + dfp.removed.values
        cp = cp[:len(c)]
        res = np.linalg.norm(c-cp)**2
        return res


    def SIRFitter(self, c, N=350000,  x0=(0.3896, 0.08149, 0.1), R0=0):
        a = minimize(self.SIRSSE, x0, args=(N, R0, c), method="Powell")
        print(a)
        beta, gamma, I0 = a.x

        dfm = self.SIRModel(N=N, I0=I0, R0=R0, beta=beta, gamma=gamma)
        dfm.plot(figsize=[15, 6])

        actual_pos = np.zeros((len(dfm)))
        start_index = 0
        actual_pos[start_index: start_index + len(c)] += c

        dfm["actual_pos"] = actual_pos
        dfm["total_pos"] = dfm.infected + dfm.removed

        dfm.plot(y=["actual_pos", "total_pos"], figsize=[15, 6], ylim=[0, 1.2*np.max(c)])
        return a.x

class SIR4:
    def __init__(self):
        pass

    def SIRModel(self, N=100, I0=1, R0=0, beta0=0.3, alpha=-0.3, gamma=0.1):
        # e.g. 5696 (thousand) for population of CO
        # print("R-nought={}".format(beta/gamma))
        # Total population, N.
        # Initial number of infected and recovered individuals, I0 and R0.
        # Everyone else, S0, is susceptible to infection initially.
        S0 = N - I0 - R0
        # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
        # A grid of time points (in days)
        t = np.linspace(0, 160, 260)
        # The SIR model differential equations.
        def deriv(y, t, N, alpha, gamma):
            S, I, R, beta = y
            dSdt = -beta * S * I / N
            dIdt = beta * S * I / N - gamma * I
            dRdt = gamma * I
            dbetadt = - alpha
            #dbetadt = - np.log(alpha) * alpha ** t
            return dSdt, dIdt, dRdt, dbetadt

        # Initial conditions vector
        y0 = S0, I0, R0, beta0
        # Integrate the SIR equations over the time grid, t.
        ret = odeint(deriv, y0, t, args=(N, alpha, gamma))
        S, I, R, beta = ret.T

        df = pd.DataFrame(data={"susceptible": S, "infected": I, "removed": R, "beta": beta})
        return df


    def SIRSSE(self, x, N, R0, c):
        alpha, beta0, gamma, I0 = x
        dfp = self.SIRModel(N, I0, R0, beta0, alpha, gamma)
        cp = dfp.infected.values + dfp.removed.values
        cp = cp[:len(c)]
        res = np.linalg.norm(c-cp)**2
        return res


    def SIRFitter(self, c, N=350000,  x0=(1, 0.33, 0.08, 0.1), R0=0):
        a = minimize(self.SIRSSE, x0, args=(N, R0, c), method="Powell")
        print(a)
        alpha, beta0, gamma, I0 = a.x

        dfm = self.SIRModel(N=N, I0=I0, R0=R0, beta0=beta0, alpha=alpha, gamma=gamma)
        dfm.plot(figsize=[15, 6])

        actual_pos = np.zeros((len(dfm)))
        start_index = 0
        actual_pos[start_index: start_index + len(c)] += c

        dfm["actual_pos"] = actual_pos
        dfm["total_pos"] = dfm.infected + dfm.removed

        dfm.plot(y=["actual_pos", "total_pos"], figsize=[15, 6], ylim=[0, 1.2*np.max(c)])
        return a.x

def period_factor_plot(dfw, code="CHN", window_size=10, resolution_time=10, ylimit=7):
    dfq = get_state_df(dfw, code)
    try:
        state_name = dfq.Entity.values[0]
    except AttributeError:
        if code == "*":
            state_name = "US"
        else:
            state_name = code
    start_date, end_date = dfq.date.min(), dfq.date.max()
    delta_t = pd.Timedelta(days=1)
    days = int((end_date - start_date).days)
    dtv, dtt = [], []
    for i in range(days - window_size + 2):
        sdt = start_date + i * delta_t
        edt = sdt + window_size * delta_t
        _df = dfq.loc[(dfq.date >= sdt) & (dfq.date < edt)].copy()
        dfa, dt, params = get_doubling_df(_df, use_last_n_days=window_size)
        dtv.append(dt)
        dtt.append(_df.date.values[-1])
    plt.figure(figsize=[7, 2])
    plt.ylim((0, min([100, 1.1 * max(dtv)])))
    plt.plot(dtt, dtv, "*r-")
    plt.title("{} Doubling Period ({} day moving window)".format(state_name, window_size))
    plt.ylabel("Doubling Period")
    plt.xlabel("Date")
    plt.show()
    # by ratio
    plt.figure(figsize=[7, 2])
    plt.plot(dtt, np.array(dtv) / resolution_time, "*b-")
    plt.plot(dtt, np.ones(len(dtt)), "g")
    plt.fill_between(dtt, np.ones(len(dtt)) * 3, np.ones(len(dtt)) * 5, where=np.ones(len(dtt)), color="yellow",
                     alpha=0.1)
    plt.fill_between(dtt, np.ones(len(dtt)) * 5, np.ones(len(dtt)) * ylimit, where=np.ones(len(dtt)), color="yellow",
                     alpha=0.05)
    plt.fill_between(dtt, np.zeros(len(dtt)), np.ones(len(dtt)) * 3, where=np.ones(len(dtt)), color="red", alpha=0.1)
    plt.title("{} Ratio ({} day moving average)".format(state_name, window_size))
    plt.ylim((0, ylimit))
    plt.ylabel("Doubling Period/\nRecovery Time")
    plt.xlabel("Date")
    plt.show()