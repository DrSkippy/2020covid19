import logging
import numpy as np

logfile = "/Users/drskippy/logs/covid.log"
logging.basicConfig(level=logging.DEBUG, filename=logfile)


def get_state_df(df, state, pos_key="positive"):
    if state == "*":
        key_list = ["date",
             pos_key,
             "negative",
             "pending"]
        if "totalTestResults" in df.columns:
            key_list.extend(
             ["totalTestResults",
             "death"]
            )
        # all states in the list, aggregated
        dfq = df.groupby('date', as_index=False)[key_list].sum()
        dfq["lastUpdateEt"] =  max(df["lastUpdateEt"])  # use most recent for everything
    else:
        # select data for the state
        dfq = df.loc[state == df["state"]].copy()
    # pull off last updated value
    last_update_date = dfq["lastUpdateEt"].values[0]
    dfq["daily_new_positive"] = dfq[pos_key].diff(1)
    return dfq, last_update_date

