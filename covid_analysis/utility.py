import logging

logfile = "/home/scott/log/covid.log"
logging.basicConfig(level=logging.DEBUG, filename=logfile)


def _create_aggregate_df(df, aak=None):
    key_list = ["positive", "daily_new_positive", "death", "daily_new_death", "tests"]
    if aak is not None:
        key_list.extend(aak)
    dfq = df.groupby(['date'], as_index=False)[key_list].sum()
    return dfq


def get_state_df(df, state, additional_aggregation_keys=None):
    if state == "*":
        dfq = _create_aggregate_df(df, additional_aggregation_keys)
    else:
        dfq = df.loc[state == df["state"]].copy()
    # pull off last updated value
    return dfq
