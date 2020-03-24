import pandas as pd
import numpy as np
import requests
import logging
import time
import datetime
import csv

logfile = "/Users/drskippy/logs/covid.log"
logging.basicConfig(level=logging.DEBUG, filename=logfile)

url = "https://covidtracking.com/api/{}"
logging.info("using url={}".format(url))

data_file = "/Users/drskippy/data/{}_{}.csv"


def get_state_description():
    success = False
    while not success:
        res = requests.get(url.format("states"))
        if res.status_code == requests.codes.ok:
            success = True
        else:
            print("sleeping 5 s...")
            time.sleep(5)
    res_json = res.json()
    logging.info("state descriptions retrieved={}".format(len(res_json)))
    return res_json


def get_state_daily():
    success = False
    while not success:
        res = requests.get(url.format("states/daily"))
        if res.status_code == requests.codes.ok:
            success = True
        else:
            print("sleeping 5 s...")
            time.sleep(5)
    res_json = res.json()
    logging.info("daily information retrieved={}".format(len(res_json)))
    return res_json


def get_ranked_state_description_df(state_description_json):
    dfs = pd.DataFrame.from_dict(state_description_json)
    dfs["lastUpdateEt"] = dfs["lastUpdateEt"].apply(lambda x: "2020/{}".format(x))
    dfs["checkTimeEt"] = dfs["checkTimeEt"].apply(lambda x: "2020/{}".format(x))
    dfs["lastUpdateEt"] = pd.to_datetime(dfs["lastUpdateEt"], format="%Y/%m/%d %H:%M")
    dfs["checkTimeEt"] = pd.to_datetime(dfs["checkTimeEt"], format="%Y/%m/%d %H:%M")
    dfs = dfs.sort_values(by=["positive"], ignore_index=True, ascending=False)
    dfs.reset_index(inplace=True)
    states_in_order = dfs.state.unique()
    dfs = dfs.rename(columns={"index": "order"})
    dfs = dfs.set_index("state")
    return dfs, states_in_order


def get_state_daily_df(state_daily_json):
    df = pd.DataFrame.from_dict(state_daily_json)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df


def get_joined_dataframe(df_daily, df_state):
    df = df_daily.join(df_state[["order", "lastUpdateEt"]], on="state", how="outer")
    df = df.sort_values(by=["order", "date"], ignore_index=True)
    return df


def save_data(df, s):
    dts = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H%M")
    df.to_csv(data_file.format(dts, "state_daily_data"))
    wrtr = csv.writer(open(data_file.format(dts, "state_rank"), "w"))
    wrtr.writerow(s.tolist())
    return (data_file.format(dts, "state_daily_data"), data_file.format(dts, "state_rank"))


def get_dataset_df_from_file(fn=(None, None)):
    df = pd.read_csv(fn[0])
    df["lastUpdateEt"] = pd.to_datetime(df["lastUpdateEt"], format="%Y-%m-%d %H:%M:%S")
    df["dateChecked"] = pd.to_datetime(df["dateChecked"], format="%Y-%m-%d %H:%M:%S")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S")
    del(df["Unnamed: 0"])
    rdr = csv.reader(open(fn[1], "r"))
    sl = [r for r in rdr][0]
    return df, sl


def get_dataset_df():
    df_description, state_ordered_list = get_ranked_state_description_df(get_state_description())
    df_daily = get_state_daily_df(get_state_daily())
    df = get_joined_dataframe(df_daily, df_description)
    #print(df.head())
    print(df.describe())
    return df, state_ordered_list


if "__main__" == __name__:
    df, sl = get_dataset_df()
    print(save_data(df, sl))

