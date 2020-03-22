import pandas as pd
import requests
import logging

logging.basicConfig(level=logging.DEBUG, filename="/Users/drskippy/logs/covid.log")

url = "https://covidtracking.com/api/{}"
logging.info("using url={}".format(url))


def get_state_information():
    res1 = requests.get(url.format("states"))
    res1j = res1.json()
    logging.info("state information retrieved={}".format(len(res1j)))
    return res1j


def get_state_daily():
    res = requests.get(url.format("states/daily"))
    resj = res.json()
    logging.info("daily information retrieved={}".format(len(resj)))
    return resj


def get_ranked_state_dataframe(state_data_json):
    dfs = pd.DataFrame.from_dict(state_data_json)
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


def get_state_daily_dataframe(state_daily_json):
    df = pd.DataFrame.from_dict(state_daily_json)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df


def get_joined_dataframe(df_daily, df_state):
    df = df_daily.join(df_state[["order", "lastUpdateEt"]], on="state", how="outer")
    df = df.sort_values(by=["order", "date"], ignore_index=True)
    return df


def get_dataset():
    df_state, states_list = get_ranked_state_dataframe(get_state_information())
    df_daily = get_state_daily_dataframe(get_state_daily())
    df = get_joined_dataframe(df_daily, df_state)
    print(df.head())
    print(df.describe())
    return df, states_list


if "__main__" == __name__:
    get_dataset()
