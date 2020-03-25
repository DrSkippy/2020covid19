import pandas as pd
import numpy as np
import requests
import logging
import time
import datetime
import csv

logfile = "/Users/drskippy/logs/covid.log"
logging.basicConfig(level=logging.DEBUG, filename=logfile)

url = "https://ourworldindata.org/"
logging.info("using url={}".format(url))

def get_dataset_df_from_file(fn="/Users/drskippy/Working/2020covid19/data/total-cases-covid-19.csv"):
    df = pd.read_csv(fn)
    # Entity,Code,Year,Total confirmed cases of COVID-19 (cases)
    mapper = {
        "Code": "state",
        "Total confirmed cases of COVID-19 (cases)": "positive"
    }
    df = df.rename(columns=mapper)
    epoch = datetime.datetime(2000,1,1,0,0,0)
    dday = datetime.timedelta(days=1)
    df["date"] = df.Year.apply(lambda x: epoch+x*dday)
    df["lastUpdateEt"] = datetime.datetime.utcnow()
    sl = df.state.unique()
    return df, sl

if "__main__" == __name__:
    df, sl = get_dataset_df_from_file()
    print(df.describe())

