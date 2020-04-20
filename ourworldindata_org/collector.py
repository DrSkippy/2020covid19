import pandas as pd
import logging
import datetime
import csv

logfile = "/Users/drskippy/logs/covid.log"
logging.basicConfig(level=logging.DEBUG, filename=logfile)

url = "https://ourworldindata.org/"
in_filename_cases = "/Users/drskippy/Working/2020covid19/data/total-cases-covid-19.csv"
in_filename_deaths = "/Users/drskippy/Working/2020covid19/data/total-daily-covid-deaths.csv"

logging.info("using url={}".format(url))

def get_dataset_df_from_files(fnc=in_filename_cases, fnd=in_filename_deaths):

    df_case = pd.read_csv(fnc, quoting=csv.QUOTE_MINIMAL)
    df_death = pd.read_csv(fnd, quoting=csv.QUOTE_MINIMAL)

    # Entity,Code,Date,Total confirmed cases of COVID-19 (cases)
    logging.debug("case header={}".format(df_case.columns))
    # Entity,Code,Date,Total confirmed deaths (deaths),Daily new confirmed deaths (deaths)
    logging.debug("death header={}".format(df_death.columns))

    df_death = df_death.set_index(keys=["Entity", "Code", "Date"])
    df = df_case.join(df_death, on=["Entity", "Code", "Date"], how="inner", rsuffix="_fromdeath")

    # required fields: date, state, positive, daily_now_positive, death, daily_new_death, last_update, tests
    mapper = {
        "Code": "state",
        "Total confirmed cases of COVID-19 (cases)": "positive",
        "Total confirmed deaths due to COVID-19 (deaths)": "death"
             }

    df = df.rename(columns=mapper)
    df["date"] = pd.to_datetime(df.Date.apply(lambda x: x.strip().lower()), format="%b %d, %Y")
    df["last_update"] = datetime.datetime.utcnow()
    df["daily_new_positive"] = df.positive.diff(1)
    df["daily_new_death"] = df.death.diff(1)
    sl = df.state.unique()
    return df, sl


if "__main__" == __name__:
    df, sl = get_dataset_df_from_files(in_filename_cases, in_filename_deaths)
    print(df.columns)
    print(df.head())
    print(sl)
