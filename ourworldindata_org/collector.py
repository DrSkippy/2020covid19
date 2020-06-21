import pandas as pd
import logging
import datetime
import csv

logfile = "/Users/drskippy/logs/covid.log"
logging.basicConfig(level=logging.DEBUG, filename=logfile)

"""
get the data!
clone at the same level as teh 2020covid19 repo
    git clone git@github.com:owid/covid-19-data.git
"""
in_filename = "../covid-19-data/public/data/owid-covid-data.csv"

logging.info("using file={}".format(in_filename))

def get_dataset_df_from_files(fn):

    df_ = pd.read_csv(fn, quoting=csv.QUOTE_MINIMAL)
    # HEADER
    # iso_code, continent, location, date, total_cases, new_cases, total_deaths, new_deaths,
    # total_cases_per_million, new_cases_per_million, total_deaths_per_million, new_deaths_per_million,
    # total_tests, new_tests, total_tests_per_thousand, new_tests_per_thousand, new_tests_smoothed,
    # new_tests_smoothed_per_thousand, tests_units, stringency_index, population, population_density,
    # median_age, aged_65_older, aged_70_older, gdp_per_capita, extreme_poverty, cvd_death_rate,
    # diabetes_prevalence, female_smokers, male_smokers, handwashing_facilities,
    # hospital_beds_per_thousand, life_expectancy

    df = pd.DataFrame()
    # Entity,Code,Date,Total confirmed cases of COVID-19 (cases)
    df["Entity"] = df_["location"]
    df["state"] = df_["iso_code"]
    df["Date"] = df_["date"]
    df["positive"] = df_["total_cases"]
    df["death"] = df_["total_deaths"]
    df["last_update"] = datetime.datetime.now()

    logging.debug("header={}".format(df.columns))

    df["date"] = pd.to_datetime(df.Date.apply(lambda x: x.strip().lower()), format="%Y-%m-%d")
    df["daily_new_positive"] = df.positive.diff(1)
    df["daily_new_death"] = df.death.diff(1)
    df["daily_new_positive"] = df.daily_new_positive.fillna(0)
    df["daily_new_death"] = df.daily_new_death.fillna(0)
    sl = df.state.unique()
    return df, sl


if "__main__" == __name__:
    df, sl = get_dataset_df_from_files(in_filename)
    print(df.columns)
    print(df.head())
    print(sl)
