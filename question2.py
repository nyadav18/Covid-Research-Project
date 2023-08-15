"""
Nikita Yadav
CSE 163 AA

This file contains code for part two of our Covid-19 analysis. It includes
functions to load and process the datasets used in this research question:
dataset 1 contains cumulative counts of covid deaths, dataset 2 contains
counts of the homeless population in each state in the US, dataset 3 contains
percentages representing the change in homeless counts from 2020-2022, and
dataset 4 contains state names and state abbreviations to assist with the
merging process. The file also contains a function to run a linear
regression model on this data and determine the correlation coefficent and
p-value of certain features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr


def load_data(PIT_data_path: str, covid_state_data_path: str,
              mapping_data_path: str, change_data_path: str) -> tuple:
    """
    Takes in four file paths: three representing the datasets to be analyzed
    (cumulative covid counts for 2022, homeless population counts for 2022, and
    change in homelessness from 2020-2022) and one containing state names and
    state abbreviations to be used in the following merge function. Returns a
    tuple containing the four pre-processed dataframes.
    """

    PIT_data = pd.read_csv(PIT_data_path)
    covid_data = pd.read_csv(covid_state_data_path)
    mapping_data = pd.read_csv(mapping_data_path)
    change_data = pd.read_csv(change_data_path)

    # remove unnecessary columns / rows
    PIT_data = PIT_data.loc[:, ['State', 'Overall Homeless, 2022']]
    PIT_data['Overall Homeless, 2022'] = \
        PIT_data['Overall Homeless, 2022'].str.replace(',', '')
    # replaces non-numeric values with NaN
    PIT_data['Overall Homeless, 2022'] = \
        pd.to_numeric(PIT_data['Overall Homeless, 2022'], errors='coerce')

    PIT_data['Overall Homeless, 2022'] = \
        PIT_data['Overall Homeless, 2022'].fillna(0).astype(int)

    covid_data = covid_data[covid_data['date'] == '2022-12-31']

    mapping_data = mapping_data.loc[:, [
        'Official Code State',
        'Official Name State',
        'United States Postal Service state abbreviation']]

    mapping_data = mapping_data.rename(columns={
        'United States Postal Service state abbreviation': 'abbreviation'})

    change_data = change_data.loc[:, [
        'State', 'Change in Total Homelessness, 2020-2022']]
    change_data = change_data.rename(columns={
        'Change in Total Homelessness, 2020-2022':
        'change 2020-2022'})

    return PIT_data, covid_data, mapping_data, change_data


def merge_data(PIT_data: pd.DataFrame, covid_data: pd.DataFrame,
               mapping_data: pd.DataFrame,
               change_data: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in the four dataframes as parameters. Merges the covid_data
    dataframe to the mapping dataframe to create a common column for the
    main merges between the covid dataframe, the PIT dataframe, and the PIT
    change dataframe. Returns a single dataframe representing the combined
    datasets.
    """

    # join mapping dataset to covid data
    covid_mapping_merged = covid_data.merge(mapping_data, left_on='fips',
                                            right_on='Official Code State',
                                            how='inner')
    # remove unnecessary columns
    covid_data_cleaned = covid_mapping_merged.loc[:, ['state', 'fips',
                                                      'cases', 'deaths',
                                                      'abbreviation']]

    # join cleaned and merged covid dataset to PIT dataset and homeless counts
    merge1 = covid_data_cleaned.merge(PIT_data, left_on='abbreviation',
                                      right_on='State', how='inner')
    main_data = merge1.merge(change_data, left_on='abbreviation',
                             right_on='State',
                             how='inner')

    # remove unnecessary columns from main dataset
    main_data = main_data.loc[:, ['state', 'fips', 'cases', 'deaths',
                                  'abbreviation', 'Overall Homeless, 2022',
                                  'change 2020-2022']]

    return main_data


def run_model(main_data: pd.DataFrame) -> np.ndarray:
    """
    Takes in the merged data frame as a parameter. Splits the data into
    training and testing set. Fits model and predicts covid death counts
    for each state based on homeless population counts and change in
    homelessness for the respective state. Evaluates the model using two
    common evaluation metrics: mean squared error and R-squared.
    Returns the results of these metrics.
    """
    features = pd.get_dummies(main_data.loc[:, [
        'state', 'Overall Homeless, 2022', 'change 2020-2022']])
    label = main_data['deaths']

    features_train, features_test, label_train, label_test = \
        train_test_split(features, label, test_size=0.2)

    model = LinearRegression()
    model.fit(features_train, label_train)

    test_predictions = model.predict(features_test)
    error = mean_squared_error(label_test, test_predictions)
    print(error)
    r2 = r2_score(label_test, test_predictions)
    print(r2)

    return error, r2


def correlation_coefficient(main_data: pd.DataFrame) -> float:
    """
    Takes in the main data frame and calculates the correlation coefficient
    and p-value to determine the relationship between homeless counts in each
    state and covid death counts for the same states. Returns the correlation
    coefficient and the p-value.
    """
    homeless_counts = main_data['Overall Homeless, 2022']
    covid_deaths = main_data['deaths']

    correlation_coefficient, p_value = pearsonr(homeless_counts, covid_deaths)

    print(correlation_coefficient)
    print(p_value)
    return correlation_coefficient, p_value


def main():
    PIT_data_path = (
        '/Users/nikitayadav/CSE163/Covid-Research-Project/question2data/'
        'PITcountsbystate.csv'
        )

    covid_state_data_path = (
        '/Users/nikitayadav/CSE163/Covid-Research-Project/question2data/'
        'us-states.csv'
        )

    mapping_data_path = (
        '/Users/nikitayadav/CSE163/Covid-Research-Project/question2data/'
        'mapping.csv'
        )

    change_data_path = (
        '/Users/nikitayadav/CSE163/Covid-Research-Project/question2data/'
        'changeinhomelessness.csv'
        )

    PIT_data, covid_data, mapping_data, change_data = load_data(
        PIT_data_path, covid_state_data_path, mapping_data_path,
        change_data_path)
    merged_data = merge_data(PIT_data, covid_data, mapping_data, change_data)
    run_model(merged_data)
    correlation_coefficient(merged_data)


if __name__ == "__main__":
    main()
