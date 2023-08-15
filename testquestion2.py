"""
Nikita Yadav
CSE 163 AA

This file contains functions to test the methods defined in question2.py.
"""
from question2 import load_data, merge_data, run_model, correlation_coefficient
from testingdataq2 import sampledataframe
import pandas as pd

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


def test_load_data() -> None:
    """
    Tests the load_data function.
    """
    PIT_data, covid_data, mapping_data, change_data = load_data(
        PIT_data_path, covid_state_data_path, mapping_data_path,
        change_data_path)

    assert isinstance(PIT_data, pd.DataFrame)
    assert isinstance(covid_data, pd.DataFrame)
    assert isinstance(mapping_data, pd.DataFrame)
    assert isinstance(change_data, pd.DataFrame)

    assert 'State' in PIT_data.columns
    assert 'date' in covid_data.columns
    assert 'abbreviation' in mapping_data.columns
    assert 'change 2020-2022' in change_data.columns


def test_merge_data() -> None:
    """
    Tests merge_data function.
    """
    PIT_data, covid_data, mapping_data, change_data = load_data(
        PIT_data_path, covid_state_data_path, mapping_data_path,
        change_data_path)

    main_data = merge_data(PIT_data, covid_data,
                           mapping_data, change_data)

    assert isinstance(main_data, pd.DataFrame)
    assert 'State' not in main_data.columns
    assert 'state' in main_data.columns
    assert 'Change in Total Homelessness, 2020-2022' \
        not in main_data.columns


def test_run_model() -> None:
    """
    Tests run_model function
    """
    error, r2 = run_model(sampledataframe.sample1)

    assert isinstance(error, float)
    assert isinstance(r2, float)


def test_correlation_coefficient() -> None:
    """
    Tests correlation_coefficient function
    """
    corr_coefficient, p_value = correlation_coefficient(
        sampledataframe.sample1)

    assert isinstance(corr_coefficient, float)
    assert isinstance(p_value, float)
    assert -1 <= corr_coefficient <= 1


def main():

    test_load_data()
    test_merge_data()
    test_run_model()
    test_correlation_coefficient()


if __name__ == "__main__":
    main()
