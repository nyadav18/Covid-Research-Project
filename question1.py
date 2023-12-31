"""
Netra Krishnan
Section AB

Assessing the Ongoing Impacts of the COVID-19 Pandemic in the U.S.

Question 1:
How have COVID-19 cases in the United States changed over the course of the
pandemic, and what are some possible reasons for this?

To answer this question, COVID-19 data is graphed and mapped, also utilizing a
separate dataset of mask usage to analyze the data.
Mask data and COVID-19 data is also used alongside a geojson file of U.S.
counties in order to map the data using the plotly library.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

import json
import plotly.express as px

sns.set()


def covid_progression_line_plot(covid_data: pd.DataFrame) -> None:
    """
    Returns two lineplots; the top plot shows the progression of COVID-19
    cases from 2020-2023, and the bottom plot shows the progression of deaths
    due to COVID-19 from 2020-2023.
    """
    # Data Manipulation
    covid_data['date'] = pd.to_datetime(covid_data['date'])

    # Plot data
    fig, [ax1, ax2] = plt.subplots(2, figsize=(9, 7))
    ax1.plot(covid_data['date'], covid_data['cases'])
    ax2.plot(covid_data['date'], covid_data['deaths'], color='#FFA500')

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%y'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%y'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())

    ax1.set_title("COVID-19 Data in the United States from 2020-2023")
    ax1.set_xlabel("Date")
    ax2.set_xlabel("Date")
    ax1.set_ylabel("Cases")
    ax2.set_ylabel("Deaths")

    plt.savefig('line_plot_covid_progression.png', bbox_inches='tight')


def never_wearing_mask_map(mask_data: pd.DataFrame) -> None:
    """
    Plots a map of the percentage of people per U.S. county who reported from
    July 2-14 of 2020 that they never wear a mask.
    """

    # Read json counties file for mapping
    with open('geojson-counties-fips.json') as f:
        us_counties = json.load(f)

    # Plot mask usage data on map
    fig2 = px.choropleth_mapbox(mask_data, geojson=us_counties,
                                locations='COUNTYFP', color='NEVER',
                                color_continuous_scale='speed',
                                mapbox_style='carto-positron',
                                zoom=3,
                                center={'lat': 37.0902, 'lon': -95.7129},
                                opacity=0.65,
                                labels={'NEVER':
                                        'Percentage Never Wearing Mask'})

    t = 'Percentage of People Never Wearing a Mask,'
    fig2.update_layout(title_text=(t + ' per U.S. County (July 2020)'),
                       margin={'r': 0, 'l': 0})
    fig2.show()


def always_wearing_mask_map(mask_data: pd.DataFrame) -> None:
    """
    Plots a map of the percentage of people per U.S. county who reported from
    July 2-14 of 2020 that they always wear a mask.
    """

    # Read json counties file for mapping
    with open('geojson-counties-fips.json') as f:
        us_counties = json.load(f)

    # Plot mask usage data on map
    fig1 = px.choropleth_mapbox(mask_data, geojson=us_counties,
                                locations='COUNTYFP', color='ALWAYS',
                                color_continuous_scale='speed',
                                mapbox_style='carto-positron',
                                zoom=3,
                                center={'lat': 37.0902, 'lon': -95.7129},
                                opacity=0.65,
                                labels={'ALWAYS':
                                        'Percentage Always Wearing Mask'})

    t = 'Percentage of People Always Wearing a Mask,'
    fig1.update_layout(title_text=(t + ' per U.S. County (July 2020)'),
                       margin={'r': 0, 'l': 0})
    fig1.show()


def covid_case_map(covid_counties: pd.DataFrame) -> None:
    """
    Plots a map of the number of COVID-19 cases per U.S. county from
    July 2-14 2020, which can be used to compare with the mask usage data
    visualizations.
    """

    # Data Manipulation (filter for cases from July 2-14 in 2020)
    covid_counties['year'] = covid_counties['date'].map(lambda date:
                                                        date.split('-')[0])
    covid_counties['month'] = covid_counties['date'].map(lambda date:
                                                         date.split('-')[1])
    covid_counties['day'] = covid_counties['date'].map(lambda date:
                                                       date.split('-')[2])
    covid_counties['day'] = covid_counties['day'].astype(int)

    year = covid_counties['year'] == '2020'
    month = covid_counties['month'] == '07'
    day1 = covid_counties['day'] >= 2
    day2 = covid_counties['day'] <= 14

    filtered_df = covid_counties[year & month & day1 & day2]

    # Read json counties file for mapping
    with open('geojson-counties-fips.json') as f:
        us_counties = json.load(f)

    # Plot data
    fig = px.choropleth_mapbox(filtered_df, geojson=us_counties,
                               locations='fips', color='cases',
                               color_continuous_scale='speed',
                               range_color=(0, 1500),
                               mapbox_style='carto-positron',
                               zoom=3,
                               center={'lat': 37.0902, 'lon': -95.7129},
                               opacity=0.65,
                               labels={'cases': 'Number of Cases per County'})

    t = 'Covid-19 Cases in the United States Counties (July 2020)'
    fig.update_layout(title_text=t, margin={'r': 0, 'l': 0})
    fig.show()


def main():
    # Load in datasets
    covid_data = pd.read_csv('us.csv', na_values=['---'])
    mask_data = pd.read_csv('mask_use.csv', dtype={'COUNTYFP': str})
    covid_counties = pd.read_csv('us-counties-2020.csv', na_values=['---'])

    # Call functions to create data visualizations
    covid_progression_line_plot(covid_data)
    never_wearing_mask_map(mask_data)
    always_wearing_mask_map(mask_data)
    covid_case_map(covid_counties)


if __name__ == '__main__':
    main()
