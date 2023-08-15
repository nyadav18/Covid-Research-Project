# Covid-Research-Project

## Question 1:
The main file containing code for question 1 of this research project is [question1.py](question1.py). This file defines four different functions that are called in the 'main' function, which is where they are run. In order for these functions to yield their respective results, all the files in the *question1data* folder must be downloaded.

Links for quick access:
[Covid counts in the United States (2020-2023)](question1data/us.csv)
[Covid counts per county in the United States](https://github.com/nytimes/covid-19-data/blob/master/us-counties-2020.csv)
[Mask Usage Data](question1data/mask_use.csv)

In addition, the following libraries need to be downloaded in order to run the code:
- pandas
- matplotlib (for matplotlib.pyplot and matplotlib.dates)
- seaborn
- json
- plotly (for plotly.express)

The first function in question1.py will save the plot to an image called _'line_plot_covid_progression.png'_. The final three functions will display their respective figures on separate web pages, which should automatically open once developed.

## Question 2:
The main file to be accessed for question 2 of this research project is [question2.py](question2.py). Within this file there are four functions to be run. In order to run them, you must download all the files in the *question2data* folder. 

Links for quick access:
[Covid Counts](question2data/us-states.csv)
[PIT Change In Homelessness Counts](question2data/changeinhomelessness.csv)
[PIT Homeless Counts](question2data/PITcountsbystate.csv)
[State Names + Abbreviations](question2data/mapping.csv)

There are a few libraries necessary to download as well. These are numpy, pandas, from sckit-learn: train_test_split, mean_squared_error, r2_score, LinearRegression. Finally, from scipy.stats: pearsonr.

In order to run tests on *question2.py*, you will use [testquestion2.py](testquestion2.py). The data required to run the tests is in the folder *testingdataq2*. 

Link for quick access:
[Sample Data Frame](testingdataq2/sampledataframe.py)

Two out of the four funtions in *question2.py* don't print anything, they just return the processed datasets and the merged dataframe. The final two functions print the results along with returning them. When you run the file, you will have 4 lines of output. The first is the mean squared error, second is R-squared, third is the correlation coefficient, and the fourth is the p-value.

