# Energy Project
This project is part of the data analyst study by DataScientest.
It is presented by Aras Ergunes and Andreas Moeller.
## Available Data
The primary data source is from (website)
It can be downloaded by this (link)
The data source describes the consumtion of energy and the production of emission free energy in france. It is aggregated on a daily base for different types of energy sources and for the french regions.
There are further data sources to be merged with our primary data source:
*
*
## Provided Data 
The primary data consists of several columns. They were analyzed and qualified in a spreadsheet (link).
## Import Data and necessary libraries
import numpy as np
import pandas as pd
df = pd.read_csv('eco2mix-regional-cons-def.csv', sep = ';')
## Data analysis
* Display the data type information of the variables
* Show number of null entries for the variables
* View the statistical values for the variables

df.info()
df.isna().sum()
df.describe()
## More Key Values
* Calculate variable categorisation
* Completeness of the variable values in percent
* Distribution of categorical variables
def get_values(columns):
    for column in columns:
        # Percentage of missing values
        percentage = (df[column].isna().sum() * 100) / len(df)
        # Categorical / Quantitative 
        no_categories = len(df[column].unique())
        if no_categories <= 12:
            cat_class = 'Categorical - up to 12 categories'
        else:
            cat_class = 'Quantitative'
            
        print('--Variable: ', column,'--')
    
        print(cat_class)
        print('Percentage of missing values ', round(percentage, 2).astype(str), '%')        

        if no_categories <= 12:
            print('Distribution: ', df[column].unique())
        print('')
# Wrong values provided

get_values(df.columns)
## Cut the time frame
Our source has a daily growing data base. Its's obvious that we can't always take the newest data for our investigation. To have a clear and concise database we decided to cut a dataframe with the period of one year out of the whole data frame. To avoid difficult one time effects and to have data from the near past we decidet to investigate the time between July 2021 and June 2022. With our choice we tried to avoid side effects from the corona crisis and the ukraine war as well.
* Convert "Date - Heure" column to datetime
* Copy data from July 2021 to June 2022 into a df_project dataframe
df["Date - Heure"] = pd.to_datetime(df["Date - Heure"])

start_date = pd.to_datetime("2021-07-01 00:00:00+02:00")
end_date = pd.to_datetime("2022-06-30 00:00:00+02:00")

df_project = df[(df['Date - Heure'] >= start_date) & (df['Date - Heure'] <= end_date)]
# New Data Analysis for the project data frame
* Statistical values
* Remaining null values
* Additional analysis
df_project.describe()
df_project.isna().sum()
def get_values(columns):
    for column in columns:
        # Percentage of missing values
        percentage = (df_project[column].isna().sum() * 100) / len(df_project)
        # Categorical / Quantitative 
        no_categories = len(df_project[column].unique())
        if no_categories <= 12:
            cat_class = 'Categorical - up to 12 categories'
        else:
            cat_class = 'Quantitative'
            
        print('--Variable: ', column,'--')
    
        print(cat_class)
        print('Percentage of missing values ', round(percentage, 2).astype(str), '%')        

        if no_categories <= 12:
            print('Distribution: ', df_project[column].unique())
        print('')
# Wrong values provided

get_values(df_project.columns)
## Examine the null values at Nucléaire (MW)
The nuclear sector might be important to our investigation but we have a lot of null values in this sector. We suspect that some regions simply do not have nuclear power plants. To support this thesis, we first filter out all regions with null values from the nuclear sector. We then check whether even a single value for nuclear energy exists in these regions. If there are none, we regard our thesis as confirmed.
# regions_nuc - regions where we have null values for nuclear energy
regions_nuc = df_project.loc[df['Nucléaire (MW)'].isna()]['Région'].unique()
# proof if there are any valid values for nuclear energy in these regions
df_project.loc[(df_project.apply(lambda x: x['Région'] in [regions_nuc.any()], axis=1)) & (df['Nucléaire (MW)'].notnull())]
## Examine the null values at Pompage (MW)
The same applies to the pump sector as to the nuclear sector. Here, too, an identical proof is provided.
# regions_pmp - regions where we have null values for pumping energy
regions_pmp = df_project.loc[df['Pompage (MW)'].isna()]['Région'].unique()
# proof if there are any valid values for pumping energy in these regions
df_project.loc[(df_project.apply(lambda x: x['Région'] in [regions_pmp.any()], axis=1)) & (df['Pompage (MW)'].notnull())]

## Examine the null values at Ech. physiques (MW)
In contrast to the values for nuclear and pumped power plants, we can assume that there should be values for energy exchange with other regions in principle for all regions. Nevertheless, we have a relevant proportion of missing values in a number of regions.
# regions_eph - regions where we have null values for eph energy
regions_eph = df_project.loc[df_project['Ech. physiques (MW)'].isna()]['Région'].unique()
df_project.loc[df_project['Ech. physiques (MW)'].isna()]['Région'].unique()
## Searching for regional gaps in the data
We still achieve null values in three important columns:
* Nucléaire (MW)
* Pompage (MW)
* Ech. physiques (MW)

For further data processing, the values for nuclear and pumping power plants are replaced with 0. For the missing values in energy exchange we use the mean value.
exchange_mean = df_project.loc[(df_project.apply(lambda x: x['Région'] in [regions_eph.any()], axis=1)) & (df_project['Ech. physiques (MW)'].notnull()), 'Ech. physiques (MW)'].mean()
exchange_mean
df_project.loc[df_project['Région'].isin(regions_nuc), 'Nucléaire (MW)'] = df_project.loc[df_project['Région'].isin(regions_nuc), 'Nucléaire (MW)'].fillna(0)
df_project.loc[df_project['Région'].isin(regions_pmp), 'Pompage (MW)'] = df_project.loc[df_project['Région'].isin(regions_pmp), 'Pompage (MW)'].fillna(0)
brit_mean = df_project.loc[(df_project['Région'] == 'Bretagne') & (df_project['Ech. physiques (MW)'].notnull()), 'Ech. physiques (MW)'].mean()
df_project.loc[df_project['Région'].isin(regions_eph), 'Ech. physiques (MW)'] = df_project.loc[df_project['Région'].isin(regions_eph), 'Ech. physiques (MW)'].fillna(exchange_mean)
## Add additional variables
* Total production
* Total green production
df_project.loc[:, 'Total Production'] = df_project['Thermique (MW)'] + df_project['Nucléaire (MW)'] + df_project['Eolien (MW)'] + df_project['Solaire (MW)'] + df_project['Hydraulique (MW)'] + df_project['Pompage (MW)'] + df_project['Bioénergies (MW)'] + df_project['Ech. physiques (MW)']

df_project.loc[:, 'Green Production'] = df_project.loc[:, 'Total Production'] - df_project.loc[:, 'Nucléaire (MW)']
## Again check for null values
df_project.isna().sum()
## Prepare for statistical analysis
Simpler names for the variables are a good choice for further investigation, especially for some statistical tests. For easyer understanding we choose to rename the columns to english, too.
# Replace the variable names with easy to handle english names
bib_rename = {'Code INSEE région'   : 'insee_regional_code',
              'Région'              : 'region',
              'Nature'              : 'nature',
              'Date'                : 'date',
              'Heure'               : 'time',
              'Date - Heure'        : 'datetime',
              'Consommation (MW)'   : 'consumtion_mw',
              'Thermique (MW)'      : 'thermal_mw',
              'Nucléaire (MW)'      : 'nuclear_mw',
              'Eolien (MW)'         : 'wind_mw',
              'Solaire (MW)'        : 'solar_mw',
              'Hydraulique (MW)'    : 'hydraulic_mw',
              'Pompage (MW)'        : 'pumping_mw',
              'Bioénergies (MW)'    : 'biological_mw',
              'Ech. physiques (MW)' : 'energy_exchange_mw',
              'Stockage batterie'   : 'battery_storage',
              'Déstockage batterie' : 'battery_clearance',
              'Eolien terrestre'    : 'onshore_wind',
              'Eolien offshore'     : 'offshore_wind',
              'TCO Thermique (%)'   : 'thermal_coverage',
              'TCH Thermique (%)'   : 'thermal_utilization',
              'TCO Nucléaire (%)'   : 'nuclear_coverage',
              'TCH Nucléaire (%)'   : 'nuclear_utilization',
              'TCO Eolien (%)'      : 'wind_coverage',
              'TCH Eolien (%)'      : 'wind_utilization',
              'TCO Solaire (%)'     : 'solar_coverage',
              'TCH Solaire (%)'     : 'solar_utilization',
              'TCO Hydraulique (%)' : 'hydraulic_coverage',
              'TCH Hydraulique (%)' : 'hydraulic_utilization',
              'TCO Bioénergies (%)' : 'biological_coverage',
              'TCH Bioénergies (%)' : 'biological_utilization',
              'Column 30'           : 'column_30',
              'Total Production'    : 'total_production',
              'Green Production'    : 'green_production'}

df_project = df_project.rename(bib_rename, axis = 1)
df_project.info()
# Pearson test (numerical variables on both sides)

from scipy.stats import pearsonr

# Numerical variables
var_num = ['thermal_mw',
           'nuclear_mw',
           'wind_mw',
           'solar_mw',
           'hydraulic_mw',
           'pumping_mw',
           'biological_mw',
           'energy_exchange_mw',
           #'battery_storage',
           #'battery_clearance',
           #'onshore_wind',
           #'offshore_wind',
           'thermal_coverage',
           'thermal_utilization',
           #'nuclear_coverage',
           #'nuclear_utilization',
           'wind_coverage',
           'wind_utilization',
           'solar_coverage',
           'solar_utilization',
           #'hydraulic_coverage',
           #'hydraulic_utilization',
           #'biological_coverage',
           #'biological_utilization',
           'total_production',
           'green_production']

for var in var_num:
    print('--', var, '------------------------')
    print('H0: The production of', var, 'is correlated to the consumption')
    print('H1: The production of', var, 'is not correlated to the consumption')
    
    result = pearsonr(x = df_project['consumtion_mw'], y = df_project[var]) 

    print("p-value: ", result[1])
    print("coefficient: ", result[0])
    print('')

# ANOVA test (categorical to numeral variables)
import statsmodels.api

var_cat = ['insee_regional_code',
           'region',
           'nature',
           'time']

print('-- region ------------------------')
print('H0: The region is correlated to the consumption')
print('H1: The region is not correlated to the consumption')

result = statsmodels.formula.api.ols('consumtion_mw ~ region', data = df_project).fit()
table = statsmodels.api.stats.anova_lm(result)
display(table)
print('')

print('-- nature ------------------------')
print('H0: The nature is correlated to the consumption')
print('H1: The nature is not correlated to the consumption')

result = statsmodels.formula.api.ols('consumtion_mw ~ nature', data = df_project).fit()
table = statsmodels.api.stats.anova_lm(result)
display(table)
print('')

print('-- time ------------------------')
print('H0: The time is correlated to the consumption')
print('H1: The time is not correlated to the consumption')

result = statsmodels.formula.api.ols('consumtion_mw ~ time', data = df_project).fit()
table = statsmodels.api.stats.anova_lm(result)
display(table)
print('')

## Variables with many null values
There are still a number of variables with a significant number of missing values that prevent the execution of statistical tests

* battery_storage            17856
* battery_clearance          52992
* onshore_wind               52992
* offshore_wind             111312
* nuclear_coverage           22080
* nuclear_utilization        22080
* hydraulic_coverage        139968
* hydraulic_utilization     139968
* biological_coverage       139968
* biological_utilization    139968

The null values will be replaced by the mean of the variable
mean = df_project.mean()
df_project.fillna(mean, inplace=True)
df_project.isna().sum()
# Finding the right granularity
To go on with investigation the data should be transformed on a daily base. A monthly base would be an alternative choice. With a daily base we will have more records and expect to get better results.
df_daily = df_project.drop(['time', 'datetime'], axis = 1)
df_daily = df_daily.groupby(['insee_regional_code', 'region', 'nature', 'date']).sum()
df_daily.head(20)
# Normalisation of numerical variables
For ml modelling we normalize the numerical variables
df_daily = (df_daily - df_daily.mean()) / df_daily.std()
df_daily.head(20)

## Missing variables
Additional Variables woult be a good choice for further investigations:
* month
## Normalized variables
In some cases it would be better to compare the normalized consumption with other variables
* coverage variables
* utilisation variables
* graphical display
* decide what is our target variable (comnsumption)
* decide which variables are important
* add useful variables from other dataframes
  * meteorogical data (link)
  * geographical data
  * population data
  * industrial production
* ML techniques
# We will use  Histogram-based Gradient Boosting Regressor Model to predict the consumption of the df_daily dataframe.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# features and target variable
features = ['thermal_mw', 'nuclear_mw', 'wind_mw', 'solar_mw', 'hydraulic_mw', 'pumping_mw', 'biological_mw', 'energy_exchange_mw', 'wind_utilization', 'solar_coverage','solar_utilization','hydraulic_coverage','hydraulic_utilization', 'biological_coverage', 'biological_utilization', 'total_production', 'green_production']
target = 'consumtion_mw'

X = df_daily[features]
y = df_daily[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Adjust the HistGradientBoostingRegressor model
model = HistGradientBoostingRegressor(
    max_iter=100,
    learning_rate=0.1,
    max_depth=3,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=0.0, 
    verbose=1,
    random_state=42
)

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Calculate R^2 score
r2 = r2_score(y_test, y_pred)
print("R^2 Score:", r2)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# Predict on the test data
y_pred = model.predict(X_test)





import matplotlib.pyplot as plt
# Create a scatterplot to visualize the results
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Consumption")
plt.ylabel("Predicted Consumption")
plt.title("Actual vs. Predicted Consumption (HistGradientBoostingRegressor)")
plt.grid(True)
plt.show()


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Select features and target variable
features = ['thermal_mw', 'nuclear_mw', 'wind_mw', 'solar_mw', 'hydraulic_mw', 'pumping_mw', 'biological_mw', 'energy_exchange_mw', 'wind_utilization', 'solar_coverage','solar_utilization','hydraulic_coverage','hydraulic_utilization', 'biological_coverage', 'biological_utilization', 'total_production', 'green_production']
target = 'consumtion_mw' 

X = df_daily[features]
y = df_daily[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Calculate R^2 score
r2 = r2_score(y_test, y_pred)
print("R^2 Score:", r2)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)



# Create a scatterplot to visualize the results
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Consumption")
plt.ylabel("Predicted Consumption")
plt.title("Actual vs. Predicted Consumption (LinearRegression)")
plt.grid(True)
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


# Select features and target variable
features = ['thermal_mw', 'nuclear_mw', 'wind_mw', 'solar_mw', 'hydraulic_mw', 'pumping_mw', 'biological_mw', 'energy_exchange_mw', 'wind_utilization', 'solar_coverage','solar_utilization','hydraulic_coverage','hydraulic_utilization', 'biological_coverage', 'biological_utilization', 'total_production', 'green_production']
target = 'consumtion_mw' 

X = df_daily[features]
y = df_daily[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Decision Tree Regression model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Calculate R^2 score
r2 = r2_score(y_test, y_pred)
print("R^2 Score:", r2)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)




# Create a scatterplot to visualize the results
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Consumption")
plt.ylabel("Predicted Consumption")
plt.title("Actual vs. Predicted Consumption (DecisionTreeRegressor)")
plt.grid(True)
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# Select features and target variable
features = ['thermal_mw', 'nuclear_mw', 'wind_mw', 'solar_mw', 'hydraulic_mw', 'pumping_mw', 'biological_mw', 'energy_exchange_mw', 'wind_utilization', 'solar_coverage','solar_utilization','hydraulic_coverage','hydraulic_utilization', 'biological_coverage', 'biological_utilization', 'total_production', 'green_production']
target = 'consumtion_mw' 

X = df_daily[features]
y = df_daily[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Calculate R^2 score
r2 = r2_score(y_test, y_pred)
print("R^2 Score:", r2)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# Create a scatterplot to visualize the results
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Consumption")
plt.ylabel("Predicted Consumption")
plt.title("Actual vs. Predicted Consumption (RandomForestRegressor)")
plt.grid(True)
plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


# Select features and target variable
features = ['thermal_mw', 'nuclear_mw', 'wind_mw', 'solar_mw', 'hydraulic_mw', 'pumping_mw', 'biological_mw', 'energy_exchange_mw', 'wind_utilization', 'solar_coverage','solar_utilization','hydraulic_coverage','hydraulic_utilization', 'biological_coverage', 'biological_utilization', 'total_production', 'green_production']
target = 'consumtion_mw' 

X = df_daily[features]
y = df_daily[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVR model
model = SVR(kernel='linear')  # I have choosed 'linear' model. But there are also 'rbf' or 'poly' models.
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Calculate R^2 score
r2 = r2_score(y_test, y_pred)
print("R^2 Score:", r2)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# Create a scatterplot to visualize the results
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Consumption")
plt.ylabel("Predicted Consumption")
plt.title("Actual vs. Predicted Consumption (SVR)")
plt.grid(True)
plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error


# Select features and target variable
features = ['thermal_mw', 'nuclear_mw', 'wind_mw', 'solar_mw', 'hydraulic_mw', 'pumping_mw', 'biological_mw', 'energy_exchange_mw', 'wind_utilization', 'solar_coverage','solar_utilization','hydraulic_coverage','hydraulic_utilization', 'biological_coverage', 'biological_utilization', 'total_production', 'green_production']
target = 'consumtion_mw'  

X = df_daily[features]
y = df_daily[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the KNeighborsRegressor model
model = KNeighborsRegressor(n_neighbors=5)  # I have choosed 5 neighbors. But I can choose any number.
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Calculate R^2 score
r2 = r2_score(y_test, y_pred)
print("R^2 Score:", r2)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# Create a scatterplot to visualize the results
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Consumption")
plt.ylabel("Predicted Consumption")
plt.title("Actual vs. Predicted Consumption (KNeighborsRegressor)")
plt.grid(True)
plt.show()
# Regression Algorithm Performance Comparison

Here is a comparison table of the performance of various regression algorithms based on the Mean Squared Error (MSE) and \( R^2 \)  metrics.

| Regression Algorithm                    | MSE (Mean Squared Error) | \( R^2 \) Score                          |
|-----------------------------------------|--------------------------|------------------------------------------|
| Histogram-based Gradient Boosting       | [0.0263535371837449]     | [0.9753206546920193]                     |
| Support Vector Regression (SVR)         | [0.3247215560503265]     | [0.6959074087536965]                     |
| K-Neighbors Regressor                   | [0.040313647807803964]   | [0.9622474042882265]                     |
| Decision Tree Regressor                 | [0.03806926801898515]    | [0.9643492027460319]                     |
| Random Forest Regressor                 | [0.014027360194652009]   | [0.986863772267475]                      |
| Linear Regression                       | [0.2948698310122171]     | [0.7238627084585425]                     |
import numpy as np
import pandas as pd
df = pd.read_csv(r'C:\Users\Philosophie\Downloads\eco2mix-regional-cons-def (1).csv', sep = ';')

# Display the first few rows of the dataframe

df.head()
# Extract only the "Date" and "Consommation (MW)" columns
df_timeseries = df[["Date", "Consommation (MW)"]]

# Display the first few rows of the new dataframe
df_timeseries.head()
# Fill the missing values in "Consommation (MW)" with its mean
mean_consumption = df_timeseries["Consommation (MW)"].mean()
df_timeseries_filled = df_timeseries.copy()
df_timeseries_filled["Consommation (MW)"] = df_timeseries_filled["Consommation (MW)"].fillna(mean_consumption)

# Check if there are any missing values left
missing_values_after_fill = df_timeseries_filled["Consommation (MW)"].isna().sum()
missing_values_after_fill
import matplotlib.pyplot as plt
# Attempt to load the dataset in chunks to optimize memory usage
chunk_iter = pd.read_csv(r"C:\Users\Philosophie\Downloads\eco2mix-regional-cons-def (1).csv", header=0, sep=";", chunksize=50000)

# Initialize an empty list to store each processed chunk
df_chunks = []

# Process each chunk
for chunk in chunk_iter:
    # Retain only the "Date" and "Consommation (MW)" columns
    chunk = chunk[["Date", 'Consommation (MW)']]
    
    # Convert the "Date" column to a datetime format
    chunk['Date'] = pd.to_datetime(chunk['Date'])
    
    # Fill missing values with the mean
    chunk['Consommation (MW)'] = chunk['Consommation (MW)'].fillna(mean_consumption)
    
    # Append the processed chunk to the list
    df_chunks.append(chunk)

# Concatenate all the processed chunks
df_time_series = pd.concat(df_chunks)

# Set the "Date" column as the index
df_time_series = df_time_series.set_index("Date")

# Resample the data by month and compute the mean for each month
df_time_series = df_time_series.resample('M').mean()

# Rename the "Consommation (MW)" column to "Consumption"
df_time_series.columns = ["Consumption"]

# Convert the dataframe to a series
df_time_series = df_time_series.squeeze()

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(df_time_series, label="Monthly Average Consumption")
plt.title("Time Series of Monthly Average Consumption")
plt.xlabel("Date")
plt.ylabel("Consumption (MW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

# Seasonal decomposition using the additive model
res_additive = seasonal_decompose(df_time_series)
fig_additive = res_additive.plot()

# Seasonal decomposition using the multiplicative model
res_multiplicative = seasonal_decompose(df_time_series, model='multiplicative')
fig_multiplicative = res_multiplicative.plot()

# Logarithm transformation
df_log = np.log(df_time_series)

# Plot the transformed data
plt.figure(figsize=(12, 6))
plt.plot(df_log)
plt.title("Logarithm Transformation of Monthly Average Consumption")
plt.xlabel("Date")
plt.ylabel("Log(Consumption)")
plt.grid(True)
plt.tight_layout()
plt.show()
# Re-import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Re-process the data
df_time_series = pd.read_csv(r"C:\Users\Philosophie\Downloads\eco2mix-regional-cons-def (1).csv", header=0, sep=";")
df_time_series = df_time_series[["Date", 'Consommation (MW)']]
df_time_series['Date'] = pd.to_datetime(df_time_series['Date'])
df_time_series = df_time_series.set_index("Date")
df_time_series = df_time_series.resample('M').mean()
df_time_series.columns = ["Consumption"]
df_time_series = df_time_series.squeeze()
df_log = np.log(df_time_series)

# Define the SARIMA model with the specified parameters
model = sm.tsa.SARIMAX(df_log, order=(1, 1, 1), seasonal_order=(0, 1, 0, 24))

# Fit the model
sarima = model.fit(disp=False)

# Predict values
start_point = 113
end_point = 128
pred_log = sarima.predict(start=start_point, end=end_point)

# Convert predictions back to original scale
pred = np.exp(pred_log)

# Concatenate the original series with predictions
df_pred = pd.concat([df_time_series, pred])

# Plot the original series along with predictions
plt.figure(figsize=(12, 6))
plt.plot(df_pred, label="Observed + Predicted")
plt.title("Monthly Average Consumption with SARIMA Predictions")
plt.xlabel("Date")
plt.ylabel("Consumption (MW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


