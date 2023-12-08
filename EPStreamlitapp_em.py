from sys import displayhook
from sysconfig import get_python_version
from tkinter.tix import DisplayStyle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

df = pd.read_csv('eco2mix-regional-cons-def.csv', sep = ';')
df_project = pd.read_csv ('df_project.csv', sep = ',')



def get_values(df, columns):
    for column in columns:
        percentage = (df[column].isna().sum() * 100) / len(df)
        no_categories = len(df[column].unique())
        if no_categories <= 12:
            cat_class = 'Categorical - up to 12 categories'
        else:
            cat_class = 'Quantitative'

        # Use streamlit's st.write() instead of print() for Streamlit apps
        st.write('--Variable: ', column,'--')
        st.write(cat_class)
        st.write('Percentage of missing values ', round(percentage, 2), '%')
        
        if no_categories <= 12:
            st.write('Distribution: ', df[column].unique())
        st.write('')

menu = ['Introduction', 'Data Presentation', 'Preprocessing', 'Data Visualization', 'ML Techniques', 'Time Series', 'Conclusion']
selection = st.sidebar.radio("Menu", menu)

if selection == 'Introduction':
    with header:
        image = Image.open('zollverein.jpg')
        st.image(image, caption='Zeche Zollverein, Essen, Germany. ® Aras Ergunes')
    with dataset:
        st.header ('This project is part of the data analyst study by DataScientest.')
        st.write ('It can be downloaded by this link (https://www.data.gouv.fr/fr/datasets/r/6a69cffb-a123-4f81-b6d1-a86f466f61a0')
        st.write ('The data source describes the consumption of energy and the production of emission free energy in france. It is aggregated on a daily base for different types of energy sources and for the french regions.')


elif selection == 'Data Presentation':
    st.write("Here is the main dataset")
    st.write ('It is from (https://www.data.gouv.fr/fr/datasets/donnees-eco2mix-regionales-consolidees-et-definitives-janvier-2013-a-mai-2022/')
    
    df = pd.read_csv ('eco2mix-regional-cons-def.csv', sep = ';')
    st.write (df.head (20))
    

elif selection == 'Preprocessing':
    st.header('Preprocessing')
    st.write ('* Cut the time frame from july 21 to june 22')
#    df["Date - Heure"] = pd.to_datetime(df["Date - Heure"])
#    start_date = pd.to_datetime("2021-07-01 00:00:00+02:00")
#    end_date = pd.to_datetime("2022-06-30 00:00:00+02:00")
#
#    df_project = df[(df['Date - Heure'] >= start_date) & (df['Date - Heure'] <= end_date)]
#    df_project.loc[:, 'Total Production'] = df_project['Thermique (MW)'] + df_project['Nucléaire (MW)'] + df_project['Eolien (MW)'] + df_project['Solaire (MW)'] + df_project['Hydraulique (MW)'] + df_project['Pompage (MW)'] + df_project['Bioénergies (MW)'] + df_project['Ech. physiques (MW)']
#    df_project.loc[:, 'Green Production'] = df_project.loc[:, 'Total Production'] - df_project.loc[:, 'Nucléaire (MW)']
#    # What to do with nans?
#    #st.write(df.isna().sum()) AM_20231117
#
#    # regions_nuc - regions where we have null values for nuclear energy
#    regions_nuc = df_project.loc[df['Nucléaire (MW)'].isna()]['Région'].unique()
#    # proof if there are any valid values for nuclear energy in these regions
#    #df_project.loc[(df_project.apply(lambda x: x['Région'] in [regions_nuc.any()], axis=1)) & (df['Nucléaire (MW)'].notnull())] AM_20231117
#
#
#    # regions_pmp - regions where we have null values for pumping energy
#    regions_pmp = df_project.loc[df['Pompage (MW)'].isna()]['Région'].unique()
#    # proof if there are any valid values for pumping energy in these regions
#    #df_project.loc[(df_project.apply(lambda x: x['Région'] in [regions_pmp.any()], axis=1)) & (df['Pompage (MW)'].notnull())] AM_20231117
#
#
#
#    # regions_eph - regions where we have null values for eph energy
#    regions_eph = df_project.loc[df_project['Ech. physiques (MW)'].isna()]['Région'].unique()
#    #print(regions_eph)
#    # proof if there are any valid values for eph energy in these regions
#    #df_project.loc[(df_project.apply(lambda x: x['Région'] in [regions_eph.any()], axis=1)) & (df['Ech. physiques (MW)'].notnull())]
#    #df_eph = df_project.loc[(df_project.apply(lambda x: x['Région'] in [regions_eph.any()], axis=1)) & (df['Ech. physiques (MW)'].notnull())]
#    #regions_eph = ['Bourgogne-Franche-Comté', 'Île-de-France', "Provence-Alpes-Côte d'Azur", 'Pays de la Loire', 'Grand Est', 'Auvergne-Rhône-Alpes', 'Hauts-de-France', 'Nouvelle-Aquitaine', 'Occitanie', 'Centre-Val de Loire', 'Normandie']
#    #df_project.loc[(df_project.apply(lambda x: x['Région'] in [regions_eph], axis=1)) & (df['Ech. physiques (MW)'].notnull())]
#    # Find out to which regions these found values belong
#    #print(df_eph['Région'], df_eph['Ech. physiques (MW)'])
#    #print(df_eph.loc[df_eph['Ech. physiques (MW)'].isna()])#['Ech. physiques (MW)']
    
    st.write('* Replace n/a values for nuclear energy, pumping energy and energy exchange with 0')

#    df_project.loc[df_project['Région'].isin(regions_nuc), 'Nucléaire (MW)'] = df_project.loc[df_project['Région'].isin(regions_nuc), 'Nucléaire (MW)'].fillna(0)
#    df_project.loc[df_project['Région'].isin(regions_pmp), 'Pompage (MW)'] = df_project.loc[df_project['Région'].isin(regions_pmp), 'Pompage (MW)'].fillna(0)
#    df_project.loc[df_project['Région'].isin(regions_eph), 'Ech. physiques (MW)'] = df_project.loc[df_project['Région'].isin(regions_eph), 'Ech. physiques (MW)'].fillna(0)
#
#    # ## Variables with many null values
#    # There are still a number of variables with a significant number of missing values that prevent the execution of statistical tests
#    # 
#    # * battery_storage            17856
#    # * battery_clearance          52992
#    # * onshore_win
#    # d               52992
#    # * offshore_wind             111312
#    # * nuclear_coverage           22080
#    # * nuclear_utilization        22080
#    # * hydraulic_coverage        139968
#    # * hydraulic_utilization     139968
#    # * biological_coverage       139968
#    # * biological_utilization    139968
#    # 
    st.write('* The null values will be replaced by the mean of the variable')

#    mean = df_project.mean()
#    df_project.fillna(mean, inplace=True)
#    df_project.isna().sum()
    
    st.write('* Adding total and green production')
#
#    # ## Prepare for statistical analysis
#    # Simpler names for the variables are a good choice for further investigation, especially for some statistical tests. For easyer understanding we choose to rename the columns to english, too.

    st.write('* Replace the variable names with easy to handle english names')

#    bib_rename = {'Code INSEE région'   : 'insee_regional_code',
#                'Région'              : 'region',
#                'Nature'              : 'nature',
#                'Date'                : 'date',
#                'Heure'               : 'time',
#                'Date - Heure'        : 'datetime',
#                'Consommation (MW)'   : 'consumtion_mw',
#                'Thermique (MW)'      : 'thermal_mw',
#                'Nucléaire (MW)'      : 'nuclear_mw',
#                'Eolien (MW)'         : 'wind_mw',
#                'Solaire (MW)'        : 'solar_mw',
#                'Hydraulique (MW)'    : 'hydraulic_mw',
#                'Pompage (MW)'        : 'pumping_mw',
#                'Bioénergies (MW)'    : 'biological_mw',
#                'Ech. physiques (MW)' : 'energy_exchange_mw',
#                'Stockage batterie'   : 'battery_storage',
#                'Déstockage batterie' : 'battery_clearance',
#                'Eolien terrestre'    : 'onshore_wind',
#                'Eolien offshore'     : 'offshore_wind',
#                'TCO Thermique (%)'   : 'thermal_coverage',
#                'TCH Thermique (%)'   : 'thermal_utilization',
#                'TCO Nucléaire (%)'   : 'nuclear_coverage',
#                'TCH Nucléaire (%)'   : 'nuclear_utilization',
#                'TCO Eolien (%)'      : 'wind_coverage',
#                'TCH Eolien (%)'      : 'wind_utilization',
#                'TCO Solaire (%)'     : 'solar_coverage',
#                'TCH Solaire (%)'     : 'solar_utilization',
#                'TCO Hydraulique (%)' : 'hydraulic_coverage',
#                'TCH Hydraulique (%)' : 'hydraulic_utilization',
#                'TCO Bioénergies (%)' : 'biological_coverage',
#                'TCH Bioénergies (%)' : 'biological_utilization',
#                'Column 30'           : 'column_30',
#                'Total Production'    : 'total_production',
#                'Green Production'    : 'green_production'}
#
#    df_project = df_project.rename(bib_rename, axis = 1)
#
#    df_project.to_csv('C:/Users/a.moeller/GitHub/dataproject/eco2mix_preprocessed.csv')
    df_project = pd.read_csv ('eco2mix_preprocessed.csv', sep = ',')

    st.write(df_project.head())
    
    st.header('Statistical Analysis')
    image = Image.open('Pearson.png')
    st.image(image)
    image = Image.open('ANOVA.png')
    st.image(image)


    st.write('* Pearson test (numerical variables on both sides)')

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
            'thermal_coverage',
            'thermal_utilization',
            'wind_coverage',
            'wind_utilization',
            'solar_coverage',
            'solar_utilization',
            'total_production',
            'green_production']

    for var in var_num:
        st.write('--', var, '------------------------')
        st.write('H0: The production of', var, 'is correlated to the consumption')
        st.write('H1: The production of', var, 'is not correlated to the consumption')
        
        result = pearsonr(x = df_project['consumtion_mw'], y = df_project[var]) 

        st.write("p-value: ", result[1])
        st.write("coefficient: ", result[0])
        st.write('')



    st.write('* ANOVA test (categorical to numeral variables)')

    import statsmodels.api

    var_cat = ['insee_regional_code',
            'region',
            'nature',
            'time']

    st.write('-- region ------------------------')
    st.write('H0: The region is correlated to the consumption')
    st.write('H1: The region is not correlated to the consumption')

    result = statsmodels.formula.api.ols('consumtion_mw ~ region', data = df_project).fit()
    table = statsmodels.api.stats.anova_lm(result)
    #displayhook(table))
    st.write(table)
    st.write('')

    st.write('-- nature ------------------------')
    st.write('H0: The nature is correlated to the consumption')
    st.write('H1: The nature is not correlated to the consumption')

    result = statsmodels.formula.api.ols('consumtion_mw ~ nature', data = df_project).fit()
    table = statsmodels.api.stats.anova_lm(result)
    #DisplayStyle(table)
    st.write(table)
    st.write('')

    st.write('-- time ------------------------')
    st.write('H0: The time is correlated to the consumption')
    st.write('H1: The time is not correlated to the consumption')

    result = statsmodels.formula.api.ols('consumtion_mw ~ time', data = df_project).fit()
    table = statsmodels.api.stats.anova_lm(result)
    #display(table)
    st.write(table)
    st.write('')

elif selection == 'Data Visualization':
    import streamlit as st
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    import joblib
    import os
    from PIL import Image

    def load_and_process_data():
        df_project = pd.read_csv ('df_project.csv', sep = ',')
        df_project['Date'] = pd.to_datetime(df_project['Date'])
        df_project['Date - Heure'] = pd.to_datetime(df_project['Date - Heure'], utc=True)
        df_project['Year'] = df_project['Date - Heure'].dt.year
        df_project['Month'] = df_project['Date - Heure'].dt.month
        df_project['Day'] = df_project['Date - Heure'].dt.day
        return df_project
    df_project = load_and_process_data()


    # save DataFrame with joblib
    #def save_data_with_joblib(df_project, filename):
        #joblib.dump(df_project, filename)

    # load DataFrame with joblib
    #def load_data_with_joblib(filename):
        #return joblib.load(filename)

    #def load_and_process_data():
        #processed_data_file = 'processed_df_project.pkl'

        #if os.path.exists(processed_data_file):
            #df_project = load_data_with_joblib(processed_data_file)
        #else:
            #df_project = pd.read_csv('df_project.csv', sep=',')
            #df_project['Date'] = pd.to_datetime(df_project['Date'])
            #df_project['Date - Heure'] = pd.to_datetime(df_project['Date - Heure'], utc=True)
            #df_project['Year'] = df_project['Date - Heure'].dt.year
            #df_project['Month'] = df_project['Date - Heure'].dt.month
            #df_project['Day'] = df_project['Date - Heure'].dt.day
            #save_data_with_joblib(df_project, processed_data_file)

        #return df_project

    #df_project = load_and_process_data()
    

    st.header("Data Visualization")
    st.write ('Dataframe Columns for Visualization')
    st.write(df_project.columns)

    st.write ('Plotting the Consommation (MW)')

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=df_project["Date"], y=df_project["Consommation (MW)"])
    ax.set_title("Energy Consumption")
    ax.set_xlabel("Date")
    ax.set_ylabel("Consommation (MW)")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.write ('Consommation (MW) based on year')

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=df_project["Year"], y=df_project["Consommation (MW)"])
    ax.set_title('Consumption (MW) by Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Consommation (MW)')
    st.pyplot(fig)

    st.write ('Consommation (MW) based on month')

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=df_project["Month"], y=df_project["Consommation (MW)"])
    ax.set_title('Consumption (MW) by Month')
    ax.set_xlabel('Month')
    ax.set_ylabel('Consommation (MW)')
    st.pyplot(fig)

    st.write ('Consommation (MW) based on day')

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=df_project["Day"], y=df_project["Consommation (MW)"])
    ax.set_title('Consumption (MW) by Day')
    ax.set_xlabel('Day')
    ax.set_ylabel('Consommation (MW)')
    st.pyplot(fig)



    st.write ('Plotting the Consommation (MW) vs Total Production')
    image = Image.open('consommation.jpg')
    st.image(image)

    #sns.set_style("whitegrid")
    #fig, ax = plt.subplots(figsize=(12, 6))
    #sns.lineplot(x=df_project["Total Production"], y=df_project["Consommation (MW)"])
    #ax.set_title("Consommation (MW) vs Total Production")
    #ax.set_xlabel("Total Production")
    #ax.set_ylabel("Consommation (MW)")
    #plt.xticks(rotation=45)
    #st.pyplot(fig)

    st.write ('Plotting the Green Production vs Total Production')

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.regplot(x=df_project["Total Production"], y=df_project["Green Production"])
    ax.set_title("Green Production vs Total Production")
    ax.set_xlabel("Total Production")
    ax.set_ylabel("Green Production")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.write ('Plotting the Consommation (MW) vs Green Production')

    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    sns.regplot(x=df_project["Green Production"], y=df_project["Consommation (MW)"])
    ax.set_title("Consommation (MW) vs Green Production")
    ax.set_xlabel("Green Production")
    ax.set_ylabel("Consommation (MW)")
    plt.xticks(rotation=45)
    st.pyplot(fig)


elif selection == 'ML Techniques':
    st.header("Machine Learning Techniques")
    image = Image.open('regression.table.jpg')
    st.image(image, caption= 'The random forest regression model performs the best.')

    import streamlit as st
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    import matplotlib.pyplot as plt
    import numpy as np
    import joblib
    import os
    from PIL import Image


    # Assuming a function to load data
    
    def load_data():
        df = pd.read_csv('df_daily.csv')
        return df

    df_daily = load_data()
    st.write(df_daily.head())

    # UI element to select the model
    model_option = st.selectbox(
        'Select a regression model:',
        ('HistGradientBoostingRegressor', 'LinearRegression', 'DecisionTreeRegressor', 'RandomForestRegressor', 'SVR', 'KNeighborsRegressor')
    )

    #  HistGradientBoostingRegressor
    def train_hist_gradient_boosting(X_train, X_test, y_train, y_test):
        model = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, max_depth=3, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.0, verbose=1, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return model, mse, r2, y_pred

    # LinearRegression
    def train_linear_regression(X_train, y_train, X_test, y_test):
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return model, mse, r2, y_pred

    # DecisionTreeRegressor
    def train_decision_tree(X_train, y_train, X_test, y_test):
        model = DecisionTreeRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return model, mse, r2, y_pred

    # RandomForestRegressor
    def train_random_forest(X_train, y_train, X_test, y_test):
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return model, mse, r2, y_pred

    # SVR
    def train_svr(X_train, y_train, X_test, y_test):
        model = SVR()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return model, mse, r2, y_pred

    # KNeighborsRegressor
    def train_k_neighbors(X_train, y_train, X_test, y_test):
        model = KNeighborsRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return model, mse, r2, y_pred
    # Directory to store models
    model_dir = 'saved_models'
    os.makedirs(model_dir, exist_ok=True)

    # Main execution
    if st.button('Run Model'):
        features = ['thermal_mw', 'nuclear_mw', 'wind_mw', 'solar_mw', 'hydraulic_mw', 'pumping_mw', 'biological_mw', 'energy_exchange_mw', 'wind_utilization', 'solar_coverage','solar_utilization','hydraulic_coverage','hydraulic_utilization', 'biological_coverage', 'biological_utilization', 'total_production', 'green_production']
        target = 'consumtion_mw'

        X = df_daily[features]
        y = df_daily[target]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model, mse, r2, y_pred = None, None, None, None
        model_filename = os.path.join(model_dir, f'{model_option}.pkl')
        if model_option == 'HistGradientBoostingRegressor':
            model, mse, r2, y_pred = train_hist_gradient_boosting(X_train, X_test, y_train, y_test)
            joblib.dump(model, model_filename)
        elif model_option == 'LinearRegression':
            model, mse, r2, y_pred = train_linear_regression(X_train, y_train, X_test, y_test)
            joblib.dump(model, model_filename)
        elif model_option == 'DecisionTreeRegressor':
            model, mse, r2, y_pred = train_decision_tree(X_train, y_train, X_test, y_test)
            joblib.dump(model, model_filename)
        elif model_option == 'RandomForestRegressor':
            model, mse, r2, y_pred = train_random_forest(X_train, y_train, X_test, y_test)
            joblib.dump(model, model_filename)
        elif model_option == 'SVR':
            model, mse, r2, y_pred = train_svr(X_train, y_train, X_test, y_test)
            joblib.dump(model, model_filename)
        elif model_option == 'KNeighborsRegressor':
            model, mse, r2, y_pred = train_k_neighbors(X_train, y_train, X_test, y_test)
            joblib.dump(model, model_filename)
        if model:
            st.write(f'Model: {model_option}')
            st.write(f'Mean Squared Error: {mse}')
            st.write(f'R-squared Score: {r2}')

            # Plotting
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, y_pred, alpha=0.5)
            ax.set_xlabel("Actual Consumption")
            ax.set_ylabel("Predicted Consumption")
            ax.set_title(f"Actual vs. Predicted Consumption ({model_option})")
            ax.grid(True)

            st.pyplot(fig)
            st.write(f'Model saved as {model_filename}')

elif selection == 'Time Series':
    st.header("Time Series Analyses")

    import streamlit as st
    import numpy as np
    import pandas as pd
    import numpy as np
    from statsmodels.tsa.seasonal import seasonal_decompose
    import matplotlib.pyplot as plt

    df = pd.read_csv('eco2mix-regional-cons-def.csv', sep = ';')
    # Extract only the "Date" and "Consommation (MW)" columns
    df_timeseries = df[["Date", "Consommation (MW)"]]
    # Display the first few rows of the new dataframe in Streamlit
    # st.write(df_timeseries.head())


    # Fill the missing values in "Consommation (MW)" with its mean
    mean_consumption = df_timeseries["Consommation (MW)"].mean()
    df_timeseries_filled = df_timeseries.copy()
    df_timeseries_filled["Consommation (MW)"] = df_timeseries_filled["Consommation (MW)"].fillna(mean_consumption)

    # Check if there are any missing values left
    missing_values_after_fill = df_timeseries_filled["Consommation (MW)"].isna().sum()
    st.write("Missing values after filling:", missing_values_after_fill)

    # Attempt to load the dataset in chunks to optimize memory usage
    chunk_iter = pd.read_csv("eco2mix-regional-cons-def.csv", header=0, sep=";", chunksize=50000)

    # Initialize an empty list to store each processed chunk
    df_chunks = []

    # Process each chunk
    for chunk in chunk_iter:
        for chunk in chunk_iter:
            chunk = chunk[["Date", 'Consommation (MW)']]
            chunk['Date'] = pd.to_datetime(chunk['Date'])
            chunk['Consommation (MW)'] = chunk['Consommation (MW)'].fillna(mean_consumption)
            df_chunks.append(chunk)

    df_time_series = pd.concat(df_chunks)
    df_time_series = df_time_series.set_index("Date")
    df_time_series = df_time_series.resample('M').mean()
    df_time_series.columns = ["Consumption"]
    df_time_series = df_time_series.squeeze()

    # Display the final resampled data
    st.write(df_time_series)

    # Monthly Average Consumption Plot
    
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df_time_series.index, df_time_series.values, label="Monthly Average Consumption")

    ax.set_title("Time Series of Monthly Average Consumption")
    ax.set_xlabel("Date")
    ax.set_ylabel("Consumption (MW)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    # Display the plot in Streamlit
    st.pyplot(fig)

    
   

    # ## Seasonal Decompass
    

    st.write('Seasonal decomposition using the additive model')
    res_additive = seasonal_decompose(df_time_series)
    fig_additive = res_additive.plot()
    plt.tight_layout()
    st.pyplot(fig_additive)

    st.write ('Seasonal decomposition using the multiplicative model')
    res_multiplicative = seasonal_decompose(df_time_series, model='multiplicative')
    fig_multiplicative = res_multiplicative.plot()
    plt.tight_layout()
    st.pyplot(fig_multiplicative)

    st.write('Logarithm transformation')
    df_log = np.log(df_time_series)

    # Plot the transformed data
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_log)
    ax.set_title("Logarithm Transformation of Monthly Average Consumption")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log(Consumption)")
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)


    st.write ('Monthly Average Consumption with SARIMA Predictions')

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import statsmodels.api as sm

    def load_data():
        df = pd.read_csv("eco2mix-regional-cons-def.csv", header=0, sep=";")
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index("Date")
        df = df[['Consommation (MW)']].resample('M').mean()  
        df.columns = ["Consumption"]
        return df.squeeze()
    df_time_series = load_data()
    

    # Logarithmic transformation
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
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_pred, label="Observed + Predicted")
    ax.set_title("Monthly Average Consumption with SARIMA Predictions")
    ax.set_xlabel("Date")
    ax.set_ylabel("Consumption (MW)")
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)


elif selection == 'Conclusion':
    st.header("Conclusion")
    st.markdown('''
    **Key Insights:**
    -   **Correlation between Consumption and Production:** This project's overall goal was to develop a robust model capable of predicting energy consumption patterns to prevent possible blackouts. According to our exploratory and regression analyses, consumption and production are significantly correlated. As a result of this strong relationship, the impact of production on consumption patterns could be seen more clearly
    -   **Temporal Dynamics and COVID-19 Impact:** As we delve deeper into energy consumption's temporal dynamics, we found some distinct patterns. There was a noticeable drop in energy consumption in 2020. This reduction could be attributed to the unprecedented global situation caused by COVID-19. Due to countries around the world implementing strict lockdown measures to contain the virus, industrial activities slowed down, and many businesses temporarily ceased operations. As a result, energy demand decreased.
    -   **Seasonality in Consumption Patterns:** Our analysis also highlighted the seasonality inherent in energy consumption patterns. This is a crucial insight, emphasizing that energy consumption is not just influenced by exceptional global events like the pandemic but also by recurring annual cycles.
        ''', unsafe_allow_html=True)

    st.markdown("---")

    st.header("Perspectives")
    st.markdown('''
        Moving forward, we recommend focusing on the following areas for further analysis and model refinement:

    -   **Detailed Temporal Analysis:** While our current analysis has identified monthly/yearly patterns, diving deeper into daily consumption patterns could offer more detailed insights. This would help in fine-tuning predictions and in understanding short-term fluctuations better.
    -   **Incorporate Additional Variables:** External factors, such as temperature, public holidays, significant global events, or geographical differences can greatly influence energy consumption. Integrating these into the analysis could enhance the prediction accuracy of our model.
    -   **Scenario Planning:** Given the unpredictability of global events, as seen with the COVID-19 pandemic, it would be prudent to develop models that take into account different scenarios. This would ensure preparedness for unforeseen drops or surges in energy consumption.
        ''', unsafe_allow_html=True)
    
    
    
    
    
    
    
    
    







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





# # Finding the right granularity
# To go on with investigation the data should be transformed on a daily base. A monthly base would be an alternative choice. With a daily base we will have more records and expect to get better results.
