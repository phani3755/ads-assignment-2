import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from matplotlib.colors import ListedColormap

cmap = ListedColormap(plt.cm.RdYlGn(np.linspace(0, 1, 256)) ** 2)


def merge_df():
    # aggregations for heat maps
    green_house_df = pd.read_csv('green-house-emissions.csv')

    # Select the relevant columns
    cols = ['Country Name',   'Indicator Code',
            '2007', '2008', '2009', '2010', '2011', '2012']
    green_house_df = green_house_df[cols]

    # Melt  DataFrame to create new DataFrame with the year as separate column
    green_house_melted = pd.melt(green_house_df, id_vars=[
                                 'Country Name',   'Indicator Code'],
        var_name='Year', value_name='Total GHG Emissions (kt CO2 equivalent)')

    green_house_grouped = green_house_melted.groupby(
        ['Year', 'Country Name']).sum(numeric_only=True).reset_index()
    # print(green_house_grouped.columns.values)

    agri_land_df = pd.read_csv('agri-land.csv')

    # Select the relevant columns
    cols = ['Country Name',   'Indicator Code',
            '2007', '2008', '2009', '2010', '2011', '2012']
    agri_land_df = agri_land_df[cols]

    # Melt the DataFrame to create a new DataFrame with the year 
    #as a separate column
    agri_land_melted = pd.melt(agri_land_df, id_vars=[
                               'Country Name',   'Indicator Code'],
        var_name='Year', value_name='Agricultural land (% of land area)')

    agri_land_grouped = agri_land_melted.groupby(
        ['Year', 'Country Name']).sum(numeric_only=True).reset_index()
    # print(agri_land_grouped.columns.values)

    access_to_electricity_df = pd.read_csv('access-to-electricity.csv')

    # Select the relevant columns
    cols = ['Country Name',   'Indicator Code',
            '2007', '2008', '2009', '2010', '2011', '2012']
    access_to_electricity_df = access_to_electricity_df[cols]

    # Melt the DataFrame to create a new DataFrame with the year 
    #as a separate column
    access_to_electricity_melted = pd.melt(access_to_electricity_df, id_vars=[
                                           'Country Name',
        'Indicator Code'], var_name='Year',
        value_name='Access to electricity (% of population)')

    access_to_electricity_grouped = access_to_electricity_melted.groupby(
        ['Year', 'Country Name']).sum(numeric_only=True).reset_index()
    # print(access_to_electricity_grouped.columns.values)

    forest_area_df = pd.read_csv('forest-area.csv')

    # Select the relevant columns
    cols = ['Country Name',   'Indicator Code',
            '2007', '2008', '2009', '2010', '2011', '2012']
    forest_area_df = forest_area_df[cols]

    # Melt the DataFrame to create a new DataFrame with the year 
    #as a separate column
    forest_area_melted = pd.melt(forest_area_df, id_vars=[
                                 'Country Name',
        'Indicator Code'], var_name='Year',
        value_name='Forest area (% of land area)')

    forest_area_grouped = forest_area_melted.groupby(
        ['Year', 'Country Name']).sum(numeric_only=True).reset_index()
    # print(forest_area_grouped.columns.values)

    population_growth_df = pd.read_csv('popl-growth.csv')

    # Select the relevant columns
    cols = ['Country Name',   'Indicator Code',
            '2007', '2008', '2009', '2010', '2011', '2012']
    population_growth_df = population_growth_df[cols]

    # Melt the DataFrame to create a new DataFrame with the year
    #as a separate column
    population_growth_melted = pd.melt(population_growth_df, id_vars=[
                                       'Country Name',
        'Indicator Code'], var_name='Year',
        value_name='Population growth (annual %)')

    population_growth_grouped = population_growth_melted.groupby(
        ['Year', 'Country Name']).sum(numeric_only=True).reset_index()
    # print(population_growth_grouped.columns.values)

    renewable_energy_consumption_df = \
        pd.read_csv('renewable-energy-consumption.csv')

    # Select the relevant columns
    cols = ['Country Name',   'Indicator Code',
            '2007', '2008', '2009', '2010', '2011', '2012']
    renewable_energy_consumption_df = renewable_energy_consumption_df[cols]

    # Melt the DataFrame to create a new DataFrame with the year 
    #as a separate column
    renewable_energy_consumption_melted = \
    pd.melt(renewable_energy_consumption_df, id_vars=[
    'Country Name',   'Indicator Code'],
        var_name='Year',
        value_name='Renewable energy consumption \
            (% of total final energy consumption)')

    renewable_energy_consumption_grouped =\
        renewable_energy_consumption_melted.groupby(
        ['Year', 'Country Name']).sum(numeric_only=True).reset_index()

    merged_df = pd.merge(green_house_grouped,
                         renewable_energy_consumption_grouped, on=[
                         'Year', 'Country Name'])
    merged_df = pd.merge(merged_df, agri_land_grouped,
                         on=['Year', 'Country Name'])
    merged_df = pd.merge(merged_df, access_to_electricity_grouped, on=[
                         'Year', 'Country Name'])
    merged_df = pd.merge(merged_df, forest_area_grouped,
                         on=['Year', 'Country Name'])
    merged_df = pd.merge(merged_df, population_growth_grouped, on=[
                         'Year', 'Country Name'])

    return merged_df

def bar_plot(filename, indicator):
    """
        Parameters
        -----------
        filename : STR
            csv file name to create dataframe.
        indicator : STR
            takes in input indicator for making bar plot
    """
    # Load the data from the CSV file
    df = pd.read_csv(filename)

    # Select the relevant columns
    cols = ['Country Name', 'Country Code', 'Indicator Name',
            'Indicator Code', '2007', '2008', '2009', '2010', '2011', '2012']
    df = df[cols]

    # Melt  DataFrame to create new DataFrame with the year  separate column
    melted = pd.melt(df, id_vars=['Country Name', 'Country Code',
    'Indicator Name','Indicator Code'], var_name='Year', value_name=indicator)

    # Group the melted DataFrame by year and calculate the total greenhouse 
    #gas emissions for each year and country
    grouped = melted.groupby(['Year', 'Country Name', 'Country Code']).sum(
        numeric_only=True).reset_index()
    print(grouped.describe())
    # Sort the grouped DataFrame by total greenhouse gas emissions
    sorted_grouped = grouped.sort_values(indicator, ascending=False)
    # Select the top countries
    top_countries = ['China', 'North America', 'United States',
                    'India', 'Russian Federation', 'Japan', 'Brazil','Germany']
    # Filter the grouped DataFrame to include only the top 5 countries
    grouped = grouped[grouped['Country Name'].isin(top_countries)]
    # print(grouped)

    # Pivot the melted DataFrame to create a new DataFrame with years
    #as columns, countries as rows,and total greenhouse gas emissions as values
    pivoted = grouped.pivot(index=[
    'Country Name', 'Country Code'], columns='Year',
        values=indicator).reset_index()

    # Create a grouped bar chart
    ax = pivoted.plot(x='Country Name', kind='bar', width=0.8)
    ax.set_xlabel('Country')
    ax.set_ylabel(indicator)
    ax.set_title(indicator + 'by Country and Year (2007-2012)')
    ax.legend(title='Year', ncol=1)

    plt.show()


def heat_map(merged_df, country):
    """
        Parameters
        -----------
        merged_df : pandas.Dataframe
            df is a supertore of csv file.
        country : STR
            takes in input country for making heat map
    """
    merged_df = merged_df[merged_df['Country Name'] == country]
    # renaming columns for
    merged_df.rename(columns={
    'Renewable energy consumption (% of total final energy consumption)':\
        'Renewable energy consumption'}, inplace=True)
    merged_df.rename(columns={
    'Access to electricity (% of population)':\
        'Access to electricity'}, inplace=True)
    merged_df.rename(columns={
    'Total GHG Emissions (kt CO2 equivalent)':\
        'Total GHG Emissions'}, inplace=True)
    merged_df.rename(
        columns={'Agricultural land (% of land area)':\
                 'Agricultural land'}, inplace=True)

    merged_df_correlation = merged_df.corr()
    
    fig, ax = plt.subplots()
    heatmap = ax.imshow(merged_df_correlation, cmap=cmap)
    columns = merged_df_correlation.columns.values
    # set the x and y axis labels
    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(columns)))
    ax.set_xticklabels(columns)
    ax.set_yticklabels(columns)

    # rotate the x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")
    # add text annotations with correlation values
    for i in range(len(columns)):
        for j in range(len(columns)):
            text = ax.text(j, i, round(
                merged_df_correlation.iloc[i, j], 2), ha="center",
                va="center", color="w")

    # add a colorbar legend
    plt.colorbar(heatmap)
    plt.title(country)
    plt.show()



def line_plot(filename, indicator):
    """
        Parameters
        -----------
        filename : STR
            csv file name to create dataframe.
        indicator : STR
            takes in input indicator for making line plot
    """
    # Load the data from the CSV file
    df = pd.read_csv(filename)

    # Select the relevant columns
    cols = ['Country Name', 'Country Code', 'Indicator Name',
            'Indicator Code', '2007', '2008', '2009', '2010', '2011', '2012']
    df = df[cols]

   
    melted = pd.melt(df, id_vars=['Country Name', 'Country Code',
    'Indicator Name','Indicator Code'], var_name='Year', value_name=indicator)

    # Group the melted DataFrame by year and calculate the total
    #greenhouse gas emissions for each year and country
    grouped = melted.groupby(
        ['Year', 'Country Name', 'Country Code']).sum().reset_index()

    # Sort the grouped DataFrame by total greenhouse gas emissions
    sorted_grouped = grouped.sort_values(indicator, ascending=False)

    # Filter the grouped DataFrame to include only the top 5 countries
    grouped = grouped[grouped['Country Name'].isin(
        ['China', 'North America', 'United States', 'India',
         'Russian Federation', 'Japan', 'Brazil', 'Germany'])]
    # print(grouped)

    # Create the plot
    fig, ax = plt.subplots()

    # Loop through each country and plot a line for each year
    for name, group in grouped.groupby('Country Name'):
        group.plot(x='Year', y=indicator, ax=ax, label=name)

    # Set the x-axis label
    ax.set_xlabel('Year')

    # Set the y-axis label
    ax.set_ylabel(indicator)
    ax.set_title(indicator+ 'by Year and Country 2007 - 2012')
    # Show the legend
    ax.legend()

    # Show the plot
    plt.show()


def pie_chart(filename):
    """
        Parameters
        -----------
        filename : STR
            csv file name to create dataframe.
    """
    co2_df = pd.read_csv(filename)
    # Select the relevant columns
    cols = ['Country Name',  'Indicator Code', '2007',
            '2008', '2009', '2010', '2011', '2012']
    co2_df = co2_df[cols]

    # Melt the DataFrame to create a new DataFrame with the year
    #as a separate column
    co2_melted = pd.melt(co2_df, id_vars=[
                         'Country Name',   'Indicator Code'], var_name='Year',
        value_name='CO2 emissions (kt)')

    co2_grouped = co2_melted.groupby(['Year', 'Country Name']).sum(
        numeric_only=True).reset_index()
    # print(co2_grouped.columns.values)

    meth_oxide_df = pd.read_csv('meth-oxide.csv')

    # Select the relevant columns
    cols = ['Country Name',   'Indicator Code',
            '2007', '2008', '2009', '2010', '2011', '2012']
    meth_oxide_df = meth_oxide_df[cols]

    # Melt the DataFrame to create a new DataFrame with the year 
    #as a separate column
    meth_oxide_melted = pd.melt(meth_oxide_df, id_vars=[
    'Country Name',   'Indicator Code'], var_name='Year',
        value_name='Methane emissions (kt of CO2 equivalent)')

    meth_oxide_grouped = meth_oxide_melted.groupby(
        ['Year', 'Country Name']).sum(numeric_only=True).reset_index()
    # print(meth_oxide_grouped.columns.values)

    n_oxide_df = pd.read_csv('n-oxide.csv')

    # Select the relevant columns
    cols = ['Country Name',   'Indicator Code',
            '2007', '2008', '2009', '2010', '2011', '2012']
    n_oxide_df = n_oxide_df[cols]

    # Melt the DataFrame to create a new DataFrame with the year
    #as a separate column
    n_oxide_melted = pd.melt(n_oxide_df, id_vars=['Country Name', 
    'Indicator Code'], var_name='Year',
    value_name='Nitrous oxide emissions \
        (thousand metric tons of CO2 equivalent)')

    n_oxide_grouped = n_oxide_melted.groupby(
        ['Year', 'Country Name']).sum(numeric_only=True).reset_index()
    # print(n_oxide_grouped.columns.values)

    merged_df = pd.merge(n_oxide_grouped, meth_oxide_grouped, on=[
                         'Country Name', 'Year'])
    merged_df = pd.merge(merged_df, co2_grouped, on=['Country Name', 'Year'])

    merged_df_grouped = merged_df.sum(numeric_only=True).reset_index()
    values = merged_df_grouped[0]
    merged_df_grouped = merged_df.sum(numeric_only=True)
    plt.pie(merged_df_grouped, labels=['Nitrous oxide emissions',
                                       'Methane emissions',
            'CO2 emissions'], autopct=lambda x: '{:.0f}'\
            .format(x*values.sum()/100))
    plt.legend(title='Emission Type', ncol=1,
               loc="lower left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.title('Total emissions by all countries by \
              types of Emissions(2007-2012)')
    plt.show()


def read_worldbank_data(filename):
    """
        Parameters
        -----------
        filename : STR
            csv file name to create dataframe.
        Returns 
        --------
        year_df : pandas.Dataframe
            it is a superstore for csv file
        country_df : pandas.Dataframe
            It is a superstore for csv file
    """
    # Read in the Worldbank data file as a pandas dataframe
    data = pd.read_csv(filename)

    # Transpose the dataframe to have years as columns and countries as rows
    data_transposed = data.transpose()

    # Extract the row containing the country names
    country_row = data_transposed.iloc[0]

    # Remove the row with country names from the transposed dataframe
    data_transposed = data_transposed[1:]

    # Set the country names as the column headers for the transposed dataframe
    data_transposed.columns = country_row
    data = data.drop('Unnamed: 66', axis='columns')
    year_df = data
    country_df = data_transposed
    return country_df, year_df

# country_df,year_df = read_worldbank_data('./data/co2.csv')
# print(country_df.columns.values)
# print(year_df.columns.values)

# Example funtion calls for report charts
merged_data_frame = merge_df()
# bar plot for GHG emissions
bar_plot('green-house-emissions.csv',
         'Total greenhouse gas emissions (kt of CO2 equivalent)')
# heat map for china
heat_map(merged_data_frame,'China')
# bar plot for renewable energy cosumption
bar_plot('renewable-energy-consumption.csv',
         'Renewable energy consumption (% of total final energy consumption)')
# heat map for Brazil
heat_map(merged_data_frame,'Brazil')
# line plot for population growth
line_plot('popl-growth.csv',
          'Population, total')
# heatmap for India
heat_map(merged_data_frame,'India')
# line plot for forest area
line_plot('forest-area.csv',
          'Forest area (% of land area)')
# pie chart for GHG Emission by Gas
pie_chart('green-house-emissions.csv')