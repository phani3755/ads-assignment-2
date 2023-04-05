import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from matplotlib.colors import ListedColormap

cmap = ListedColormap(plt.cm.RdYlGn(np.linspace(0, 1, 256)) **2)

# aggregations for heat maps
green_house_df = pd.read_csv('data/green-house-emissions.csv')

# Select the relevant columns
cols = ['Country Name',   'Indicator Code', '2007', '2008', '2009', '2010', '2011', '2012']
green_house_df = green_house_df[cols]

# Melt the DataFrame to create a new DataFrame with the year as a separate column
green_house_melted = pd.melt(green_house_df, id_vars=['Country Name',   'Indicator Code'], var_name='Year', value_name='Total GHG Emissions (kt CO2 equivalent)')

green_house_grouped = green_house_melted.groupby(['Year', 'Country Name' ]).sum(numeric_only=True).reset_index()
#print(green_house_grouped.columns.values)

agri_land_df = pd.read_csv('data/agri-land.csv')

# Select the relevant columns
cols = ['Country Name',   'Indicator Code', '2007', '2008', '2009', '2010', '2011', '2012']
agri_land_df = agri_land_df[cols]

# Melt the DataFrame to create a new DataFrame with the year as a separate column
agri_land_melted = pd.melt(agri_land_df, id_vars=['Country Name',   'Indicator Code'], var_name='Year', value_name='Agricultural land (% of land area)')

agri_land_grouped = agri_land_melted.groupby(['Year', 'Country Name' ]).sum(numeric_only=True).reset_index()
#print(agri_land_grouped.columns.values)


access_to_electricity_df = pd.read_csv('data/access-to-electricity.csv')

# Select the relevant columns
cols = ['Country Name',   'Indicator Code', '2007', '2008', '2009', '2010', '2011', '2012']
access_to_electricity_df = access_to_electricity_df[cols]

# Melt the DataFrame to create a new DataFrame with the year as a separate column
access_to_electricity_melted = pd.melt(access_to_electricity_df, id_vars=['Country Name',   'Indicator Code'], var_name='Year', value_name='Access to electricity (% of population)')

access_to_electricity_grouped = access_to_electricity_melted.groupby(['Year', 'Country Name' ]).sum(numeric_only=True).reset_index()
#print(access_to_electricity_grouped.columns.values)


forest_area_df = pd.read_csv('data/forest-area.csv')

# Select the relevant columns
cols = ['Country Name',   'Indicator Code', '2007', '2008', '2009', '2010', '2011', '2012']
forest_area_df = forest_area_df[cols]

# Melt the DataFrame to create a new DataFrame with the year as a separate column
forest_area_melted = pd.melt(forest_area_df, id_vars=['Country Name',   'Indicator Code'], var_name='Year', value_name='Forest area (% of land area)')

forest_area_grouped = forest_area_melted.groupby(['Year', 'Country Name' ]).sum(numeric_only=True).reset_index()
#print(forest_area_grouped.columns.values)


population_growth_df = pd.read_csv('data/popl-growth.csv')

# Select the relevant columns
cols = ['Country Name',   'Indicator Code', '2007', '2008', '2009', '2010', '2011', '2012']
population_growth_df = population_growth_df[cols]

# Melt the DataFrame to create a new DataFrame with the year as a separate column
population_growth_melted = pd.melt(population_growth_df, id_vars=['Country Name',   'Indicator Code'], var_name='Year', value_name='Population growth (annual %)')

population_growth_grouped = population_growth_melted.groupby(['Year', 'Country Name' ]).sum(numeric_only=True).reset_index()
#print(population_growth_grouped.columns.values)

renewable_energy_consumption_df = pd.read_csv('data/renewable-energy-consumption.csv')

# Select the relevant columns
cols = ['Country Name',   'Indicator Code', '2007', '2008', '2009', '2010', '2011', '2012']
renewable_energy_consumption_df = renewable_energy_consumption_df[cols]

# Melt the DataFrame to create a new DataFrame with the year as a separate column
renewable_energy_consumption_melted = pd.melt(renewable_energy_consumption_df, id_vars=['Country Name',   'Indicator Code'], var_name='Year', value_name='Renewable energy consumption (% of total final energy consumption)')

renewable_energy_consumption_grouped = renewable_energy_consumption_melted.groupby(['Year', 'Country Name' ]).sum(numeric_only=True).reset_index()
#print(renewable_energy_consumption_grouped.columns.values)


# merging all data frames for getting different indicators
# creating a new column for using the combination of year and country


merged_df = pd.merge(green_house_grouped,renewable_energy_consumption_grouped,on=['Year','Country Name'])
merged_df = pd.merge(merged_df,agri_land_grouped,on=['Year','Country Name'])
merged_df = pd.merge(merged_df,access_to_electricity_grouped,on=['Year','Country Name'])
merged_df = pd.merge(merged_df,forest_area_grouped,on=['Year','Country Name'])
merged_df = pd.merge(merged_df,population_growth_grouped,on=['Year','Country Name'])


# GHG Bar chart

def ghg_emission_bar():
    """
        Displays bar plot for GreenHouse gas emissions of top 8 countries in the rankings of emissiosn
    """
    # Load the data from the CSV file
    df = pd.read_csv('data/gh-emissions.csv')

    # Select the relevant columns
    cols = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', '2007', '2008', '2009', '2010', '2011', '2012']
    df = df[cols]

    # Melt the DataFrame to create a new DataFrame with the year as a separate column
    melted = pd.melt(df, id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], var_name='Year', value_name='Total GHG Emissions (kt CO2 equivalent)')

    # Group the melted DataFrame by year and calculate the total greenhouse gas emissions for each year and country
    grouped = melted.groupby(['Year', 'Country Name', 'Country Code']).sum(numeric_only=True).reset_index()
    print(grouped.describe())
    # Sort the grouped DataFrame by total greenhouse gas emissions
    sorted_grouped = grouped.sort_values('Total GHG Emissions (kt CO2 equivalent)', ascending=False)
    # Select the top countries
    top_countries = ['China','North America','United States','India','Russian Federation','Japan','Brazil','Germany']
    # Filter the grouped DataFrame to include only the top 5 countries
    grouped = grouped[grouped['Country Name'].isin(top_countries)]
    # print(grouped)

    # Pivot the melted DataFrame to create a new DataFrame with years as columns, countries as rows, and total greenhouse gas emissions as values
    pivoted = grouped.pivot(index=['Country Name', 'Country Code'], columns='Year', values='Total GHG Emissions (kt CO2 equivalent)').reset_index()

    # Create a grouped bar chart
    ax = pivoted.plot(x='Country Name', kind='bar', width=0.8)
    ax.set_xlabel('Country')
    ax.set_ylabel('Total GHG Emissions (kt CO2 equivalent)')
    ax.set_title('Total Greenhouse Gas Emissions by Country and Year (2007-2012)')
    ax.legend(title='Year', ncol=1)

    plt.show()

def brazil_heat_map():
    """
        Displays a heatmap showing the correlation between Climate Change indicators
    """
    brazil_df = merged_df[merged_df['Country Name']== 'Brazil']
    brazil_df.rename(columns = {'Renewable energy consumption (% of total final energy consumption)':'Renewable energy consumption'}, inplace = True)
    brazil_df.rename(columns = {'Access to electricity (% of population)':'Access to electricity'}, inplace = True)
    brazil_df.rename(columns = {'Total GHG Emissions (kt CO2 equivalent)':'Total GHG Emissions'}, inplace = True)
    brazil_df.rename(columns = {'Agricultural land (% of land area)':'Agricultural land'}, inplace = True)

    brazil_df_correlation = brazil_df.corr()
    # sns.heatmap(brazil_df_correlation,square=True,vmax=1.0,cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) , center=0,annot = True)
    fig, ax = plt.subplots()
    heatmap = ax.imshow(brazil_df_correlation, cmap=cmap)
    columns = brazil_df_correlation.columns.values
    # set the x and y axis labels
    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(columns)))
    ax.set_xticklabels(columns)
    ax.set_yticklabels(columns)

    # rotate the x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # add text annotations with correlation values
    for i in range(len(columns)):
        for j in range(len(columns)):
            text = ax.text(j, i, round(brazil_df_correlation.iloc[i, j], 2), ha="center", va="center", color="w")

    # add a colorbar legend
    plt.colorbar(heatmap)
    plt.title('Brazil')
    plt.show()

def india_heat_map():
    """
        Displays a heatmap showing the correlation between Climate Change indicators
    """
    india_df = merged_df[merged_df['Country Name']== 'India']
    india_df.rename(columns = {'Renewable energy consumption (% of total final energy consumption)':'Renewable energy consumption'}, inplace = True)
    india_df.rename(columns = {'Access to electricity (% of population)':'Access to electricity'}, inplace = True)
    india_df.rename(columns = {'Total GHG Emissions (kt CO2 equivalent)':'Total GHG Emissions'}, inplace = True)
    india_df.rename(columns = {'Agricultural land (% of land area)':'Agricultural land'}, inplace = True)

    india_df_correlation = india_df.corr()
    # sns.heatmap(india_df_correlation,square=True,vmax=1.0,cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) , center=0,annot = True)
    fig, ax = plt.subplots()
    heatmap = ax.imshow(india_df_correlation, cmap=cmap)
    columns = india_df_correlation.columns.values
    # set the x and y axis labels
    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(columns)))
    ax.set_xticklabels(columns)
    ax.set_yticklabels(columns)

    # rotate the x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # add text annotations with correlation values
    for i in range(len(columns)):
        for j in range(len(columns)):
            text = ax.text(j, i, round(india_df_correlation.iloc[i, j], 2), ha="center", va="center", color="w")

    # add a colorbar legend
    plt.colorbar(heatmap)
    plt.title('India')
    plt.show()

def china_heat_map():
    """
        Displays a heatmap showing the correlation between Climate Change indicators
    """
    china_df = merged_df[merged_df['Country Name']== 'China']
    china_df.rename(columns = {'Renewable energy consumption (% of total final energy consumption)':'Renewable energy consumption'}, inplace = True)
    china_df.rename(columns = {'Access to electricity (% of population)':'Access to electricity'}, inplace = True)
    china_df.rename(columns = {'Total GHG Emissions (kt CO2 equivalent)':'Total GHG Emissions'}, inplace = True)
    china_df.rename(columns = {'Agricultural land (% of land area)':'Agricultural land'}, inplace = True)

    china_df_correlation = china_df.corr()
    # sns.heatmap(china_df_correlation,square=True,vmax=1.0,cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) , center=0,annot = True)
    fig, ax = plt.subplots()
    heatmap = ax.imshow(china_df_correlation, cmap=cmap)
    columns = china_df_correlation.columns.values
    # set the x and y axis labels
    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(columns)))
    ax.set_xticklabels(columns)
    ax.set_yticklabels(columns)

    # rotate the x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # add text annotations with correlation values
    for i in range(len(columns)):
        for j in range(len(columns)):
            text = ax.text(j, i, round(china_df_correlation.iloc[i, j], 2), ha="center", va="center", color="w")

    # add a colorbar legend
    plt.colorbar(heatmap)
    plt.title('China')
    plt.show()

def renewable_energy_bar():
    """
        Displays a bar plot of comparing renewable energy consumption for top 8 coutries in terms of GHG emissions.
    """
    # Load the data from the CSV file
    df = pd.read_csv('data/renewable-energy-consumption.csv')
    # df = pd.read_csv('data/popul-growth.csv')

    # Select the relevant columns
    cols = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', '2007', '2008', '2009', '2010', '2011', '2012']
    df = df[cols]

    # Melt the DataFrame to create a new DataFrame with the year as a separate column
    melted = pd.melt(df, id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], var_name='Year', value_name='Renewable energy consumption (% of total final energy consumption)')
    # melted = pd.melt(df, id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], var_name='Year', value_name='Population, total')


    # Group the melted DataFrame by year and calculate the total greenhouse gas emissions for each year and country
    grouped = melted.groupby(['Year', 'Country Name', 'Country Code']).sum().reset_index()

    grouped = grouped.loc[grouped['Country Name'].isin(['China','North America','United States','India','Russian Federation','Japan','Brazil','Germany'])]


    # print(grouped)

    # Pivot the melted DataFrame to create a new DataFrame with years as columns, countries as rows, and total greenhouse gas emissions as values
    pivoted = grouped.pivot(index=['Country Name', 'Country Code'], columns='Year', values='Renewable energy consumption (% of total final energy consumption)').reset_index()

    # Create a grouped bar chart
    ax = pivoted.plot(x='Country Name', kind='bar', width=0.8)
    ax.set_xlabel('Country')
    ax.set_ylabel('Renewable energy consumption (% of total final energy consumption)')
    ax.set_title('Renewable energy consumption (% of total final energy consumption) by Country and Year (2007-2012)')
    ax.legend(title='Year', ncol=6)


    plt.show()

def forest_area_line():
    """
     Displays a line plot for showing forest area by % for different countries.
    """
    # Load the data from the CSV file
    df = pd.read_csv('data/forest-area.csv')

    # Select the relevant columns
    cols = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', '2007', '2008', '2009', '2010', '2011', '2012']
    df = df[cols]

    # Melt the DataFrame to create a new DataFrame with the year as a separate column
    melted = pd.melt(df, id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], var_name='Year', value_name='Forest area (% of land area)')

    # Group the melted DataFrame by year and calculate the total greenhouse gas emissions for each year and country
    grouped = melted.groupby(['Year', 'Country Name', 'Country Code']).sum().reset_index()

    # Sort the grouped DataFrame by total greenhouse gas emissions
    sorted_grouped = grouped.sort_values('Forest area (% of land area)', ascending=False)



    # Filter the grouped DataFrame to include only the top 5 countries
    grouped = grouped[grouped['Country Name'].isin(['China','North America','United States','India','Russian Federation','Japan','Brazil','Germany'])]
    # print(grouped)

    # Create the plot
    fig, ax = plt.subplots()

    # Loop through each country and plot a line for each year
    for name, group in grouped.groupby('Country Name'):
        group.plot(x='Year', y='Forest area (% of land area)', ax=ax, label=name)

    # Set the x-axis label
    ax.set_xlabel('Year')

    # Set the y-axis label
    ax.set_ylabel('Forest area (% of land area)')
    ax.set_title('Forest Area by Year and Country 2007 - 2012')
    # Show the legend
    ax.legend()

    # Show the plot
    plt.show()

def population_line():
    """
     Displays a line plot for showing population by % for different countries.
    """
    # Load the data from the CSV file
    df = pd.read_csv('data/popl-growth.csv')

    # Select the relevant columns
    cols = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', '2007', '2008', '2009', '2010', '2011', '2012']
    df = df[cols]

    # Melt the DataFrame to create a new DataFrame with the year as a separate column
    melted = pd.melt(df, id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], var_name='Year', value_name='Population, total')

    # Group the melted DataFrame by year and calculate the total greenhouse gas emissions for each year and country
    grouped = melted.groupby(['Year', 'Country Name', 'Country Code']).sum().reset_index()

    # Sort the grouped DataFrame by total greenhouse gas emissions
    sorted_grouped = grouped.sort_values('Population, total', ascending=False)



    # Filter the grouped DataFrame to include only the top 5 countries
    grouped = grouped[grouped['Country Name'].isin(['China','North America','United States','India','Russian Federation','Japan','Brazil','Germany'])]
    # print(grouped)

    # Create the plot
    fig, ax = plt.subplots()

    # Loop through each country and plot a line for each year
    for name, group in grouped.groupby('Country Name'):
        group.plot(x='Year', y='Population, total', ax=ax, label=name)

    # Set the x-axis label
    ax.set_xlabel('Year')

    # Set the y-axis label
    ax.set_ylabel('Population, total')

    # Show the legend
    ax.legend()

    ax.set_title('Population by Country and Year between (2007 and 2012)')

    # Show the plot
    plt.show()

def ghg_pie():
    """
        Pie chart for showing different types of GHG contributing overall emissions
    """
    co2_df = pd.read_csv('data/co2.csv')
    # Select the relevant columns
    cols = ['Country Name',  'Indicator Code', '2007', '2008', '2009', '2010', '2011', '2012']
    co2_df = co2_df[cols]

    # Melt the DataFrame to create a new DataFrame with the year as a separate column
    co2_melted = pd.melt(co2_df, id_vars=['Country Name',   'Indicator Code'], var_name='Year', value_name='CO2 emissions (kt)')

    co2_grouped = co2_melted.groupby(['Year', 'Country Name' ]).sum(numeric_only=True).reset_index()
    #print(co2_grouped.columns.values)

    meth_oxide_df = pd.read_csv('data/meth-oxide.csv')

    # Select the relevant columns
    cols = ['Country Name',   'Indicator Code', '2007', '2008', '2009', '2010', '2011', '2012']
    meth_oxide_df = meth_oxide_df[cols]

    # Melt the DataFrame to create a new DataFrame with the year as a separate column
    meth_oxide_melted = pd.melt(meth_oxide_df, id_vars=['Country Name',   'Indicator Code'], var_name='Year', value_name='Methane emissions (kt of CO2 equivalent)')

    meth_oxide_grouped = meth_oxide_melted.groupby(['Year', 'Country Name' ]).sum(numeric_only=True).reset_index()
    #print(meth_oxide_grouped.columns.values)


    n_oxide_df = pd.read_csv('data/n-oxide.csv')

    # Select the relevant columns
    cols = ['Country Name',   'Indicator Code', '2007', '2008', '2009', '2010', '2011', '2012']
    n_oxide_df = n_oxide_df[cols]

    # Melt the DataFrame to create a new DataFrame with the year as a separate column
    n_oxide_melted = pd.melt(n_oxide_df, id_vars=['Country Name',   'Indicator Code'], var_name='Year', value_name='Nitrous oxide emissions (thousand metric tons of CO2 equivalent)')

    n_oxide_grouped = n_oxide_melted.groupby(['Year', 'Country Name' ]).sum(numeric_only=True).reset_index()
    #print(n_oxide_grouped.columns.values)


    merged_df = pd.merge(n_oxide_grouped,meth_oxide_grouped,on=['Country Name','Year'])
    merged_df = pd.merge(merged_df,co2_grouped,on=['Country Name','Year'])

    merged_df_grouped = merged_df.sum(numeric_only=True).reset_index()
    values = merged_df_grouped[0]
    merged_df_grouped = merged_df.sum(numeric_only=True)
    plt.pie(merged_df_grouped,labels=['Nitrous oxide emissions','Methane emissions','CO2 emissions'],autopct= lambda x: '{:.0f}'.format(x*values.sum()/100))
    plt.legend(title='Emission Type', ncol=1,loc="lower left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.title('Total emissions by all countries by types of Emissions(2007-2012)')
    plt.show()


# Calling all the chart functions

ghg_emission_bar() #Greenhouse emissions bar chart for top countries
brazil_heat_map() # Correlation for indicators in Brazil
renewable_energy_bar() # Renewable energy consumption between countries comaprision
china_heat_map() # Correlation for indicators in Brazil
forest_area_line() # Forest area by land for different countries
india_heat_map() # Correlation for indicators in Brazil
population_line() # population growth for countries
ghg_pie() # Greenhouse gas emissions segregation by Gas

