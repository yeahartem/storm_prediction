import pandas as pd
import numpy as np

def preprocess_grouped(grouped, days_in_period=25, periods_in_a_row=6):
    """
    Returns continuous date periods of a grouped DataFrame.

    grouped - DataFrame grouped by 'STATION';
    days_in_period - minimum number of days in a 4-week period to count this period as continuous;
    periods_in_a_row - minimum number of periods in a row to save this interval to output.
    """
    cont = 0
    previous_was_okay = 0
    list_of_dates = []
    start_date = grouped.index[0]
    for (date, i) in zip(grouped.index, grouped.STATION):
        if i >= days_in_period:                            # try >=20 and ==28     Days in a 4-week period, should be 28
            cont += 1
            if cont >= periods_in_a_row:            # try >=3 and >=12      Continuous periods in a row
                end_date = date
                previous_was_okay = 1                      # Flag that says that previous 4-week period was continuous
        
        else: 
            cont = 0
            if previous_was_okay:
                list_of_dates.append([start_date, end_date])
                previous_was_okay = 0
            start_date = date
    if previous_was_okay:
        list_of_dates.append([start_date, end_date])     
    return list_of_dates

def drop_discrete_data(df, days_in_period, periods_in_a_row):
    """
    Drops discrete data from the full DataFrame, returns final version of DataFrame.

    df - full DataFrame to be cleaned;
    days_in_period - minimum number of days in a 4-week period to count this period as continuous;
    periods_in_a_row - minimum number of periods in a row to save this interval to output.    
    """
    df_reduced = df[['DATE', 'STATION']]

    gg = df_reduced.groupby('STATION')
    indices_final = []
    for one_station in gg:
        print(one_station[0])
        grouped = one_station[1].groupby([pd.Grouper(key='DATE', freq='4W')]).count()
        cont_periods = preprocess_grouped(grouped, days_in_period=days_in_period, periods_in_a_row=periods_in_a_row)
        if cont_periods:

            mask = one_station[1].DATE.between(cont_periods[0][0], cont_periods[0][1])
            for i in cont_periods:
                mask += one_station[1].DATE.between(i[0], i[1])
            indices_final.extend(one_station[1].index[mask].tolist())

    indices_to_drop = np.delete(np.array(df_reduced.index), indices_final, None)

    df.drop(index=indices_to_drop, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def fahr_to_celsius(temp_fahr):
    """Convert Fahrenheit to Celsius.
    
    Return Celsius conversion of input"""
    return (temp_fahr - 32) * 5 / 9

def knots_to_ms(windspeed):
    """Convert Knots to m/s.
    
    Return m/s conversion of input"""
    return windspeed * 0.51444

def preprocess_csv(title):
    """Preprocess a .csv file.
    
    Return preprocessed .csv of a station"""
    
    df = pd.read_csv(title)

    df = df[['DATE', 'STATION', 'NAME', 'GUST', 'WDSP', 'TEMP', 'STP', 'SLP', 'PRCP', 'DEWP', 'MXSPD', 'LATITUDE', 'LONGITUDE', 'ELEVATION']]

    df = df.replace([-999.9, -999., 99.99, 999.9, 9999.9], np.nan) 

    df['DATE'] = pd.to_datetime(df['DATE'])

    # Speed
    df["MAXWDSP"] = df[["GUST", "MXSPD"]].max(axis=1)
    df.insert(3, 'MXWDSP', df.pop("MAXWDSP"))
    df.drop('GUST', axis=1, inplace=True)
    df.drop('MXSPD', axis=1, inplace=True)

    # Temperature
    df['TEMP'] = df['TEMP'].apply(fahr_to_celsius)
    df['DEWP'] = df['DEWP'].apply(fahr_to_celsius)

    df['MXWDSP'] = df['MXWDSP'].apply(knots_to_ms)
    df['WDSP'] = df['WDSP'].apply(knots_to_ms)

    df['STP'] = df['STP'].apply(lambda x: (x + 1000) if x < 100 else x)
    # df['SLP'] = df['SLP'].apply(lambda x: (x + 1000) if x < 100 else x)

    df['PRCP'] = df['PRCP'] * 2.54

    return df