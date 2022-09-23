import numpy as np
import pandas as pd
# datetime utilities
from datetime import timedelta, datetime
# visualization
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_merged_data(df):
    """
        This function will take in the merged_data dataframe and return a prepped dataframe
    """
    #Convert the Date column to a datetime format
    df.sale_date = pd.to_datetime(df.sale_date)
    
    #Add a month and a year column to dataframe
    df['month'] = df.sale_date.dt.month
    df['year'] = df.sale_date.dt.year
    
    #set datatime as index
    df = df.set_index('sale_date').sort_index()
    
    return df