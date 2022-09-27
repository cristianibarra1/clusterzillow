import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")
from env import user, password, host
import acquire
import prepare
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import os 

#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#acquire

def acquire_global():
    '''acquire from a csv file downloaded from kaggle'''
    df=pd.read_csv('GlobalLandTemperaturesByCountry.csv')
    return df

#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#prepare

def prepare_global(df,k):
    '''preparing df by 
    creating a year using datime 
    converting cel in fer
    removing all of nulls 
    only getting greenland from country'''
    df['year'] = pd.to_datetime( df['dt']).dt.year
    df['Fahrenheit'] = df.apply(lambda x: (9/5)*x['AverageTemperature']+32,axis=1)
    df['Fahrenheit_Uncertainty'] = df.apply(lambda x: (9/5)*x['AverageTemperatureUncertainty']+32,axis=1)
    col_list=['AverageTemperature','AverageTemperatureUncertainty','Fahrenheit','Fahrenheit_Uncertainty']
    df.dt = pd.to_datetime(df.dt)
    df = df.set_index('dt')
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
    df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
    df = df.loc[(df['Country'] == 'Greenland')]
    df=df.fillna(0)
    return df     
 
def split_timeseries(df):
    '''splitting data into train,val,test'''
    train_size = int(len(df) * .5)
    validate_size = int(len(df) * .3)
    test_size = int(len(df) - train_size - validate_size)
    validate_end_index = train_size + validate_size

    # split into train, validation, test
    train = df[: train_size]
    val = df[train_size : validate_end_index]
    test = df[validate_end_index : ]
    
    return train,val,test


#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------
#graphs

plt.rcParams["figure.figsize"] = (25,16)
def sea_born1(train):
    '''graphs Fahrenheit months'''
    #average temp by months()
    sns.violinplot(data = train, x = train.index.month, y = 'Fahrenheit')
    plt.title("Average temperture by months")
    plt.xlabel("Months")
    plt.ylabel("Average temperature")
    plt.plot()

def Greenland_graph(train):
    '''graphs Fahrenheit_Uncertainty'''
    #mean temp for greenland
    Greenland = train[train['Country']=='Greenland']
    plt.rcParams["figure.figsize"] = (25,16)
    new_Greenland = Greenland.groupby('year')['Fahrenheit_Uncertainty'].mean().reset_index()
    new_Greenland.plot(kind='line', x='year', y='Fahrenheit_Uncertainty', title='MeanTemperature trend in Greenland')
    
def Greenland_graph4(train):
    '''graphs Fahrenheit years'''
    train.resample('Y').Fahrenheit.plot()
    plt.rcParams["figure.figsize"] = (25,16)
    plt.show()

    
def Greenland_graph6(train):
    '''graphs Fahrenheit 6months'''
    train.resample('6M').mean().Fahrenheit.plot(title='Fahrenheit',
                                         xlabel='Year',
                                         ylabel='Temperature(F)')
    plt.legend()
    plt.show()

#--------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------stats:

# statistics
from scipy import stats
    
def stats0(train):
    '''#T-Test: Two-sample, Two-tail'''
    before = train[:'1905'].Fahrenheit
    after = train['1905':].Fahrenheit
    α = 0.05

    s, pval = stats.levene(before, after)
    t, p = stats.ttest_ind(before, after, equal_var=(pval > α))

    if p < α:
        print('Reject the Null Hypothesis. different')
    else:
        print('Fail to reject the Null Hypothesis')
        
def stats2(train):
    '''#T-Test: Two-sample, Two-tail'''
    before = train[:'1905'].Fahrenheit_Uncertainty
    after = train['1905':].Fahrenheit_Uncertainty
    α = 0.05

    s, pval = stats.levene(before, after)
    t, p = stats.ttest_ind(before, after, equal_var=(pval > α))

    if p < α:
        print('Reject the Null Hypothesis. different')
    else:
        print('Fail to reject the Null Hypothesis')
    
    
def stats3(train):
    '''#T-Test: Two-sample, Two-tail'''
    before = train[:'1905'].Fahrenheit
    after = train['1905':].Fahrenheit
    α = 0.05

    s, pval = stats.levene(before, after)
    t, p = stats.ttest_ind(before, after, equal_var=(pval > α))

    if p < α:
        print('Reject the Null Hypothesis. different')
    else:
        print('Fail to reject the Null Hypothesis')
        
def stats4(train):
    '''#T-Test: Two-sample, Two-tail'''
    before = train[:'1900'].Fahrenheit
    after = train['1900':].Fahrenheit
    α = 0.05

    s, pval = stats.levene(before, after)
    t, p = stats.ttest_ind(before, after, equal_var=(pval > α))

    if p < α:
        print('Reject the Null Hypothesis. different')
    else:
        print('Fail to reject the Null Hypothesis')
    
#--------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------Modeling:


# dates
from datetime import datetime

# evaluated performance rmse
from sklearn.metrics import mean_squared_error
from math import sqrt 

# for tsa 
import statsmodels.api as sm

# holt's linear trend model. 
from statsmodels.tsa.api import Holt




def evaluate(target_var, val, yhat):
    '''
    This function will take the actual values of the target_var from validate, 
    and the predicted values , and compute the rmse. it will return the rmse. 
    '''
    rmse = round(mean_squared_error(val[target_var], yhat[target_var], squared=False), 0)
    return rmse


def plot_and_eval(target_var, train, val, yhat, title):
    '''
    This function takes in the target var name (string),
    returns a plot of the values of train for that variable, 
    validate, and the predicted values. 
    it will als lable the rmse. 
    '''
    plt.figure(figsize = (18,8))
    plt.plot(train[target_var], label='Train', linewidth=1)
    plt.plot(val[target_var], label='Validate', linewidth=1)
    plt.plot(yhat[target_var], label='Prediction', linewidth=1)
    plt.title(title)
    rmse = evaluate(target_var, val, yhat)
    plt.legend()
    plt.show()
    
    
def append_eval_df(model_type, target_var, eval_df, val, yhat):
    '''
    this function takes in as arguments the type of model run, and the name of the target variable. 
    It returns the eval_df with the rmse appended to it for that model and target_var. 
    '''
    rmse = evaluate(target_var, val, yhat)
    d = {'model_type': [model_type], 'target_var': [target_var],
        'rmse': [rmse]}
    d = pd.DataFrame(d)
    return eval_df.append(d, ignore_index = True)


def run_models(train, val):
    '''running the modeling in a dataframe to connect all models '''
    eval_df = pd.DataFrame(columns=['model_type', 'target_var', 'rmse'])
    
    # Run last observed value model
    pred = train['Fahrenheit'][-1]
    yhat = pd.DataFrame({
    'Fahrenheit': [pred]},
    index = val.index)
    eval_df = append_eval_df(model_type='last_observed_value', target_var='Fahrenheit', eval_df=eval_df, val=val, yhat=yhat)
    plot_and_eval('Fahrenheit', train, val, yhat, 'Last Observed Value Model')
    

    # Run simple average model
    pred = round(train['Fahrenheit'].mean(), 2)
    yhat = pd.DataFrame({
    'Fahrenheit': [pred]},
    index = val.index)
    eval_df = append_eval_df(model_type='simple_average', target_var='Fahrenheit', eval_df=eval_df, val=val, yhat=yhat)
    plot_and_eval('Fahrenheit', train, val, yhat, 'Simple Average Model')

    
    # Run mimic of last year model
    y = train.tail(483)[['Fahrenheit']]
    yhat = y[['Fahrenheit']] + y[['Fahrenheit']].mean()
    yhat.index = val.index
    eval_df = append_eval_df(model_type = "previous_year", 
                            target_var = 'Fahrenheit',
                        eval_df=eval_df, val=val, yhat=yhat)
    plot_and_eval('Fahrenheit', train, val, yhat, 'Previous Year Model')

    
    return eval_df


def best_model(train, val, test):
    '''aacquiring the best model last year models'''
    y = train.tail(323)[['Fahrenheit']]
    yhat = y[['Fahrenheit']] + y[['Fahrenheit']].mean()
    yhat.index = test.index
    plt.figure(figsize = (18,8))
    plt.plot(train['Fahrenheit'], label='Train', linewidth=1)
    plt.plot(val['Fahrenheit'], label='Validate', linewidth=1)
    plt.plot(test['Fahrenheit'], label='Test', linewidth=1, color='red')
    plt.plot(yhat['Fahrenheit'], label='Prediction', linewidth=1, color='green')
    plt.title('Previous Year Model')
    plt.legend()
    plt.show()
    
    rmse = evaluate('Fahrenheit', val.tail(323), yhat)
    d = {'model_type': ['test_resutls'], 'target_var': ['Fahrenheit'],
        'rmse': [rmse]}
    
    return pd.DataFrame(d)





