from fileinput import filename
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


'''directories'''
#directory portability
curr_dir = os.getcwd()
data_dir = curr_dir + r'\data'
res_dir = data_dir + r'\resources'
fs_dir = data_dir + r'\financial_statements'
nas_list_dir = res_dir + r'\full_list_final.csv'

'''data import'''
#import local 
#import list of companies
nas_df = pd.read_csv(nas_list_dir)
df_na = nas_df[nas_df['Sector'].isnull()]

def filter_by_sector(sector):
    df = nas_df[nas_df['Sector'] == sector]
    return df

def drop(df):
    df = df.drop(columns=['Unnamed: 0', 'reportedCurrency', 'cik', 'fillingDate', 'acceptedDate',
       'calendarYear', 'period','link', 'finalLink','symbol', 'date'])
    return df

model_ols = LinearRegression()
# model_el = ElasticNet(tol=0.0001)
model_rf = RandomForestRegressor(n_estimators=25)

divider_number = 1534
count_2 = 1
count_1 = 1
pre_act_dict = {}

list_of_sectors = ['Finance', 'Consumer Services', 'Health Care', 'Technology', 
'Capital Goods', 'Consumer Non-Durables', 'Public Utilities', 'Basic Industries',
'Energy', 'Miscellaneous', 'Transportation', 'Consumer Durables']

list_of_sectors_dict = {
    'Finance': {
        'file_name': 'finance',
        'divider_number': 292
    },
        'Health Care': {
        'file_name': 'health_care',
        'divider_number': 201
    },
        'Consumer Services': {
        'file_name': 'cs',
        'divider_number': 213
    },
        'Technology': {
        'file_name': 'tech',
        'divider_number': 180
    },
        'Capital Goods': {
        'file_name': 'cg',
        'divider_number': 171
    },
        'Consumer Non-Durables': {
        'file_name': 'cnd',
        'divider_number': 83
    },
        'Public Utilities': {
        'file_name': 'pu',
        'divider_number': 76
    },
        'Basic Industries': {
        'file_name': 'bi',
        'divider_number': 67
    },
        'Energy': {
        'file_name': 'energy',
        'divider_number': 66
    },
        'Miscellaneous': {
        'file_name': 'misc',
        'divider_number': 51
    },
        'Transportation': {
        'file_name': 'trans',
        'divider_number': 32
    },
        'Consumer Durables': {
        'file_name': 'cd',
        'divider_number': 32
    },
}


# for u in list_of_sectors:
#     '''model'''
#     model = RandomForestRegressor(n_estimators=100)
#     print('**************')
#     print(u)
#     print('**************')
#     label = list_of_sectors_dict[u]
#     file_name = list_of_sectors_dict[u]['file_name']
#     divider_number = list_of_sectors_dict[u]['divider_number']
#     count_2 = 1
#     for i in filter_by_sector(u)['Symbol'][:divider_number]:
#         print(i)
#         print(str(count_2) + '/' + str(len(filter_by_sector(u)['Symbol'][:divider_number])))
#         df = pd.read_csv(fs_dir + r'\fs_' + str(i) +'.csv')
#         df_n = drop(df)
#         y_scaler = MinMaxScaler(feature_range=(-1,1))
#         y = df_n['ebitda'].values.reshape(-1,1)
#         y = y_scaler.fit_transform(y).ravel()
#         x_scaler = MinMaxScaler(feature_range=(-1,1))
#         df_nx = df_n.drop(columns=['ebitda'])
#         x = df_nx.values
#         x = x_scaler.fit_transform(x)
#         model.fit(x, y)
#         count_2 += 1
   
#     count_1 += 1
#     pre_act_dict = {}
#     for k in filter_by_sector(u)['Symbol'][divider_number:]:
#         df_2 = pd.read_csv(fs_dir + r'\fs_' + str(k) +'.csv')
#         df_2 = drop(df_2)
#         y_scaler_1 = MinMaxScaler(feature_range=(-1,1))
#         df_2_y = df_2['ebitda']
#         y_1 = df_2_y.values.reshape(-1,1)
#         x_scaler_1 = MinMaxScaler(feature_range=(-1,1))
#         df_2_x = df_2.drop(columns=['ebitda'])
#         x_1 = df_2.iloc[:,1:].values
#         x_1 = x_scaler_1.fit_transform(x_1)
#         y_predict = model.predict(x_1)
#         pre_act_dict[k] = {}
#         pre_act_dict[k]['true'] = y_1.flatten()
#         pre_act_dict[k]['predict'] = y_predict.flatten()

#     results_df = pd.DataFrame(pre_act_dict)
#     results_df_new = results_df.transpose()
#     results_df_new = results_df_new.explode(column=['true', 'predict'])
#     results_df_new.to_csv('results_rf_100_{}.csv'.format(file_name))
#     print(results_df_new.head())
#     print(results_df_new.tail())





# for u in range(1):
#     '''model'''
#     model = RandomForestRegressor(n_estimators=100)
#     print('**************')
#     print(u)
#     print('**************')
#     label = "NaN"
#     file_name = "nan"
#     divider_number = 71
#     count_2 = 1
#     for i in df_na['Symbol'][:divider_number]:
#         print(i)
#         print(str(count_2) + '/' + str(len(df_na['Symbol'][:divider_number])))
#         df = pd.read_csv(fs_dir + r'\fs_' + str(i) +'.csv')
#         df_n = drop(df)
#         y_scaler = MinMaxScaler(feature_range=(-1,1))
#         y = df_n['ebitda'].values.reshape(-1,1)
#         y = y_scaler.fit_transform(y).ravel()
#         x_scaler = MinMaxScaler(feature_range=(-1,1))
#         df_nx = df_n.drop(columns=['ebitda'])
#         x = df_nx.values
#         x = x_scaler.fit_transform(x)
#         model.fit(x, y)
#         count_2 += 1
   
#     count_1 += 1
#     pre_act_dict = {}
#     for k in df_na['Symbol'][divider_number:]:
#         df_2 = pd.read_csv(fs_dir + r'\fs_' + str(k) +'.csv')
#         df_2 = drop(df_2)
#         y_scaler_1 = MinMaxScaler(feature_range=(-1,1))
#         df_2_y = df_2['ebitda']
#         y_1 = df_2_y.values.reshape(-1,1)
#         x_scaler_1 = MinMaxScaler(feature_range=(-1,1))
#         df_2_x = df_2.drop(columns=['ebitda'])
#         x_1 = df_2.iloc[:,1:].values
#         x_1 = x_scaler_1.fit_transform(x_1)
#         y_predict = model.predict(x_1)
#         pre_act_dict[k] = {}
#         pre_act_dict[k]['true'] = y_1.flatten()
#         pre_act_dict[k]['predict'] = y_predict.flatten()

#     results_df = pd.DataFrame(pre_act_dict)
#     results_df_new = results_df.transpose()
#     results_df_new = results_df_new.explode(column=['true', 'predict'])
#     results_df_new.to_csv('results_rf_100_{}.csv'.format(file_name))
#     print(results_df_new.head())
#     print(results_df_new.tail())





for u in range(1):
    '''model'''
    model = RandomForestRegressor(n_estimators=500, n_jobs=-1)
    print('**************')
    print(u)
    print('**************')
    label = "All"
    file_name = "all"
    divider_number = 1534
    count_2 = 1
    for i in nas_df['Symbol'][:divider_number]:
        print(i)
        print(str(count_2) + '/' + str(len(nas_df['Symbol'][:divider_number])))
        df = pd.read_csv(fs_dir + r'\fs_' + str(i) +'.csv')
        df_n = drop(df)
        y_scaler = MinMaxScaler(feature_range=(-1,1))
        y = df_n['ebitda'].values.reshape(-1,1)
        y = y_scaler.fit_transform(y).ravel()
        x_scaler = MinMaxScaler(feature_range=(-1,1))
        df_nx = df_n.drop(columns=['ebitda'])
        x = df_nx.values
        x = x_scaler.fit_transform(x)
        model.fit(x, y)
        count_2 += 1
   
    count_1 += 1
    pre_act_dict = {}
    for k in nas_df['Symbol'][divider_number:]:
        df_2 = pd.read_csv(fs_dir + r'\fs_' + str(k) +'.csv')
        df_2 = drop(df_2)
        y_scaler_1 = MinMaxScaler(feature_range=(-1,1))
        df_2_y = df_2['ebitda']
        y_1 = df_2_y.values.reshape(-1,1)
        x_scaler_1 = MinMaxScaler(feature_range=(-1,1))
        df_2_x = df_2.drop(columns=['ebitda'])
        x_1 = df_2.iloc[:,1:].values
        x_1 = x_scaler_1.fit_transform(x_1)
        y_predict = model.predict(x_1)
        pre_act_dict[k] = {}
        pre_act_dict[k]['true'] = y_1.flatten()
        pre_act_dict[k]['predict'] = y_predict.flatten()

    results_df = pd.DataFrame(pre_act_dict)
    results_df_new = results_df.transpose()
    results_df_new = results_df_new.explode(column=['true', 'predict'])
    results_df_new.to_csv('results_rf_500_{}.csv'.format(file_name))
    print(results_df_new.head())
    print(results_df_new.tail())