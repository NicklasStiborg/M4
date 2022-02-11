from operator import ge
from random import sample
from unittest import result
from xml.sax.handler import feature_external_ges
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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
print(len(nas_df))

def filter_by_sector(sector):
    df = nas_df[nas_df['Sector'] == sector]
    return df

def filter_by_sector_na():
    df = nas_df[nas_df['Sector'].isnull()]
    return df

def drop(df):
    df = df.drop(columns=['Unnamed: 0', 'reportedCurrency', 'cik', 'fillingDate', 'acceptedDate',
       'calendarYear', 'period','link', 'finalLink','symbol', 'date'])
    return df

list_of_sectors = ['Finance', 'Consumer Services', 'Health Care', 'Technology', 
'Capital Goods', 'Consumer Non-Durables', 'Public Utilities', 'Basic Industries',
'Energy', 'Miscellaneous', 'Transportation', 'Consumer Durables']

list_of_sectors_dict = {
    'Finance': {
        'file_name': 'finance',
        'divider_number': 302
    },
        'Health Care': {
        'file_name': 'health_care',
        'divider_number': 205
    },
        'Consumer Services': {
        'file_name': 'cs',
        'divider_number': 216
    },
        'Technology': {
        'file_name': 'tech',
        'divider_number': 182
    },
        'Capital Goods': {
        'file_name': 'cg',
        'divider_number': 173
    },
        'Consumer Non-Durables': {
        'file_name': 'cnd',
        'divider_number': 83
    },
        'Public Utilities': {
        'file_name': 'pu',
        'divider_number': 79
    },
        'Basic Industries': {
        'file_name': 'bi',
        'divider_number': 68
    },
        'Energy': {
        'file_name': 'energy',
        'divider_number': 68
    },
        'Miscellaneous': {
        'file_name': 'misc',
        'divider_number': 54
    },
        'Transportation': {
        'file_name': 'trans',
        'divider_number': 34
    },
        'Consumer Durables': {
        'file_name': 'cd',
        'divider_number': 32
    },
}

list_of_sample_models = {
    1: {
    'batch size': 2,
    'layers': 1,
    'neurons': 50,
    'drop_out': 1,
    },
    2: {
    'batch size': 2,
    'layers': 2,
    'neurons': 50,
    'drop_out': 1,
    },
    3: {
    'batch size': 2,
    'layers': 3,
    'neurons': 50,
    'drop_out': 1,
    },
    4: {
    'batch size': 2,
    'layers': 1,
    'neurons': 100,
    'drop_out': 1,
    },
    5: {
    'batch size': 2,
    'layers': 2,
    'neurons': 100,
    'drop_out': 1,
    },
    6: {
    'batch size': 2,
    'layers': 3,
    'neurons': 100,
    'drop_out': 1,
    },
    7: {
    'batch size': 2,
    'layers': 1,
    'neurons': 50,
    'drop_out': 0,
    },
    8: {
    'batch size': 2,
    'layers': 2,
    'neurons': 50,
    'drop_out': 0,
    },
    9: {
    'batch size': 2,
    'layers': 3,
    'neurons': 50,
    'drop_out': 0,
    },
    10: {
    'batch size': 2,
    'layers': 1,
    'neurons': 100,
    'drop_out': 0,
    },
    11: {
    'batch size': 2,
    'layers': 2,
    'neurons': 100,
    'drop_out': 0,
    },
    12: {
    'batch size': 2,
    'layers': 3,
    'neurons': 100,
    'drop_out': 0,
    },
    13: {
    'batch size': 10,
    'layers': 1,
    'neurons': 50,
    'drop_out': 1,
    },
    14: {
    'batch size': 10,
    'layers': 2,
    'neurons': 50,
    'drop_out': 1,
    },
    15: {
    'batch size': 10,
    'layers': 3,
    'neurons': 50,
    'drop_out': 1,
    },
    16: {
    'batch size': 10,
    'layers': 1,
    'neurons': 100,
    'drop_out': 1,
    },
    17: {
    'batch size': 10,
    'layers': 2,
    'neurons': 100,
    'drop_out': 1,
    },
    18: {
    'batch size': 10,
    'layers': 3,
    'neurons': 100,
    'drop_out': 1,
    },
    19: {
    'batch size': 10,
    'layers': 1,
    'neurons': 50,
    'drop_out': 0,
    },
    20: {
    'batch size': 10,
    'layers': 2,
    'neurons': 50,
    'drop_out': 0,
    },
    21: {
    'batch size': 10,
    'layers': 3,
    'neurons': 50,
    'drop_out': 0,
    },
    22: {
    'batch size': 10,
    'layers': 1,
    'neurons': 100,
    'drop_out': 0,
    },
    23: {
    'batch size': 10,
    'layers': 2,
    'neurons': 100,
    'drop_out': 0,
    },
    24: {
      'batch size': 10,
    'layers': 3,
    'neurons': 100,
    'drop_out': 0,
    }
}

#yellow
def model_with_1_layers_with_dropout(neurons):
    model = Sequential()
    model.add(layers.LSTM(neurons, return_sequences=False, input_shape=(27,1)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae', 'mape'])
    return model
#orange
def model_with_2_layers_with_dropout(neurons):
    model = Sequential()
    model.add(layers.LSTM(neurons, return_sequences=True, input_shape=(27,1)))
    model.add(layers.Dropout(0.1))
    model.add(layers.LSTM(neurons, return_sequences=False, input_shape=(27,1)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae', 'mape'])
    return model
#green
def model_with_3_layers_with_dropout(neurons):
    model = Sequential()
    model.add(layers.LSTM(neurons, return_sequences=True, input_shape=(27,1)))
    model.add(layers.Dropout(0.1))
    model.add(layers.LSTM(neurons, return_sequences=True, input_shape=(27,1)))
    model.add(layers.Dropout(0.1))
    model.add(layers.LSTM(neurons, return_sequences=False, input_shape=(27,1)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae', 'mape'])
    return model
#purple
def model_with_1_layers_without_dropout(neurons):
    model = Sequential()
    model.add(layers.LSTM(neurons, return_sequences=False, input_shape=(27,1)))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae', 'mape'])
    return model
#red
def model_with_2_layers_without_dropout(neurons):
    model = Sequential()
    model.add(layers.LSTM(neurons, return_sequences=True, input_shape=(27,1)))
    model.add(layers.LSTM(neurons, return_sequences=False, input_shape=(27,1)))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae', 'mape'])
    return model
#blue
def model_with_3_layers_without_dropout(neurons):
    model = Sequential()
    model.add(layers.LSTM(neurons, return_sequences=True, input_shape=(27,1)))
    model.add(layers.LSTM(neurons, return_sequences=True, input_shape=(27,1)))
    model.add(layers.LSTM(neurons, return_sequences=False, input_shape=(27,1)))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae', 'mape'])
    return model


count_1 = 0
count_2 = 1
pre_act_dict = {}



file_name = 's_9'
b_size = 2
neurons = 50
# batch_size = [2, 10]
# neurons = [50, 100]

'''model'''
model = model_with_3_layers_without_dropout(neurons)
print(type(model.summary()))



# '''test and train'''
# for i in nas_df['Symbol'][1300:1370]:
#     print(i)
#     print(str(count_2) + '/' + str(len(nas_df['Symbol'][1300:1370])))
#     df = pd.read_csv(fs_dir + r'\fs_' + str(i) +'.csv')
#     df_n = drop(df)
#     y_scaler = MinMaxScaler(feature_range=(-1,1))
#     y = df_n['ebitda'].values.reshape(-1,1)
#     y = y_scaler.fit_transform(y)
#     x_scaler = MinMaxScaler(feature_range=(-1,1))
#     df_nx = df_n.drop(columns=['ebitda'])
#     x = df_nx.values
#     x = x_scaler.fit_transform(x)
#     model.fit(x,y,epochs=10,batch_size=b_size, verbose=0)
#     count_2 += 1

# count_1 += 1
# pre_act_dict = {}

# for k in nas_df['Symbol'][1370:1400]:
#     df_2 = pd.read_csv(fs_dir + r'\fs_' + str(k) +'.csv')
#     df_2 = drop(df_2)
#     print(k)
#     y_scaler_1 = MinMaxScaler(feature_range=(-1,1))
#     df_2_y = df_2['ebitda']
#     y_1 = df_2_y.values.reshape(-1,1)
#     y_1 = y_scaler_1.fit_transform(y_1)
#     x_scaler_1 = MinMaxScaler(feature_range=(-1,1))
#     df_2_x = df_2.drop(columns=['ebitda'])
#     x_1 = df_2.iloc[:,1:].values
#     x_1 = x_scaler_1.fit_transform(x_1)
#     y_true = y_1
#     pre_1 = model.predict(x_1)
#     y_predict = y_scaler.inverse_transform(pre_1)
#     pre_act_dict[k] = {}
#     pre_act_dict[k]['true'] = y_true.flatten()
#     pre_act_dict[k]['predict'] = y_predict.flatten()

# results_df = pd.DataFrame(pre_act_dict)
# results_df_new = results_df.transpose()
# results_df_new = results_df_new.explode(column=['true', 'predict'])
# results_df_new.to_csv('results_{}.csv'.format(file_name))
# print(results_df_new.head())
# print(results_df_new.tail())
