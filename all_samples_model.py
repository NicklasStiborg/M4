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

def drop(df):
    df = df.drop(columns=['Unnamed: 0', 'reportedCurrency', 'cik', 'fillingDate', 'acceptedDate',
       'calendarYear', 'period','link', 'finalLink','symbol', 'date'])
    return df




'''test and train'''
count_1 = 0
count_2 = 1
pre_act_dict = {}

label = 'All'
file_name = 'all'
divider_number = 1569

'''model'''
model = Sequential()
model.add(layers.LSTM(100, return_sequences=True, input_shape=(27,1)))
model.add(layers.Dropout(0.1))
model.add(layers.LSTM(50, activation='relu', return_sequences=True))
model.add(layers.Dropout(0.1))
model.add(layers.LSTM(50, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae', 'mape'])


count_2 = 1
for i in nas_df['Symbol'][:divider_number]:
    print(i)
    print(str(count_2) + '/' + str(len(nas_df['Symbol'][:divider_number])))
    df = pd.read_csv(fs_dir + r'\fs_' + str(i) +'.csv')
    df_n = drop(df)
    y_scaler = MinMaxScaler(feature_range=(-1,1))
    y = df_n['ebitda'].values.reshape(-1,1)
    y = y_scaler.fit_transform(y)
    x_scaler = MinMaxScaler(feature_range=(-1,1))
    df_nx = df_n.drop(columns=['ebitda'])
    x = df_nx.values
    x = x_scaler.fit_transform(x)
    model.fit(x,y,epochs=10,batch_size=2, verbose=0)
    count_2 += 1

count_1 += 1
pre_act_dict = {}
for k in nas_df['Symbol'][divider_number:]:
    df_2 = pd.read_csv(fs_dir + r'\fs_' + str(k) +'.csv')
    df_2 = drop(df_2)
    print(k)
    y_scaler_1 = MinMaxScaler(feature_range=(-1,1))
    df_2_y = df_2['ebitda']
    y_1 = df_2_y.values.reshape(-1,1)
    y_1 = y_scaler_1.fit_transform(y_1)
    x_scaler_1 = MinMaxScaler(feature_range=(-1,1))
    df_2_x = df_2.drop(columns=['ebitda'])
    x_1 = df_2.iloc[:,1:].values
    x_1 = x_scaler_1.fit_transform(x_1)
    y_true = y_1
    pre_1 = model.predict(x_1)
    y_predict = y_scaler.inverse_transform(pre_1)
    pre_act_dict[k] = {}
    pre_act_dict[k]['true'] = y_true.flatten()
    pre_act_dict[k]['predict'] = y_predict.flatten()

results_df = pd.DataFrame(pre_act_dict)
results_df_new = results_df.transpose()
results_df_new = results_df_new.explode(column=['true', 'predict'])
results_df_new.to_csv('results_{}.csv'.format(file_name))
print(results_df_new.head())
print(results_df_new.tail())