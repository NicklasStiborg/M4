import pandas as pd
import requests
import json
import os
import matplotlib.pyplot as plt
import numpy as np

'''directories'''
#directory portability
curr_dir = os.getcwd()

#adding resource folder
res_dir = curr_dir + r'\resources'

#nasdaq list dir
nas_list_dir = res_dir + r'\nasdaq_list_with_quarter_count.csv'

fs_dir = curr_dir + r'\financial_statements'
nas_list_dir_no_f = res_dir + r'\nasdaq_list.csv'

'''access keys'''
#access keys
fmp_api_key = 

'''data import'''
#import local 
#import list of companies
nas_df = pd.read_csv(nas_list_dir)
nas_df_filtered = nas_df[nas_df['Count of quarterly reports'] >= 64]

#import list of companies no filter
nas_df_no_f = pd.read_csv(nas_list_dir_no_f)

#create empty dataframe
biss = 'AAPL'
api_url = 'https://fmpcloud.io/api/v3/income-statement/{}?period=quarter&limit=600&apikey='+fmp_api_key
response = requests.get(api_url.format(biss))
content = response.content
json_content = json.loads(content)

df_biss = pd.DataFrame(json_content)
df_empty = df_biss[0:0]

list_of_companies = {'companies': [], 'quarters': []}

'''create and append list of '''
count = 0 

for i in nas_df_no_f['Symbol']:
    print(count)
    try:
        r = requests.get(api_url.format(i))
        c = r.content
        jc = json.loads(c)
        if len(jc) >= 64:
            print(i)
            df_temp = pd.DataFrame(jc[:63])
            df_temp = df_temp.fillna(0)
            df_temp.iloc[:,8:-2] = np.log(df_temp.iloc[:,8:-2])
            df_temp = df_temp.sort_values(['symbol', 'date'])
            df_temp.iloc[:,8:-2] = df_temp.iloc[:,8:-2].diff()
            df_temp = df_temp.replace([np.inf, -np.inf], 0)
            df_temp = df_temp.fillna(0)
            df_temp_new = df_temp.iloc[1:]
            df_temp_new.to_csv(fs_dir + r'\fs' + '_' + str(i) + '.csv')
            list_of_companies['companies'].append(i)
            list_of_companies['quarters'].append(len(df_temp))
        count += 1 
    except:
        print(i + "N/A")
        count += 1 
        continue

df_list = pd.DataFrame(list_of_companies)
df_list.to_csv('full_list.csv')
