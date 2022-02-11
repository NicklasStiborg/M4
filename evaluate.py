import os
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

'''directories'''
#directory portability
curr_dir = os.getcwd()
data_dir = curr_dir + r'\data'
res_dir = data_dir + r'\resources'
sample_dir = curr_dir + r'\samples'
sample_list_dir = curr_dir + r'\sample_model_list.csv'
model_1_dir = curr_dir + r'\model_1'
model_2_dir = curr_dir + r'\model_2'
model_3_dir = curr_dir + r'\model_3'
model_tuned_dir = curr_dir
rf_dir = curr_dir + r'\results_rf.csv'
rf_dir_25 = curr_dir + r'\rf_model_25'
rf_dir_100 = curr_dir + r'\rf_model_100'
rf_dir_250 = curr_dir + r'\rf_model_250'

'''plot setting'''
plt.rcParams["font.family"] = "Times New Roman"

# df_sample_list = pd.read_csv(sample_list_dir, sep=';')
# df_sample_list['model'] = df_sample_list['Unnamed: 0']
# df_sample_list = df_sample_list.drop(columns=['Unnamed: 0'])
# # print(df_sample_list)

dictionary = {'model': [], 'average r2': [], 'average rmse': [], 'rmse 25': [], 'rmse 50': [], 'rmse 75': [],}

def r2_rmse(g):
    r2 = r2_score( g['true'], g['predict'] )
    rmse = np.sqrt( mean_squared_error( g['true'], g['predict'] ) )
    return pd.Series( dict(r2 = r2, rmse = rmse))

for i in os.listdir(rf_dir_25):
    df = pd.read_csv(rf_dir_25 + '/' + i)
    df = df.groupby('Unnamed: 0').apply(r2_rmse).reset_index()
    model = str(i[11:-4])
    mean_r2 = df['r2'].mean()
    mean_rmse = df['rmse'].mean()
    # rmse_high = df.sort_values(['rmse'], ascending=False)
    # rmse_high = rmse_high['rmse'].iloc[0]
    rmse_high = df['rmse'].quantile(q=0.75)
    # rmse_low = df.sort_values(['rmse'], ascending=True)
    # rmse_low = rmse_low['rmse'].iloc[0]
    rmse_low = df['rmse'].quantile(q=0.25)
    rmse_mid = df['rmse'].quantile(q=0.5)
    dictionary['model'].append(model)
    dictionary['average r2'].append(mean_r2)
    dictionary['average rmse'].append(mean_rmse)
    dictionary['rmse 75'].append(rmse_high)
    dictionary['rmse 50'].append(rmse_mid)
    dictionary['rmse 25'].append(rmse_low)
    # print(model)
    # print(mean_r2)
    # print(mean_rmse)
    # print(rmse_high)
    # print(rmse_low)

df_dict = pd.DataFrame(dictionary)
df_dict = df_dict.sort_values(['model'], ascending=True)
# df_merged = df_dict.merge(df_sample_list, on='model')
df_dict.to_csv('rf_25_results.csv')
# print(df_dict)


df = pd.read_csv(r"C:\Users\stibo\Desktop\M4_project\rf_model_250\results_rf_250_all.csv")
print(df.head())

df = df.groupby('Unnamed: 0')

for i, g in df:
    plt.figure()
    plt.title('Actual and predicted EBITDA of ' + str(i), pad=20, fontweight="bold", size=16)
    plt.plot(g['true'], label='Actual EBITDA')
    plt.plot(g['predict'], label='Predicted EBITDA')
    plt.ylabel('EBITDA')
    plt.xlabel('Quarters')
    plt.legend(loc=2)
    plt.xticks([], [])
    plt.savefig(curr_dir + r'\rfr_250_figures' + i)

# df_dict = pd.DataFrame(dictionary)
# df_dict = df_dict.sort_values(['model'], ascending=True)
# # df_merged = df_dict.merge(df_sample_list, on='model')
# df_dict.to_csv('model_1_tuned_results.csv')
# # print(df_dict)

# df = pd.read_csv(rf_dir)
# df = df.groupby('Unnamed: 0').apply(r2_rmse).reset_index()
# model = str('RF')
# print(model)
# mean_r2 = df['r2'].mean()
# mean_rmse = df['rmse'].mean()
# # rmse_high = df.sort_values(['rmse'], ascending=False)
# # rmse_high = rmse_high['rmse'].iloc[0]
# rmse_high = df['rmse'].quantile(q=0.75)
# # rmse_low = df.sort_values(['rmse'], ascending=True)
# # rmse_low = rmse_low['rmse'].iloc[0]
# rmse_low = df['rmse'].quantile(q=0.25)
# rmse_mid = df['rmse'].quantile(q=0.5)
# dictionary['model'].append(model)
# dictionary['average r2'].append(mean_r2)
# dictionary['average rmse'].append(mean_rmse)
# dictionary['rmse 75'].append(rmse_high)
# dictionary['rmse 50'].append(rmse_mid)
# dictionary['rmse 25'].append(rmse_low)
# print(model)
# print(mean_r2)
# print(mean_rmse)
# print(rmse_high)
# print(rmse_low)