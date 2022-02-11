import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

'''directories'''
#directory portability
curr_dir = os.getcwd()

#adding resource folder
res_dir = curr_dir + r'\resources'

#apple list dir
apple_list_dir = res_dir + r'\apple_64_quarters.csv'

#full list dir
full_list_dir = res_dir + r'\full_list_final.csv'

#nasdaq list dir
nas_list_dir = res_dir + r'\nasdaq_list.csv'

fs_dir = curr_dir + r'\financial_statements'

'''imports'''
#import apple sample
apple_df = pd.read_csv(apple_list_dir)
apple_df = apple_df.sort_values('date', ascending=True)
apple_df['date'] = [i[:-6] for i in apple_df['date']]

#import full list
full_df = pd.read_csv(full_list_dir)

#filtered list import
nas_df = pd.read_csv(nas_list_dir)
# nas_df_filtered = nas_df[nas_df['Quarters'] >= 64]
df_full_list = pd.read_csv(full_list_dir)
print(df_full_list['Market Cap'].head())

def drop(df):
    df = df.drop(columns=['Unnamed: 0', 'reportedCurrency', 'cik', 'fillingDate', 'acceptedDate',
       'calendarYear', 'period','link', 'finalLink', 'date'])
    return df

'''plot setting'''
plt.rcParams["font.family"] = "Times New Roman"


# '''data cleaning'''
# apple_df_log = apple_df.copy()
# apple_df_log.iloc[:,9:-2] = np.log(apple_df.iloc[:,9:-2])

# apple_df_diff = apple_df_log.copy()
# apple_df_diff.iloc[:,9:-2] = apple_df_diff.iloc[:,9:-2].diff()

# full_df_log = full_df.copy()
# full_df_log.iloc[:,9:-2] = np.log(full_df.iloc[:,9:-2])

# '''plots'''

# #apples EBITDA without log and diff, with log and with log and diff
# plots = {"plot_1":
# {"title": "Apple's EBITDA without natural logarithm or differencing",
# "x": apple_df['date'],
# "y": apple_df['ebitda'],
# "figname": 'apple_ebitda_without_log_and_diff.png'},
# "plot_2":
# {"title": "Apple's EBITDA with natural logarithm",
# "x": apple_df_log['date'],
# "y": apple_df_log['ebitda'],
# "figname": 'apple_ebitda_with_log.png'},
# "plot_3":
# {"title": "Apple's EBITDA with natural logarithm and differencing",
# "x": apple_df_diff['date'],
# "y": apple_df_diff['ebitda'],
# "figname": 'apple_ebitda_with_log_and_diff.png'}
# }

# def make_line_plot(title, x, y, figname):
#     plt.figure()
#     plt.title(title, pad=20, fontweight="bold", size=16)
#     plt.plot(x, y, color='black')
#     plt.ylabel('EBITDA')
#     plt.xlabel('Quarters aggregated into years')
#     plt.subplots_adjust(bottom=0.2)
#     plt.xticks(rotation=90)
#     plt.savefig(figname)

# # make_line_plot(**plots['plot_1'])
# # make_line_plot(**plots['plot_2'])
# # make_line_plot(**plots['plot_3'])

#companies across sectors
# plt.figure()
# plt.title('Companies grouped by sectors', pad=20, fontweight="bold", size=16)
# ax_3 = sns.barplot(df_full_list['Sector'].value_counts(dropna=False).index, df_full_list['Sector'].value_counts(dropna=False))
# plt.xticks(rotation=90)
# plt.subplots_adjust(bottom=0.4)
# plt.ylabel('Number of companies')
# plt.xlabel('Sectors')
# ax_3.bar_label(ax_3.containers[0])
# plt.savefig('companies_across_sectors_1.png')

'''heatmap'''


# empty_df = pd.read_csv(fs_dir + r'\fs_' + 'A' +'.csv').iloc[0:0]

# for i in full_df['Symbol'][:30]:
#     print(i)
#     df_n = pd.read_csv(fs_dir + r'\fs_' + str(i) +'.csv')
#     empty_df = empty_df.append(df_n)

# print(empty_df.head())
# print(empty_df.info())

# #correlation heatmap of financial items
# df_dropped = drop(empty_df)
# # df_dropped = df_dropped.groupby('symbol').mean().reset_index()
# print(df_dropped.head())
# print(df_dropped.columns)

# plt.figure(figsize=(35,25))
# sns.set(font_scale=3)
# plt.rcParams["font.family"] = "Times New Roman"
# plt.title('Correlation heatmap of financial items', pad=20, fontweight="bold", size=64)
# sns.heatmap(df_dropped.corr(), annot=True, annot_kws={"size": 16})
# plt.savefig('correlation_heatmap_1.png')


# '''data overview full list'''
# df_features_only = full_df.drop(columns=['Unnamed: 0', 'cik', 'calendarYear', 'link', 'finalLink'])
# print(df_features_only.info())

# plt.show()

'''marketcap diagram'''

# list_of_mc_categories = []

# for i in df_full_list['Market Cap']:
#     if i < 50000000:
#         print('nano')
#         k = 'Nano (<50M)'
#     elif 50000000 <= i <= 300000000:
#         print('micro')
#         k = 'Micro (50M-300M)'
#     elif 300000000 <= i <= 2000000000:
#         print('small')
#         k = 'Small (300M-2B)'
#     elif 2000000000 <= i <= 10000000000:
#         print('medium')
#         k = 'Medium (2B-10B)'
#     elif 10000000000 <= i <= 200000000000:
#         print('large')
#         k = 'Large (10B-200B)'
#     elif i > 2000000000:
#         print('mega')
#         k = 'Mega (>200B)'
#     else:
#         print('other')
#         k = 'NaN'
    
#     list_of_mc_categories.append(k)

# df_full_list['Market Cap Category'] = list_of_mc_categories

# print(df_full_list[['Name', 'Market Cap Category']].head())
# print(df_full_list['Market Cap Category'].value_counts())

# plt.figure()
# plt.title('Companies grouped by market cap in USD', pad=20, fontweight="bold", size=16)
# ax_1 = sns.barplot(df_full_list['Market Cap Category'].value_counts(dropna=False).index, df_full_list['Market Cap Category'].value_counts(dropna=False))
# plt.xlabel('Market cap groups')
# plt.ylabel('Number of companies')
# plt.xticks(rotation=90)
# plt.subplots_adjust(bottom=0.3)
# ax_1.bar_label(ax_1.containers[0])
# plt.savefig('companies_by_market_cap.png')


# empty_df = pd.read_csv(fs_dir + r'/fs_' + 'A' +'.csv').iloc[0:0]


# for i in df_full_list['Symbol']:
#     print(i)
#     df = pd.read_csv(fs_dir + r'/fs_' + str(i) +'.csv')
#     empty_df = empty_df.append(df)

# print(empty_df)

# empty_df['year'] = [int(i[:-6]) for i in empty_df['date']]

# filtered = empty_df[empty_df['year'] < 2005]

# print(len(filtered['symbol'].value_counts()))
# print(filtered['symbol'].value_counts())

# plt.figure()
# plt.title('Quarters registered per year', pad=20, fontweight="bold", size=16)
# ax_2 = sns.barplot(empty_df['year'].value_counts(dropna=False).index, empty_df['year'].value_counts(dropna=False))
# plt.xlabel('Years')
# plt.ylabel('Number of quarters')
# plt.xticks(rotation=90)
# plt.subplots_adjust(bottom=0.3)
# ax_2.bar_label(ax_2.containers[0])
# plt.savefig('quarters_per_year.png')

# print(empty_df['year'].value_counts())


# print(df_full_list['Market Cap'].mean())
# print('25th percentile ' + str(np.percentile(df_full_list['Market Cap'], 25)))
# print('50th percentile ' + str(np.percentile(df_full_list['Market Cap'], 50)))
# print('75th percentile ' + str(np.percentile(df_full_list['Market Cap'], 75)))
# print('90th percentile ' + str(np.percentile(df_full_list['Market Cap'], 90)))
# print('95th percentile ' + str(np.percentile(df_full_list['Market Cap'], 98)))

