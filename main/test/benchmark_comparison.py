# Benchmark comparison

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Nimbus Roman No9 L"],
    "font.size": 9,
    'axes.spines.top': False,
    'axes.spines.right': False
})
labels_hours = ['0','','','','','','6','','','','','','12','','','','','','18','','','','','23']

cbcolors = sns.color_palette('colorblind')



df = pd.DataFrame()

for f in ['df_base.csv','df_full_bb_eff.csv', 'df_full_ecp.csv','df_full_ecp_eff.csv']:
    df = pd.concat([df, pd.read_csv(f)])


lst = list(df.columns)
cases_list = [k for k in lst if 'T_' in k]
cases_list.append('test')
df_a = df.copy(deep=True)
df_a = df_a[cases_list].set_index('test').stack()
df_a = df_a.reset_index(name='index')
df_a.columns = ['test','case','time']

lst = list(df.columns)
cases_list = [k for k in lst if 'O_' in k]
cases_list.append('test')
df_b = df.copy(deep=True)
df_b = df_b[cases_list].set_index('test').stack()
df_b = df_b.reset_index(name='index')
df_b.columns = ['test','case','objective']

df_a['objective'] = df_b['objective']

df = df_a.copy(deep=True)
del df_a, df_b, lst, f

#%% Plots

plt.figure()
sns.boxplot(data=df, x='case', y='time',hue='test', palette=cbcolors)
# plt.yscale('log')
plt.ylim(0)
plt.ylabel('Execution time (s)')
plt.show()

plt.figure()
sns.barplot(data=df, x='case', y='objective',hue='test', palette=cbcolors)
plt.axhline(0,color='k')
plt.ylabel('Objective value')
plt.show()