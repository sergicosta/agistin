# Benchmark comparison

import numpy as np
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

list_cases = ['df_base.csv',
              'df_full_bb.csv','df_full_bb_eff.csv', 'df_full_bb_eqpump.csv', 
              'df_full_ecp.csv','df_full_ecp_eff.csv', 'df_full_ecp_eqpump.csv']

base_case_name = 'Base'

for f in list_cases:
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

df['objective_rel'] = np.nan
for c in df['case'].unique():
    df_base = df.query('test==@base_case_name')
    base_val = df_base.query('case==@c')['objective'].mean()
    df['objective_rel'] = df.apply(lambda x: (x.loc['objective']-base_val)/abs(base_val)*100 if x.loc['case']==c else x.loc['objective_rel'], axis=1) 

df['time_rel'] = np.nan
for c in df['case'].unique():
    df_base = df.query('test==@base_case_name')
    base_val = df_base.query('case==@c')['time'].mean()
    df['time_rel'] = df.apply(lambda x: (x.loc['time']-base_val)/abs(base_val)*100 if x.loc['case']==c else x.loc['time_rel'], axis=1) 

del df_a, df_b, lst, f

#%% Plots

fig = plt.figure()

# execution time
plt.subplot(221)
ax1 = sns.boxplot(data=df.query('test!=@base_case_name'), x='case', y='time', hue='test', palette=cbcolors, linewidth=0.5, legend=False)
n=0
for e in df.query('test==@base_case_name')['time']:
    plt.axvline(n+0.5, color='k', linewidth=0.5)
    plt.hlines(y=e,xmin=n-0.5, xmax=n+0.5, color='tab:red',linestyle='--')
    n=n+1
plt.yscale('log')
plt.ylabel('Execution time (s)')
plt.xlabel(None)
plt.xlim(-0.5)
plt.show()

# objective function
plt.subplot(222, sharex=ax1)
sns.barplot(data=df.query('test!=@base_case_name'), x='case', y='objective',hue='test', palette=cbcolors, legend=False)
n=0
N = len(df.query('test==@base_case_name')['objective'])
for e in df.query('test==@base_case_name')['objective']:
    plt.axvline(n+0.5, color='k', linewidth=0.5)
    plt.hlines(y=e,xmin=n-0.5, xmax=n+0.5, color='tab:red',linestyle='--')
    n=n+1
plt.axhline(0,color='k')
plt.ylabel('Objective value')
plt.xlabel(None)
plt.xlim(-0.5)
plt.show()

# execution time relative to Base
plt.subplot(223, sharex=ax1)
sns.barplot(data=df.query('test!=@base_case_name'), x='case', y='time_rel',hue='test', palette=cbcolors, legend=False)
n=0
N = len(df.query('test==@base_case_name')['time'])
for e in df.query('test==@base_case_name')['time']:
    plt.axvline(n+0.5, color='k', linewidth=0.5)
    n=n+1
plt.axhline(0,color='k')
plt.ylabel(f'$\Delta$ Exec time (\%)')
plt.xlim(-0.5)
plt.show()
plt.tight_layout()

# objective function relative to Base
plt.subplot(224, sharex=ax1)
sns.barplot(data=df.query('test!=@base_case_name'), x='case', y='objective_rel',hue='test', palette=cbcolors)
n=0
N = len(df.query('test==@base_case_name')['objective'])
for e in df.query('test==@base_case_name')['objective']:
    plt.axvline(n+0.5, color='k', linewidth=0.5)
    n=n+1
plt.axhline(0,color='k')
plt.ylabel(f'$\Delta$ Objective value (\%)')
plt.xlim(-0.5)
plt.legend(loc='best')
plt.show()
plt.tight_layout()
