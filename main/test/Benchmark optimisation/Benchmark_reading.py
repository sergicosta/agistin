# -*- coding: utf-8 -*-
"""
File for the interpretationa and analysis of the benchmark optimisation result

     - First part plots the results obtained in the benchmark file. Generating several boxplots comparing the logarithmic execution time per each case.
     - Second part calculates the difference between the base case and the cases with modifications such as inequality or linealized pump.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def boxplot(base_case, test_case, list_cases, cbcolors):
    df = pd.DataFrame()
    df_base = pd.DataFrame()
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Nimbus Roman No9 L"],
        "font.size": 9,
        'axes.spines.top': False,
        'axes.spines.right': False
    })


    for f in list_cases:
        df_base = pd.concat([df_base, pd.read_csv(f'./{base_case}/{f}-{base_case}.csv')])
        df = pd.concat([df, pd.read_csv(f'./{test_case}/{f}-{test_case}.csv')])

    lst = list(df.columns)
    cases_list_t = [k for k in lst if 'T_' in k]
    cases_list_t.append('test')

    df_a_time = df.copy(deep=True)
    df_a_time = df_a_time[cases_list_t].set_index('test').stack()
    df_a_time = df_a_time.reset_index(name='time')
    df_a_time.columns = ['test', 'case', 'time']

    lst = list(df_base.columns)
    cases_list_t = [k for k in lst if 'T_' in k]
    cases_list_t.append('test')

    df_b_time = df_base.copy(deep=True)
    df_b_time = df_b_time[cases_list_t].set_index('test').stack()
    df_b_time = df_b_time.reset_index(name='time')
    df_b_time.columns = ['test', 'case', 'time']

    df_time = pd.concat([df_a_time, df_b_time], ignore_index=True)

    lst = list(df.columns)
    cases_list_o = [k for k in lst if 'O_' in k]
    cases_list_o.append('test')

    df_a_goal = df.copy(deep=True)
    df_a_goal = df_a_goal[cases_list_o].set_index('test').stack()
    df_a_goal = df_a_goal.reset_index(name='goal')
    df_a_goal.columns = ['test', 'case', 'goal']

    lst = list(df_base.columns)
    cases_list_o = [k for k in lst if 'O_' in k]
    cases_list_o.append('test')

    df_b_goal = df_base.copy(deep=True)
    df_b_goal = df_b_goal[cases_list_o].set_index('test').stack()
    df_b_goal = df_b_goal.reset_index(name='goal')
    df_b_goal.columns = ['test', 'case', 'goal']

    df_goal = pd.concat([df_a_goal, df_b_goal], ignore_index=True)
    
    df_time['case'] = df_time['case'].str.replace('T_', '', regex=False)
    df_goal['case'] = df_goal['case'].str.replace('O_', '', regex=False)

    df = pd.merge(df_time, df_goal, on=['test', 'case'], how='left')
    df_infea = df[df['goal'] == 0].copy()
    df = df[df['goal'] != 0].copy()

    df[['Algorithm', 'Method']] = df['test'].str.split('-', expand=True)
    
    algorithm_order = list_cases
    method_order = [base_case, test_case]
    test_order = [f'{a}-{m}' for a in algorithm_order for m in method_order]

    plt.figure(figsize=(3.5,2.8))
    ax = sns.boxplot(
        x='test',
        y='time',
        hue='case',
        data=df,
        order=test_order,
        dodge=False,
        palette=cbcolors,
        width=0.3
    )


    plt.yscale('log')
    plt.title(f'Boxplots: {test_case} vs {base_case}')
    plt.tight_layout()

    for i in range(1, len(algorithm_order)):
        xpos = i * 2 - 0.5
        ax.axvline(x=xpos, color='gray', linestyle='--', linewidth=1)

    xtick_labels = []
    for alg in algorithm_order:
        xtick_labels.extend([alg, ''])
    ax.set_xticks(range(len(test_order)))
    ax.set_xticklabels(xtick_labels, rotation=0)
    ax.set_xlabel('') 
    ax.set_ylabel('')

    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05),  
        ncol=3,                       
        frameon=False
    )

# Generation of arrows between cases.

    # for alg in algorithm_order:
    #     for c in df['case'].unique():
    #         y1 = df[(df['Algorithm'] == alg) & (df['Method'] == base_case) & (df['case'] == c)]['time']
    #         y2 = df[(df['Algorithm'] == alg) & (df['Method'] == test_case) & (df['case'] == c)]['time']
    #         if not y1.empty and not y2.empty:
    #             y_start = np.median(y1)
    #             y_end = np.median(y2)
    #             x_start = test_order.index(f'{alg}-{base_case}')
    #             x_end = test_order.index(f'{alg}-{test_case}')
    #             ax.annotate(
    #                 '', xy=(x_end, y_end), xytext=(x_start, y_start),
    #                 arrowprops=dict(arrowstyle='->', color='black', lw=0.6),
    #                 annotation_clip=False
    #             )
    
    plt.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.22)
    plt.savefig('./'+f'boxplot_{test_case}+W.svg', format='svg', dpi=300)
    plt.show()

    return df, df_infea

cbcolors = sns.color_palette("colorblind")
list_cases = ['BB','QG', 'ECP']
for method in ['Linealized_3i', 'Linealized_4i', 'Normal']:
    df, df_infea = boxplot('Equality', method, list_cases, cbcolors)


#%% Difference from Base

list_cases = ['BB','OA','QG','ECP']

base_case = 'Equality'

df_base = pd.DataFrame()
df_ineq = pd.DataFrame()

test_case = 'Normal'
for f in list_cases:
        df_base = pd.concat([df_base, pd.read_csv(f'./{base_case}/{f}-{base_case}.csv')])
        df_ineq = pd.concat([df_ineq, pd.read_csv(f'./{test_case}/{f}-{test_case}.csv')])
        
mean_row = df_base.mean(numeric_only=True)
mean_row['test'] = 'mean'

df_base = pd.concat([df_base, pd.DataFrame([mean_row])], ignore_index=True)

mean_row = df_ineq.mean(numeric_only=True)
mean_row['test'] = 'mean'

df_ineq = pd.concat([df_ineq, pd.DataFrame([mean_row])], ignore_index=True)

df_lin3 = pd.DataFrame()        
test_case = 'Linealized_3i'
for f in list_cases:
    df_lin3 = pd.concat([df_lin3,pd.read_csv(f'./{test_case}/{f}-{test_case}.csv')])

mean_row = df_lin3.mean(numeric_only=True)
mean_row['test'] = 'mean'

df_lin3 = pd.concat([df_lin3, pd.DataFrame([mean_row])], ignore_index=True)

df_lin4 = pd.DataFrame()        
test_case = 'Linealized_4i'
for f in list_cases:
    df_lin4 = pd.concat([df_lin4,pd.read_csv(f'./{test_case}/{f}-{test_case}.csv')])

df_base['algorithm'] = df_base['test'].str.extract(r'^(BB|OA|QG|ECP)')
means_base = df_base.groupby('algorithm').mean(numeric_only=True).reset_index()
means_base['test'] = means_base['algorithm'] + '-mean'
cols = ['test'] + [col for col in df_base.columns if col not in ['test', 'algorithm']]
means_base = means_base[cols]
df_base = pd.concat([df_base, means_base], ignore_index=True)

df_ineq['algorithm'] = df_ineq['test'].str.extract(r'^(BB|OA|QG|ECP)')
means_ineq = df_ineq.groupby('algorithm').mean(numeric_only=True).reset_index()
means_ineq['test'] = means_ineq['algorithm'] + '-mean'
cols = ['test'] + [col for col in df_ineq.columns if col not in ['test', 'algorithm']]
means_ineq = means_ineq[cols]
df_ineq = pd.concat([df_ineq, means_ineq], ignore_index=True)

df_lin3['algorithm'] = df_lin3['test'].str.extract(r'^(BB|OA|QG|ECP)')
means_lin3 = df_lin3.groupby('algorithm').mean(numeric_only=True).reset_index()
means_lin3['test'] = means_lin3['algorithm'] + '-mean'
cols = ['test'] + [col for col in df_lin3.columns if col not in ['test', 'algorithm']]
means_lin3 = means_lin3[cols]
df_lin3 = pd.concat([df_lin3, means_lin3], ignore_index=True)

df_lin4['algorithm'] = df_lin4['test'].str.extract(r'^(BB|OA|QG|ECP)')
means_lin4 = df_lin4.groupby('algorithm').mean(numeric_only=True).reset_index()
means_lin4['test'] = means_lin4['algorithm'] + '-mean'
cols = ['test'] + [col for col in df_lin4.columns if col not in ['test', 'algorithm']]
means_lin4 = means_lin4[cols]
df_lin4 = pd.concat([df_lin4, means_lin4], ignore_index=True)


def calcular_diferencia_relativa(df_base, df_comparacion, nombre_metodo):
    df_base_means = df_base[df_base['test'].str.contains('-mean')].set_index('test')
    df_comp_means = df_comparacion[df_comparacion['test'].str.contains('-mean')].set_index('test')

    algoritmos = df_base_means.index.str.extract(r'^(BB|OA|QG|ECP)')[0].values
    resultados = []

    for alg in algoritmos:
        base_row = df_base_means.loc[f'{alg}-mean']
        comp_row = df_comp_means.loc[f'{alg}-mean']
        diff = ((comp_row - base_row) / base_row) * 100
        diff['test'] = f'{alg}-{nombre_metodo}'
        resultados.append(diff)

    return pd.DataFrame(resultados).reset_index(drop=True)

diff_ineq = calcular_diferencia_relativa(df_base, df_ineq, 'ineq')
diff_lin3 = calcular_diferencia_relativa(df_base, df_lin3, 'lin3')
diff_lin4 = calcular_diferencia_relativa(df_base, df_lin4, 'lin4')

df_diff = pd.concat([diff_ineq, diff_lin3, diff_lin4], ignore_index=True)


