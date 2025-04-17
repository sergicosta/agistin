import pandas as pd
import numpy as np
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

# data: pd.DataFrame() with the form {'x1':{x1_sensitivity_vals...}, 'x2':{x2vals...}, ... 'y':{yvals}}
# y: y column name as str
# base_idx: index of the row that contains the base case as int (default: 0)
def spiderplot(data, y, base_idx=0, ax=plt, linestyle=None, markers=None):
    data_pu = data/data.iloc[base_idx].abs()
    data_pu = data_pu.reset_index(drop=True)
    columns = data.drop(y,axis=1).columns
    
    ls = None
    mk = None
    
    for c in columns:
        index_plot = ((data_pu).drop([c,y],axis=1)==1).all(axis=1).where(lambda x: x).dropna().index
        data_plot = data_pu.iloc[index_plot].sort_values(by=c)
        data_plot[y] = data_plot[y]*abs(data.iloc[base_idx][y])
        
        if linestyle:
            ls = linestyle.pop()
        if markers:
            mk = markers.pop()
        
        ax.plot(data_plot[c],data_plot[y],label=c,linestyle=ls,marker=mk)






df = pd.read_excel('UpperCases.xlsx')
cbase = 166.85

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=cbcolors)

fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(figsize=(3.4, 2.2),nrows=2,ncols=2,sharex=True,sharey=True)
ax1.axhline(cbase,color='k',linestyle='--',linewidth=0.5)
ax2.axhline(cbase,color='k',linestyle='--',linewidth=0.5)
ax3.axhline(cbase,color='k',linestyle='--',linewidth=0.5)
ax4.axhline(cbase,color='k',linestyle='--',linewidth=0.5)
spiderplot(df.query('case==\'Base\'').reset_index(drop=True)[['pateff','psell','irr','COST']],y='COST', ax=ax1, linestyle=['-','--',':'], markers=['.','.','.'])
spiderplot(df.query('case==\'grid\'').reset_index(drop=True)[['pateff','psell','irr','COST']],y='COST',ax=ax2, linestyle=['-','--',':'], markers=['.','.','.'])
spiderplot(df.query('case==\'PAT\'').reset_index(drop=True)[['pateff','psell','irr','COST']],y='COST',ax=ax3, linestyle=['-','--',':'], markers=['.','.','.'])
spiderplot(df.query('case==\'gridPAT\'').reset_index(drop=True)[['pateff','psell','irr','COST']],y='COST',ax=ax4, linestyle=['-','--',':'], markers=['.','.','.'])

plt.ylim(-300,300)
plt.yticks([-200,0,200])

plt.legend(['base','$\eta_{pat}$','$c_{sell}$','$Q_{irr}$'],loc='upper center', bbox_to_anchor=(-0.2, -0.25), ncols=4)
plt.subplots_adjust(left=0.10, right=0.95, top=0.97, bottom=0.22)

plt.rcParams['savefig.format']='pdf'
plt.savefig('Spiderplots_upper', dpi=300)
plt.rcParams['savefig.format']='svg'
plt.savefig('Spiderplots_upper', dpi=300)








