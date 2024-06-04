"""
    AGISTIN - EXAMPLE 2
    
    Optimization usage example of two reservoirs, considering irrigation consumption,
    and two pumps. The pumping station gathers the pumps' power consumption, a solar
    PV plant and a connection point to the public grid.
    The PV plant can be upgraded to double its size (from 50e3 to 100e3) with a cost of 10 per power unit.
    The price of selling power to the grid is half of the cost of buying it.
    
    Authors: Sergi Costa Dilmé (CITCEA-UPC), Juan Carlos Olives-Camps (CITCEA-UPC), Paula Muñoz Peña (CITCEA-UPC)
"""

"""
RUGOSITAT:
    - PE-100: Ka = 0.050 mm
    - PVC-O: Ka = 0.050 mm
"""

from pyomo.util.infeasible import log_infeasible_constraints
import logging

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import pyomo
import pyomo.environ as pyo
from pyomo.network import Arc, Port

# Import devices
from Devices.Reservoirs import Reservoir
from Devices.Sources import Source
from Devices.Pipes import Pipe
from Devices.Pumps import Pump, RealPump, ReversiblePump, ReversibleRealPump
from Devices.Turbines import Turbine
from Devices.EB import EB
from Devices.SolarPV import SolarPV
from Devices.MainGrid import Grid
from Devices.Batteries import Battery, NewBattery
from pyomo.contrib.gdpopt.enumerate import GDP_Enumeration_Solver

# Import useful functions
from Utilities import clear_clc

#clean console and variable pane
# clear_clc() #consider removing if you are not working with Spyder

# model
m = pyo.ConcreteModel()

df_meteo_aug = pd.read_csv('data/meteo/LesPlanes_meteo_hour_aug.csv').head(24)
df_cons_aug = pd.read_csv('data/irrigation/LesPlanes_irrigation_aug.csv').head(24)
df_grid_aug = pd.read_csv('data/costs/PVPC_aug.csv').head(24)

df_meteo_jan = pd.read_csv('data/meteo/LesPlanes_meteo_hour_jan.csv').head(24)
df_cons_jan = pd.read_csv('data/irrigation/LesPlanes_irrigation_jan.csv').head(24)
df_grid_jan = pd.read_csv('data/costs/PVPC_jan.csv').head(24)

df_meteo = pd.concat([df_meteo_aug,df_meteo_jan],axis=0)
df_cons = pd.concat([df_cons_aug,df_cons_jan],axis=0)
df_grid = pd.concat([df_grid_aug,df_grid_jan],axis=0)

df_meteo.reset_index(drop=True, inplace=True)
df_cons.reset_index(drop=True, inplace=True)
df_grid.reset_index(drop=True, inplace=True)

df_grid['Excedentes_cut'] = df_grid['Excedentes']*(1-0.3*df_grid['Hour'].apply(lambda x: 1 if (x in [8,9,10,11,12,13,14,15,16]) else 0))
df_grid['Excedentes'] = df_grid['Excedentes_cut']

# time
T = 48
l_t = list(range(T))
m.t = pyo.Set(initialize=l_t)

# electricity cost
l_cost = df_grid['PVPC']
l_exc = df_grid['Excedentes']
m.cost = pyo.Param(m.t, initialize=l_cost)
m.exc = pyo.Param(m.t, initialize=l_exc)
cost_new_pv = 0.00126*T/1000
cp_bat = 0.00171*T/1000
ce_bat = 0.00856*T/1000
# Costos en €/(kwh·h). Considerant vida de 20 anys, del llibre de Oriol Gomis i Paco Diaz trobem:
#   - Ce = 750 €/kWh (10 anys de vida) -> x2 = 1500 €/kWh -> /20/365/24 = 0.00856 €/(kWh·h)

# ===== Create the system =====
m.ReservoirEbre = pyo.Block()
m.Reservoir1 = pyo.Block()
m.Irrigation1 = pyo.Block()
m.Pump1 = pyo.Block()
# m.Pump2 = pyo.Block()
# m.Turb1 = pyo.Block()
m.Pipe1 = pyo.Block()
m.PV = pyo.Block()
m.Grid = pyo.Block()
m.EBg = pyo.Block()
# m.EBpv = pyo.Block()
m.Bat = pyo.Block()


data_irr = {'Q':df_cons['Qirr']/3600*1.4} # irrigation
Source(m.Irrigation1, m.t, data_irr, {})

data_Ebre = {'dt':3600, 'W0':2e5, 'Wmin':0, 'Wmax':3e5, 'zmin':29.5, 'zmax':30}
init_Ebre = {'Q':[0]*T, 'W':[2e5]*T}
Reservoir(m.ReservoirEbre, m.t, data_Ebre, init_Ebre)
data_R1 = {'dt':3600, 'W0':10e3, 'Wmin':9e3, 'Wmax':13e3, 'zmin':135+(141-135)*9/13, 'zmax':141}
init_R1 = {'Q':[0]*T, 'W':[10e3]*T}
Reservoir(m.Reservoir1, m.t, data_R1, init_R1)

data_c1 = {'K':0.02, 'Qmax':10} # canal
init_c1 = {'Q':[0]*T, 'H':[108]*T, 'H0':[108]*T, 'zlow':[30]*T, 'zhigh':[138]*T}
Pipe(m.Pipe1, m.t, data_c1, init_c1)

# data_p = {'A':121.54, 'B':0.0003, 'n_n':2900, 'eff':0.8, 'eff_t':0.5, 'S':0.1*0.1*3.14, 'Qmin':0.5, 'Qmax':2, 'Qnom':0.0556, 'Pmax':110e3} # pumps (both equal)
# init_p = {'Q':[0]*T, 'H':[108]*T, 'n':[2900]*T, 'Pe':[110e3*0.9]*T}
# # ReversibleRealPump(m.Pump1, m.t, data_p, init_p)
# RealPump(m.Pump1, m.t, data_p, init_p)
# RealPump(m.Pump2, m.t, data_p, init_p)

data_pdouble = {'A':121.54, 'B':0.00007, 'n_n':2900, 'eff':0.8, 'eff_t':0.5, 'S':0.1*0.1*3.14, 'Qmin':0.5, 'Qmax':2*2, 'Qnom':0.0556, 'Pmax':2*110e3} # pumps (both equal)
init_pdouble = {'Q':[0]*T, 'H':[108]*T, 'n':[2900]*T, 'Pe':[110e3*0.9]*T}
# ReversibleRealPump(m.Pump1, m.t, data_p, init_p)
RealPump(m.Pump1, m.t, data_pdouble, init_pdouble)

# data_t = {'eff':0.5, 'Pmax':110e3}
# init_t = {'Q':[0]*T, 'H':[108]*T, 'Pe':[-110e3*0.9]*T}
# Turbine(m.Turb1, m.t, data_t, init_t)

data_pv = {'Pinst':215.28e3, 'Pmax':215.28e3, 'forecast':df_meteo['Irr']/1000, 'eff':0.98} # PV
SolarPV(m.PV, m.t, data_pv)

data_bat = {'dt':3600, 'E0':0.05, 'Emax':200e3, 'Pmax':200e3, 'SOCmin':0.2, 'SOCmax':1.0, 'eff_ch':0.8, 'eff_dc':0.8,'Einst':0.1, 'Pinst':0}
init_bat = {'E':[0.5]*T, 'P':[0]*T}
NewBattery(m.Bat, m.t, data_bat, init_bat)

Grid(m.Grid, m.t, {'Pmax':100e6}) # grid

EB(m.EBg, m.t)
# EB(m.EBpv, m.t)


def ConstraintW1min(m):
    return m.Reservoir1.W[23] >= m.Reservoir1.W0*0.95
    # return m.Reservoir1.W[T-1] >= m.Reservoir1.W[0]*0.95
m.Reservoir1_c_W1min = pyo.Constraint(rule=ConstraintW1min)

def ConstraintW1max(m):
    return m.Reservoir1.W[23] <= m.Reservoir1.W0*1.05
    # return m.Reservoir1.W[T-1] <= m.Reservoir1.W[0]*1.05
m.Reservoir1_c_W1max = pyo.Constraint(rule=ConstraintW1max)

def ConstraintW1minjan(m):
    return m.Reservoir1.W[T-1] >= m.Reservoir1.W0*0.95
m.Reservoir1_c_W1minjan = pyo.Constraint(rule=ConstraintW1minjan)

def ConstraintW1maxjan(m):
    return m.Reservoir1.W[T-1] <= m.Reservoir1.W0*1.05
m.Reservoir1_c_W1maxjan = pyo.Constraint(rule=ConstraintW1maxjan)

# def ConstraintPumpTurb(m, t):
#     return m.Turb1.Qout[t] * m.Pump1.Qout[t] == 0
# m.Turb1_c_PumpTurb = pyo.Constraint(m.t, rule=ConstraintPumpTurb)


# Connections
m.p1r0 = Arc(ports=(m.Pump1.port_Qin, m.ReservoirEbre.port_Q), directed=True)
m.p1c1_Q = Arc(ports=(m.Pump1.port_Qout, m.Pipe1.port_Q), directed=True)
m.p1c1_H = Arc(ports=(m.Pump1.port_H, m.Pipe1.port_H), directed=True)
m.p1eb = Arc(ports=(m.Pump1.port_P, m.EBg.port_P), directed=True)

# m.p2r0 = Arc(ports=(m.Pump2.port_Qin, m.ReservoirEbre.port_Q), directed=True)
# m.p2c1_Q = Arc(ports=(m.Pump2.port_Qout, m.Pipe1.port_Q), directed=True)
# m.p2c1_H = Arc(ports=(m.Pump2.port_H, m.Pipe1.port_H), directed=True)
# m.p2eb = Arc(ports=(m.Pump2.port_P, m.EBg.port_P), directed=True)

# m.t1r0 = Arc(ports=(m.Turb1.port_Qout, m.ReservoirEbre.port_Q), directed=True)
# m.t1c1_Q = Arc(ports=(m.Turb1.port_Qin, m.Pipe1.port_Q), directed=True)
# m.t1c1_H = Arc(ports=(m.Turb1.port_H, m.Pipe1.port_H), directed=True)
# m.t1eb = Arc(ports=(m.Turb1.port_P, m.EBg.port_P), directed=True)

m.c1r1_Q = Arc(ports=(m.Pipe1.port_Q, m.Reservoir1.port_Q), directed=True)
m.c1r1_z = Arc(ports=(m.Reservoir1.port_z, m.Pipe1.port_zhigh), directed=True)
m.c1r0_z = Arc(ports=(m.ReservoirEbre.port_z, m.Pipe1.port_zlow), directed=True)

m.r1i1 = Arc(ports=(m.Irrigation1.port_Qin, m.Reservoir1.port_Q), directed=True)

m.grideb = Arc(ports=(m.Grid.port_P, m.EBg.port_P), directed=True)
m.pveb = Arc(ports=(m.PV.port_P, m.EBg.port_P), directed=True)
m.baten = Arc(ports=(m.Bat.port_P, m.EBg.port_P), directed=True)

pyo.TransformationFactory("network.expand_arcs").apply_to(m) # apply arcs to model


#%% RUN THE OPTIMIZATION
from pyomo.environ import value
import os
import time

# Objective function
def obj_fun(m):
# 	return sum((m.Grid.Pbuy[t]*m.cost[t]/1e6 - m.Grid.Psell[t]*m.exc[t]/1e6) + 0*1/1e6*((m.PV.Pinst+m.PV.Pdim)*m.PV.forecast[t]*m.PV.eff + m.PV.P[t]) for t in l_t ) #+ (m.Bat.Pdim*cp_bat + m.Bat.Edim*ce_bat)/365/20#+ m.PV.Pdim*cost_new_pv
	return sum((m.Grid.Pbuy[t]*m.cost[t]/1e6 - m.Grid.Psell[t]*m.exc[t]/1e6) for t in l_t ) + (m.Bat.Pdim*cp_bat + m.Bat.Edim*ce_bat) #+ m.PV.Pdim*cost_new_pv
m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)

instance = m.create_instance()

start_time = time.time()

# solver = pyo.SolverFactory('asl:couenne') #ipopt asl:couenne gdpopt.enumerate
# solver.options['branch_fbbt'] = 'no'
# solver.solve(instance, tee=True)

os.environ['NEOS_EMAIL'] = 'sergi.costa.dilme@upc.edu'
solver_manager = pyo.SolverManagerFactory('neos')
# results = solver_manager.solve(instance, solver="knitro")
results = solver_manager.solve(instance, solver="couenne")
# results = solver_manager.solve(instance, solver="minlp")
results.write()

# with open("couenne.opt", "w") as file:
#     file.write('''time_limit 100000
#                 convexification_cuts 3
#                 convexification_points 3
#                 delete_redundant yes
#                 use_quadratic no
#                 feas_tolerance 1e-5
#                 ''')
# solver = pyo.SolverFactory('asl:couenne')
# results = solver.solve(instance, tee=True)
# results.write()
# os.remove('couenne.opt') #Delete options

exec_time = time.time() - start_time

#%% GET RESULTS
from Utilities import get_results

file = './results/ISGT/grid_ISGT_140irr_9k_cut'
df_out, df_param, df_size = get_results(file=file, instance=instance, results=results, l_t=l_t, exec_time=exec_time)

#%% PLOTS
from matplotlib import gridspec
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 9,
    'axes.spines.top': False,
    'axes.spines.right': False
})
labels_hours = ['0','','','','','','6','','','','','','12','','','','','','18','','','','','23']

cbcolors = sns.color_palette('colorblind')

file = 'cutprice/pat_ISGT_140irr_9k_cut'
w_lim = 8500

df_meteo_aug = pd.read_csv('data/meteo/LesPlanes_meteo_hour_aug.csv').head(24)
df_cons_aug = pd.read_csv('data/irrigation/LesPlanes_irrigation_aug.csv').head(24)
df_grid_aug = pd.read_csv('data/costs/PVPC_aug.csv').head(24)
df_meteo_jan = pd.read_csv('data/meteo/LesPlanes_meteo_hour_jan.csv').head(24)
df_cons_jan = pd.read_csv('data/irrigation/LesPlanes_irrigation_jan.csv').head(24)
df_grid_jan = pd.read_csv('data/costs/PVPC_jan.csv').head(24)
df_meteo = pd.concat([df_meteo_aug,df_meteo_jan],axis=0)
df_cons = pd.concat([df_cons_aug,df_cons_jan],axis=0)
df_grid = pd.concat([df_grid_aug,df_grid_jan],axis=0)
df_meteo.reset_index(drop=True, inplace=True)
df_cons.reset_index(drop=True, inplace=True)
df_grid.reset_index(drop=True, inplace=True)

df = pd.read_csv('results/ISGT/'+file+'.csv')
df['PV.Pf'] = -df_meteo['Irr']/1000*215.28e3*0.98


if 'Turb1.Qout' in df.columns:
    df['Rev1.Qout'] = df['Pump1.Qout'] - df['Turb1.Qout']
    df['Rev1.Pe'] = df['Pump1.Pe'] + df['Turb1.Pe']
else:
    df['Rev1.Qout'] = df['Pump1.Qout']
    df['Rev1.Pe'] = df['Pump1.Pe']

df['Irrigation1.Qin'] = -df['Irrigation1.Qout']

for c in df.columns:
    df[c] = df[c].apply(lambda x: 0 if (x>-1e-10 and x<1e-10) else x)

df_S = df.iloc[0:24].reset_index(drop=True)
df_W = df.iloc[24:].reset_index(drop=True)
df_W['t'] = df_W.index
df_S['PVPC'] = df_grid['PVPC'][0:24].reset_index(drop=True)
df_W['PVPC'] = df_grid['PVPC'][24:48].reset_index(drop=True)
df_S['Excedentes'] = df_grid['Excedentes'][0:24].reset_index(drop=True)
df_W['Excedentes'] = df_grid['Excedentes'][24:48].reset_index(drop=True)


i=0
season=['S','W']
for df in [df_S, df_W]:
    
    fig = plt.figure(figsize=(3.4, 2.1))
    plt.rcParams['axes.spines.right'] = True
    ax1 = fig.add_subplot(1,1,1)
    # df.apply(lambda x: x/1000).plot(y=['PV.Pf'], kind='bar', ax=ax1, stacked=False, ylabel='P (kW)', 
    #                             color='#F0F0F0', alpha=1, edgecolor='#808080')
    # df.apply(lambda x: x/1000).plot(y=['PV.P','Pump2.Pe','Rev1.Pe'], kind='bar', stacked=True, ax=ax1, ylabel='P (kW)',
    #                                 color=['lightgrey','#606060','#454545'], edgecolor='#808080')
    df.apply(lambda x: x/1000).plot(y=['PV.Pf'], kind='bar', ax=ax1, stacked=False, ylabel='P (kW)', 
                                color='#C0E3C0', alpha=1, edgecolor=None)
    df.apply(lambda x: x/1000).plot(y=['PV.P','Pump2.Pe','Rev1.Pe'], kind='bar', stacked=True, ax=ax1, ylabel='P (kW)',
                                color=[cbcolors[2],cbcolors[0],cbcolors[1]], edgecolor=None)
    ax1.axhline(0,color='k')
    ax1.set_ylim(-200,200)
    bars = ax1.patches
    # patterns =(None, None,'/////',None)
    patterns =(None, None, None, None)
    hatches = [p for p in patterns for i in range(len(df))]
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax1.legend(['_','$\hat{P}_{PV}$','$P_{PV}$','$P_{p,PV}$','$P_{p,g}$'],loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=False, shadow=False, ncol=4)
    
    ax2 = plt.twinx()
    ax2.set_ylim(45,300)
    sns.lineplot(df,x='t' ,y='PVPC', ax=ax2, color='tab:red')#, label='Buy')
    sns.lineplot(df,x='t' ,y='Excedentes', ax=ax2, color='tab:red', linestyle='dashed')#, label='Buy')
    ax2.set_xticks(range(24), labels=labels_hours, rotation=90)
    
    plt.ylabel('Price (€/MWh)')
    plt.xlabel('Hour')
    # plt.title('Power consumption')
    plt.show()
    plt.tight_layout()
    
    plt.rcParams['savefig.format']='pdf'
    plt.savefig('results/ISGT/' + file + season[i] + '_P', dpi=300)
    plt.rcParams['savefig.format']='svg'
    plt.savefig('results/ISGT/' + file + season[i] + '_P', dpi=300)
    
    
    
    fig = plt.figure(figsize=(3.4, 2))
    plt.rcParams['axes.spines.right'] = False
    gs = gridspec.GridSpec(2,1,height_ratios=[2,1])
    
    ax1 = fig.add_subplot(gs[0])
    ax1.set_ylim(-0.2,0.2)
    # df.plot(y=['Pump2.Qout','Irrigation1.Qin','Rev1.Qout'], kind='bar', stacked=True, ax=ax1, ylabel='Q (m$^3$/s)',
    #         color=['#606060','lightgrey','#454545'], edgecolor='#808080')
    df.plot(y=['Pump2.Qout','Irrigation1.Qin','Rev1.Qout'], kind='bar', stacked=True, ax=ax1, ylabel='Q (m$^3$/s)',
            color=[cbcolors[0],cbcolors[2],cbcolors[1]], edgecolor=None)
    bars = ax1.patches
    patterns =(None,None,None)
    hatches = [p for p in patterns for i in range(len(df))]
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax1.legend(['$Q_{p,PV}$', '$Q_{irr}$', '$Q_{p,g}$'], ncols=3)
    ax1.axhline(0,color='k')
    ax1.set_xticks(range(24), labels=labels_hours, rotation=90)
    
    ax2 = fig.add_subplot(gs[1],sharex=ax1)
    ax2.figsize=(3.4,1)
    ax2.set_ylim(7500,14000)
    ax2.set_yticks([7500,10000,13000])
    ax2.axhline(w_lim,color='#AFAFAF', alpha=1)
    ax2.axhline(13000,color='#AFAFAF', alpha=1)
    ax2.axhline(10000*0.95, color='#AFAFAF', linestyle='--', alpha=1)
    ax2.axhline(10000*1.05, color='#AFAFAF', linestyle='--', alpha=1)
    sns.lineplot(df, x='t', y='Reservoir1.W', ax=ax2, color='tab:red')
    
    plt.ylabel('R1 Volume (m$^3$)')
    plt.xlabel('Hour')
    plt.xticks(range(24),labels=labels_hours,rotation=90)
    plt.tight_layout()
    plt.show()

    plt.rcParams['savefig.format']='pdf'
    plt.savefig('results/ISGT/' + file + season[i] + '_Q', dpi=300)
    plt.rcParams['savefig.format']='svg'
    plt.savefig('results/ISGT/' + file + season[i] + '_Q', dpi=300)
    
    i=i+1



# df = pd.read_csv('data/Analogiques.csv')
# df['date'] = pd.to_datetime(df['Fecha'], format='%Y/%m/%d')
# df['month'] = df['date'].dt.month
# df['weekday'] = df['date'].dt.dayofweek
# df['Volum bassa'] = df['Nivell bassa']*13000/4.5
# df['Qdiff'] = df['Volum bassa'] - df['Volum bassa'].shift(1) 
# df['Qreg'] = df['Cabal impulsio'] - df['Qdiff']
# df['Qreg'] = df['Qreg'].apply(lambda x: 0 if x<0 else x)
# df['Qreg_lag'] = df['Qreg'].shift(1)
# df.loc[df['Qreg']>300,'Qreg'] = df['Qreg_lag']

# df['Season'] = df['month']/4
# df['Season'] = df['Season'].astype(int)
# fig = plt.figure(figsize=(3.4, 2.3))
# sns.lineplot(df, x='hour', y='Qreg', style='Season', color='tab:grey')
# plt.legend(labels=['Winter','_','Spring','_','Summer','_','Fall','_'], ncol=1)
# plt.xlabel('Hour')
# plt.ylabel('Irrigation demand (m$^3$/h)')
# plt.xticks(range(24),labels=labels_hours, rotation=90)
# plt.ylim([0,250])
# plt.tight_layout()
# plt.show()
# plt.rcParams['savefig.format']='pdf'
# plt.savefig('results/ISGT/Irrigation_season', dpi=300)

# df['Winter'] = df['date'].apply(lambda x: 0 if (x.month in [3,4,5,6,7,8]) else 1)
# fig = plt.figure(figsize=(3.4, 1.5))
# sns.lineplot(df, x='hour', y='Qreg', style='Winter', hue='Winter', palette=[cbcolors[1],cbcolors[0]], errorbar=('ci',90))
# plt.legend(labels=['S','_','W','_'], ncol=1)
# plt.xlabel('Hour')
# plt.ylabel('Demand (m$^3$/h)')
# plt.xticks(range(24),labels=labels_hours, rotation=90)
# plt.yticks([0,100,200])
# # plt.ylim([0,200])
# plt.tight_layout()
# plt.show()
# plt.rcParams['savefig.format']='pdf'
# plt.savefig('results/ISGT/Irrigation_WS', dpi=300)
# plt.rcParams['savefig.format']='svg'
# plt.savefig('results/ISGT/Irrigation_WS', dpi=300)

#%% PLOTS (PARALLEL PUMP)
from matplotlib import gridspec
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 9,
    'axes.spines.top': False,
    'axes.spines.right': False
})
labels_hours = ['0','','','','','','6','','','','','','12','','','','','','18','','','','','23']

cbcolors = sns.color_palette('colorblind')

file = 'cutprice/gridpat_ISGT_140irr_9k_cut'
w_lim = 9000

df_meteo_aug = pd.read_csv('data/meteo/LesPlanes_meteo_hour_aug.csv').head(24)
df_cons_aug = pd.read_csv('data/irrigation/LesPlanes_irrigation_aug.csv').head(24)
df_grid_aug = pd.read_csv('data/costs/PVPC_aug.csv').head(24)
df_meteo_jan = pd.read_csv('data/meteo/LesPlanes_meteo_hour_jan.csv').head(24)
df_cons_jan = pd.read_csv('data/irrigation/LesPlanes_irrigation_jan.csv').head(24)
df_grid_jan = pd.read_csv('data/costs/PVPC_jan.csv').head(24)
df_meteo = pd.concat([df_meteo_aug,df_meteo_jan],axis=0)
df_cons = pd.concat([df_cons_aug,df_cons_jan],axis=0)
df_grid = pd.concat([df_grid_aug,df_grid_jan],axis=0)
df_meteo.reset_index(drop=True, inplace=True)
df_cons.reset_index(drop=True, inplace=True)
df_grid.reset_index(drop=True, inplace=True)

df = pd.read_csv('results/ISGT/'+file+'.csv')
df['PV.Pf'] = -df_meteo['Irr']/1000*215.28e3*0.98


if 'Turb1.Qout' in df.columns:
    df['Rev1.Qout'] = df['Pump1.Qout'] - df['Turb1.Qout']
    df['Rev1.Pe'] = df['Pump1.Pe'] + df['Turb1.Pe']
else:
    df['Rev1.Qout'] = df['Pump1.Qout']
    df['Rev1.Pe'] = df['Pump1.Pe']
    
for c in df.columns:
    df[c] = df[c].apply(lambda x: 0 if (x>-1e-10 and x<1e-10) else x)

df_S = df.iloc[0:24].reset_index(drop=True)
df_W = df.iloc[24:].reset_index(drop=True)
df_W['t'] = df_W.index
df_S['PVPC'] = df_grid['PVPC'][0:24].reset_index(drop=True)
df_W['PVPC'] = df_grid['PVPC'][24:48].reset_index(drop=True)
df_S['Excedentes'] = df_grid['Excedentes'][0:24].reset_index(drop=True)
df_W['Excedentes'] = df_grid['Excedentes'][24:48].reset_index(drop=True)


i=0
season=['S','W']
for df in [df_S, df_W]:
    
    fig = plt.figure(figsize=(3.4, 2.1))
    plt.rcParams['axes.spines.right'] = True
    ax1 = fig.add_subplot(1,1,1)
    # df.apply(lambda x: x/1000).plot(y=['PV.Pf'], kind='bar', ax=ax1, stacked=False, ylabel='P (kW)', 
    #                             color='#F0F0F0', alpha=1, edgecolor='#808080')
    # df.apply(lambda x: x/1000).plot(y=['PV.P','Pump2.Pe','Rev1.Pe'], kind='bar', stacked=True, ax=ax1, ylabel='P (kW)',
    #                                 color=['lightgrey','#606060','#454545'], edgecolor='#808080')
    df.apply(lambda x: x/1000).plot(y=['PV.Pf'], kind='bar', ax=ax1, stacked=False, ylabel='P (kW)', 
                                color='#C0E3C0', alpha=1, edgecolor=None)
    df.apply(lambda x: x/1000).plot(y=['PV.P','Rev1.Pe'], kind='bar', stacked=True, ax=ax1, ylabel='P (kW)',
                                color=[cbcolors[2],cbcolors[1]], edgecolor=None)
    ax1.axhline(0,color='k')
    ax1.set_ylim(-200,200)
    bars = ax1.patches
    # patterns =(None, None,'/////',None)
    patterns =(None, None, None, None)
    hatches = [p for p in patterns for i in range(len(df))]
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax1.legend(['_','$\hat{P}_{PV}$','$P_{PV}$','$P_{p}$'],loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=False, shadow=False, ncol=4)
    
    ax2 = plt.twinx()
    ax2.set_ylim(45,300)
    sns.lineplot(df,x='t' ,y='PVPC', ax=ax2, color='tab:red')#, label='Buy')
    sns.lineplot(df,x='t' ,y='Excedentes', ax=ax2, color='tab:red', linestyle='dashed')#, label='Buy')
    ax2.set_xticks(range(24), labels=labels_hours, rotation=90)
    
    plt.ylabel('Price (€/MWh)')
    plt.xlabel('Hour')
    # plt.title('Power consumption')
    plt.show()
    plt.tight_layout()
    
    plt.rcParams['savefig.format']='pdf'
    plt.savefig('results/ISGT/' + file + season[i] + '_P', dpi=300)
    plt.rcParams['savefig.format']='svg'
    plt.savefig('results/ISGT/' + file + season[i] + '_P', dpi=300)
    
    
    
    fig = plt.figure(figsize=(3.4, 2))
    plt.rcParams['axes.spines.right'] = False
    gs = gridspec.GridSpec(2,1,height_ratios=[2,1])
    
    ax1 = fig.add_subplot(gs[0])
    ax1.set_ylim(-0.2,0.2)
    # df.plot(y=['Pump2.Qout','Irrigation1.Qin','Rev1.Qout'], kind='bar', stacked=True, ax=ax1, ylabel='Q (m$^3$/s)',
    #         color=['#606060','lightgrey','#454545'], edgecolor='#808080')
    df.plot(y=['Irrigation1.Qin','Rev1.Qout'], kind='bar', stacked=True, ax=ax1, ylabel='Q (m$^3$/s)',
            color=[cbcolors[2],cbcolors[1]], edgecolor=None)
    bars = ax1.patches
    patterns =(None,None,None)
    hatches = [p for p in patterns for i in range(len(df))]
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax1.legend(['$Q_{irr}$','$Q_{p}$'], ncols=2)
    ax1.axhline(0,color='k')
    ax1.set_xticks(range(24), labels=labels_hours, rotation=90)
    
    ax2 = fig.add_subplot(gs[1],sharex=ax1)
    ax2.figsize=(3.4,1)
    ax2.set_ylim(7500,14000)
    ax2.set_yticks([7500,10000,13000])
    ax2.axhline(w_lim,color='#AFAFAF', alpha=1)
    ax2.axhline(13000,color='#AFAFAF', alpha=1)
    ax2.axhline(10000*0.95, color='#AFAFAF', linestyle='--', alpha=1)
    ax2.axhline(10000*1.05, color='#AFAFAF', linestyle='--', alpha=1)
    sns.lineplot(df, x='t', y='Reservoir1.W', ax=ax2, color='tab:red')
    
    plt.ylabel('R1 Volume (m$^3$)')
    plt.xlabel('Hour')
    plt.xticks(range(24),labels=labels_hours,rotation=90)
    plt.tight_layout()
    plt.show()

    plt.rcParams['savefig.format']='pdf'
    plt.savefig('results/ISGT/' + file + season[i] + '_Q', dpi=300)
    plt.rcParams['savefig.format']='svg'
    plt.savefig('results/ISGT/' + file + season[i] + '_Q', dpi=300)
    
    i=i+1



#%%
logging.INFO = 20
log_infeasible_constraints(instance)

# with open('eq', 'w') as f:
#     instance.pprint(f)