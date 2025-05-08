"""
    AGISTIN - EXAMPLE CEDER
    
    3 Reservoirs wihout irrigation connected to a battery and a PV system.
    The system is supposed to service electrical loads.
    
    Authors: Sergi Costa Dilmé (CITCEA-UPC), Juan Carlos Olives-Camps (CITCEA-UPC), Paula Muñoz Peña (CITCEA-UPC), Pau García (CITCEA-UPC)
"""


from pyomo.util.infeasible import log_infeasible_constraints
import logging
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import pyomo
import pyomo.environ as pyo
from pyomo.network import Arc, Port

# Import devices
from Devices.Reservoirs import Reservoir
from Devices.Sources import Source
from Devices.Pipes import Pipe, PipeValve
from Devices.Pumps import Pump, RealPump, DiscretePump
from Devices.Turbines import Turbine
from Devices.EB import EB
from Devices.SolarPV import SolarPV
from Devices.MainGrid import Grid
from Devices.Batteries import Battery, NewBattery
from Devices.HydroSwitch import HydroSwitch
from Devices.Load import Load
from Devices.Sources import VarSource
from pyomo.contrib.gdpopt.enumerate import GDP_Enumeration_Solver

# Import useful functions
from Utilities import clear_clc

#clean console and variable pane
# clear_clc() #consider removing if you are not working with Spyder

# model
m = pyo.ConcreteModel()

df_data = pd.read_csv('G:/Unidades compartidas/CITCEA.PRJ_214_HorizonEurope_AGISTIN/06. documentacio tecnica/Dades/Datasets/Final/hour/CEDER.csv')
df_data = df_data.set_index('Date')
df_data = df_data.groupby(['Date', 'Hour']).mean().reset_index()
df_data = df_data.set_index('Date')
df_data = df_data.drop('N_deposito',axis=1)


#-----------Data analysis---------------#

from scipy import stats
import numpy as np


df_data['Date'] = pd.to_datetime(df_data.index)
df_data['Month'] = df_data['Date'].dt.strftime('%m')



winter = ['12','01','02','03']
summer = ['06','07','08','09']


df_winter = df_data[df_data['Month'].isin(winter)]
df_summer = df_data[df_data['Month'].isin(summer)]


df_winter = df_winter.drop('Date',axis=1)
df_summer = df_summer.drop('Date',axis=1)

df_winter = df_winter.drop('Month',axis=1)
df_summer = df_summer.drop('Month',axis=1)




winter_day_mean = df_winter.groupby('Hour').mean()
summer_day_mean = df_summer.groupby('Hour').mean()

winter_day_median = df_winter.groupby('Hour').median()
summer_day_median = df_summer.groupby('Hour').median()

winter_var = df_winter.groupby('Hour').var()
summer_var = df_summer.groupby('Hour').var()

# plt.figure()
# sns.boxplot(x='Hour', y='Irr', data=df_winter)
# sns.boxplot(x='Hour', y='Irr', data=df_summer)
# plt.show()


# plt.figure()
# sns.lineplot(x='Hour',y = 'Irr',data= winter_day_mean)
# sns.lineplot(x='Hour', y = 'Irr', data = winter_day_median)
# plt.show()

# plt.figure()
# sns.lineplot(x='Hour',y = 'Irr',data= summer_day_mean)
# sns.lineplot(x='Hour', y = 'Irr', data = summer_day_median)
# plt.show()


# plt.figure()
# sns.lineplot(x='Hour',y = 'CT_Edificio1',data= winter_day_mean)
# sns.lineplot(x='Hour', y = 'CT_Edificio1', data = winter_day_median)
# plt.show()

# plt.figure()
# sns.lineplot(x='Hour',y = 'CT_Edificio1',data= summer_day_mean)
# sns.lineplot(x='Hour', y = 'CT_Edificio1', data = summer_day_median)
# plt.show()

# grouped = df_winter.groupby('Hour')['Irr']
# mean_irr = grouped.mean()
# std_irr = grouped.std()
# count_irr = grouped.count()

# confidence_level = 0.95
# z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)

# margin_of_error = z_score * (std_irr / np.sqrt(count_irr))

# ci_lower = mean_irr - margin_of_error
# ci_upper = mean_irr + margin_of_error

# plt.figure(figsize=(12, 8))
# sns.boxplot(x='Hour', y='Irr', data=df_winter, whis=[5, 95], showfliers=False)

# plt.plot(mean_irr.index, mean_irr, color='red', label='Media')
# plt.fill_between(mean_irr.index, ci_lower, ci_upper, color='red', alpha=0.2, label='95% IC')

# plt.title('Box Plot de Irr por Hora (Invierno) con Intervalos de Confianza')
# plt.xlabel('Hora')
# plt.ylabel('Irradiación')
# plt.legend()
# plt.grid(True)
# plt.show()


#------Optimization problem------------

# Take in account the season of the optimization (winter or summer)
ho=0
hf=24

df_meteo = pd.DataFrame()
df_grid =  pd.read_csv('data/costs/PVPC_jan.csv').iloc[ho:hf].reset_index(drop=True)
df_meteo['Irr']  =  winter_day_mean['Irr']
df_load = - winter_day_mean['CT_Edificio1']

df_meteo.reset_index(drop=True, inplace=True)
df_load.reset_index(drop=True, inplace=True)
df_grid.reset_index(drop=True, inplace=True)

# time
T = 24
l_t = list(range(T))
m.ts = pyo.Set(initialize=l_t)

# electricity cost
demands = df_load

l_costs = df_grid['PVPC']

l_excs = df_grid['Excedentes']


m.costs = pyo.Param(m.ts, initialize=l_costs)
m.excs = pyo.Param(m.ts, initialize=l_excs)
        

cost_new_pv = 0.00126*T/1000
cp_bat = 0.00171*T/1000
ce_bat = 0.00856*T/1000

# Costos en €/(kwh·h). Considerant vida de 20 anys, del llibre de Oriol Gomis i Paco Diaz trobem:
#   - Ce = 750 €/kWh (10 anys de vida) -> x2 = 1500 €/kWh -> /20/365/24 = 0.00856 €/(kWh·h)


# ===== Create the system =====

m.Reservoir0s = pyo.Block()
m.Reservoir1s = pyo.Block()
m.Reservoir2s = pyo.Block()
m.Pump1s = pyo.Block()
m.Pump2s = pyo.Block()
m.Turb1s = pyo.Block()
m.Pipe01s = pyo.Block()
m.Pipe02s = pyo.Block()
m.VarSources = pyo.Block()
m.Pipe01Ts = pyo.Block()
m.PVs = pyo.Block()
m.Grids = pyo.Block()
m.EBs = pyo.Block()
# m.Bats = pyo.Block()
m.BatsDim = pyo.Block()
m.Loads = pyo.Block()




data_R0 = {'dt':3600, 'W0':2e3, 'Wmin':0, 'Wmax':2.1e3, 'zmin':1, 'zmax':4}
init_R0 = {'Q':[0]*T, 'W':[2e3]*T}
Reservoir(m.Reservoir0s, m.ts, data_R0, init_R0)


data_R1 = {'dt':3600, 'W0':0e3, 'Wmin':0e3, 'Wmax':1.5e3, 'zmin':67, 'zmax':69}
init_R1 = {'Q':[0]*T, 'W':[0.5e3]*T}
Reservoir(m.Reservoir1s, m.ts, data_R1, init_R1)

data_R2 = {'dt':3600, 'W0':0e3, 'Wmin':0e3, 'Wmax':0.5e3, 'zmin':77, 'zmax':79}
init_R2 = {'Q':[0]*T, 'W':[0.25e3]*T}
Reservoir(m.Reservoir2s, m.ts, data_R2, init_R2)

data_c1 = {'K':200, 'Qmax':0.5} # canal
init_c1 = {'Q':[0]*T, 'H':[67]*T, 'H0':[67]*T, 'zlow':[0]*T, 'zhigh':[67]*T}
Pipe(m.Pipe01s, m.ts, data_c1, init_c1)
Pipe(m.Pipe01Ts, m.ts, data_c1, init_c1)


data_c2 = {'K':200, 'Qmax':0.5} # canal
init_c2 = {'Q':[0]*T, 'H':[77]*T, 'H0':[77]*T, 'zlow':[2]*T, 'zhigh':[67]*T}
Pipe(m.Pipe02s, m.ts, data_c2, init_c2)


data_Source = {'Qmax':1,'Qmin':0}
init_Source = {'Q':[0.0]*T,'Qin':[0]*T,'Qout':[0]*T}
VarSource(m.VarSources, m.ts, data_Source, init_Source )


data_p1 = {'A':121.54, 'B':0.0003, 'n_n':2900, 'eff':0.8, 'S':0.1*0.1*3.14,'Pn':7.5e3, 'Npumps':4,'Qnom':0.05,'Qmax':2,'Qmin':0,'Pmax':30e3} # pumps (both equal)
init_p1 = {'Q':[0]*T, 'H':[67]*T, 'n':[2900]*T, 'Pe':[7.5e3]*T}
DiscretePump(m.Pump1s, m.ts, data_p1, init_p1)
DiscretePump(m.Pump2s, m.ts, data_p1, init_p1)


data_t = {'eff':0.8, 'Pmax':40e3}
init_t = {'Q':[0]*T, 'H':[67]*T, 'Pe':[-40e3*0.8]*T}
Turbine(m.Turb1s, m.ts, data_t, init_t)

data_pvs = {'Pinst':16e3, 'Pmax':66e3, 'forecast':df_meteo['Irr']/1000, 'eff':0.98} # PV

SolarPV(m.PVs, m.ts, data_pvs)

# data_bat = {'dt':1, 'Emax':400e3,'Pmax':90e3, 'SOCmin':0.1, 'SOCmax':0.80, 'eff_ch':0.9, 'eff_dc':0.9}
# init_bat = {'E':[45e3]*T, 'P':[0]*T}
# NewBattery(m.Bats, m.ts, data_bat, init_bat)

# m.Bats.Pdim.fix(90e3)
# m.Bats.Edim.fix(400e3)


data_batdim = {'dt':1,'Emax':400e3, 'Pmax':90e3, 'SOCmin':0.1, 'SOCmax':0.80, 'eff_ch':0.9, 'eff_dc':0.9}
init_batdim = {'E':[45e3]*T, 'P':[0]*T}
NewBattery(m.BatsDim, m.ts, data_batdim, init_batdim)


Grid(m.Grids, m.ts, {'Pmax':100e3}) # grid


data_loads = {'P':demands} # Electrical Demand

Load(m.Loads, m.ts, data_loads, {})


EB(m.EBs, m.ts)


def ConstraintPumpOnOffs (m,ts):
    return m.Pump1s.Pe[ts] * m.Pump2s.Pe[ts] == 0
m.p1p2s = pyo.Constraint(m.ts, rule = ConstraintPumpOnOffs)


#Energy constraint

# def ConstraintW1min(m):
#     return m.Reservoir0.W[T-1]*m.Reservoir0.z[T-1] + m.Reservoir1.W[T-1]*m.Reservoir1.z[T-1] + m.Reservoir2.W[T-1]*m.Reservoir2.z[T-1]  >= (m.Reservoir0.W0*m.Reservoir0.z[0] + m.Reservoir1.W0*m.Reservoir1.z[0] + m.Reservoir2.W0*m.Reservoir2.z[0])*0.9
# m.Reservoir1_c_W1min = pyo.Constraint(rule=ConstraintW1min)


# def ConstraintW1max(m):
#     return m.Reservoir0.W[T-1]*m.Reservoir0.z[T-1] + m.Reservoir1.W[T-1]*m.Reservoir1.z[T-1] + m.Reservoir2.W[T-1]*m.Reservoir2.z[T-1]  >= (m.Reservoir0.W0*m.Reservoir0.z[0] + m.Reservoir1.W0*m.Reservoir1.z[0] + m.Reservoir2.W0*m.Reservoir2.z[0])*1.1
# m.Reservoir1_c_W1max = pyo.Constraint(rule=ConstraintW1max)



# Connections

m.p1r0s = Arc(ports=(m.Pump1s.port_Qin, m.Reservoir0s.port_Q), directed=True)
m.p1ebs = Arc(ports=(m.Pump1s.port_P, m.EBs.port_P), directed=True)
m.p1h1_Qs = Arc(ports=(m.Pump1s.port_Qout, m.Pipe01s.port_Q), directed=True)
m.p1h1_Hs = Arc(ports=(m.Pump1s.port_H, m.Pipe01s.port_H), directed=True)



m.p2r0s = Arc(ports=(m.Pump2s.port_Qin, m.Reservoir0s.port_Q), directed=True)
m.p2ebs = Arc(ports=(m.Pump2s.port_P, m.EBs.port_P), directed=True)
m.p2h2_Qs = Arc(ports=(m.Pump2s.port_Qout, m.Pipe02s.port_Q), directed=True)
m.p2h2_Hs = Arc(ports=(m.Pump2s.port_H, m.Pipe02s.port_H), directed=True)


#Turbine to Reservoir1
m.t1r0s = Arc(ports=(m.Turb1s.port_Qout, m.Reservoir0s.port_Q), directed=True)
m.t1c1_Qs = Arc(ports=(m.Turb1s.port_Qin, m.Pipe01Ts.port_Q), directed=True)
m.t1c1_Hs = Arc(ports=(m.Turb1s.port_H, m.Pipe01Ts.port_H), directed=True)
m.t1ebs = Arc(ports=(m.Turb1s.port_P, m.EBs.port_P), directed=True)


#Pipe 01T

m.c01tr1s = Arc(ports=(m.Pipe01Ts.port_Q, m.Reservoir1s.port_Q),directed=True)
m.c01tr1_zs = Arc(ports=(m.Reservoir1s.port_z, m.Pipe01Ts.port_zhigh), directed=True)
m.c01tr0_zs = Arc(ports=(m.Reservoir0s.port_z, m.Pipe01Ts.port_zlow), directed=True)


#Pipe 01 to Reservoir 1
m.c01r1_Qs = Arc(ports=(m.Pipe01s.port_Q, m.Reservoir1s.port_Q), directed=True)
m.c01r1_zs = Arc(ports=(m.Reservoir1s.port_z, m.Pipe01s.port_zhigh), directed=True)
m.c01r0_zs = Arc(ports=(m.Reservoir0s.port_z, m.Pipe01s.port_zlow), directed=True)


#Pipe 02 to R2 and R0
m.c02r2_Qs = Arc(ports=(m.Pipe02s.port_Q, m.Reservoir2s.port_Q), directed=True)
m.c02r2_zs = Arc(ports=(m.Reservoir2s.port_z, m.Pipe02s.port_zhigh), directed=True)
m.c02r0_zs = Arc(ports=(m.Reservoir0s.port_z, m.Pipe02s.port_zlow), directed=True)

m.c21r2_Qs = Arc(ports=(m.VarSources.port_Qin, m.Reservoir2s.port_Q), directed=True)
m.c21r1_Qs = Arc(ports=(m.VarSources.port_Qout, m.Reservoir1s.port_Q), directed=True)



#Electrical Node

m.pvebs = Arc(ports=(m.PVs.port_P, m.EBs.port_P), directed=True)
m.gridebs = Arc(ports=(m.Grids.port_P, m.EBs.port_P), directed=True)
# m.batebs = Arc(ports=(m.Bats.port_P, m.EBs.port_P), directed = True)
m.loadebs = Arc(ports=(m.Loads.port_Pin, m.EBs.port_P), directed=True)
m.batebsdim = Arc(ports=(m.BatsDim.port_P, m.EBs.port_P), directed = True)



pyo.TransformationFactory("network.expand_arcs").apply_to(m) # apply arcs to model

#%%

from pyomo.environ import value
import os
import time

# Objective function
def obj_fun(m):
    return sum((m.Grids.Pbuy[t]*m.costs[t]/1e6 - m.Grids.Psell[t]*m.excs[t]/1e6) for t in l_t )+ m.PVs.Pdim*cost_new_pv  + (m.BatsDim.Pdim*cp_bat + m.BatsDim.Edim*ce_bat)


m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)

instance, results, exec_time = solve(m,'bonmin')


# m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)
# instance = m.create_instance()

# start_time = time.time()

# solver = pyo.SolverFactory('asl:couenne') #ipopt asl:couenne gdpopt.enumerate
# solver.options['branch_fbbt'] = 'no'
# solver.solve(instance, tee=True)

# os.environ['NEOS_EMAIL'] = 'pau.garcia.motilla@upc.edu'
# solver_manager = pyo.SolverManagerFactory('neos')
# # results = solver_manager.solve(instance, solver="knitro")
# results = solver_manager.solve(instance, solver="couenne")
# # results = solver_manager.solve(instance, solver="minlp")
# results.write()

# with open("couenne.opt", "w") as file:
#     file.write('''time_limit 4000
#                 convexification_cuts 2
#                 convexification_points 2
#                 delete_redundant yes
#                 use_quadratic no
#                 feas_tolerance 1e-1
#                 ''')

# solver = pyo.SolverFactory('asl:couenne')
# results = solver.solve(instance, tee=True)
# results.write()
# os.remove('couenne.opt') #Delete options
 
#%%
def solve(m, solver='couenne'):

    instance = m.create_instance()

    start_time = time.time()

    # if 'neos' in solver:w
    #     os.environ['NEOS_EMAIL'] = 'sergi.costa.dilme@upc.edu'
    #     solver_manager = pyo.SolverManagerFactory('neos')
        
    if solver=='couenne':
        with open("couenne.opt", "w") as file:
            file.write('''time_limit 100000
                        convexification_cuts 1
                        convexification_points 2
                        use_quadratic no
                        feas_tolerance 1e-1
                        ''')
        solver = pyo.SolverFactory('asl:couenne')
        results = solver.solve(instance, tee=True)
        results.write()
        os.remove('couenne.opt') #Delete options

    # elif solver=='bonmin':
    #     solver = pyo.SolverFactory('bonmin')
    #     results = solver.solve(instance, tee=True)

    # B-BB, B-OA, B-QG, B-Hyb, B-Ecp, B-iFP
    elif solver=='bonmin':
        with open("bonmin.opt", "w") as file:
            file.write('''bonmin.algorithm B-ECP
                       bonmin.ecp_abs_tol 0.0001
                       bonmin.warm_start optimum
                       tol 0.0001
                        ''')
        solver = pyo.SolverFactory('bonmin')
        results = solver.solve(instance, tee=True)

    elif solver=='ipopt':
        solver = pyo.SolverFactory('ipopt')
        solver.options['max_iter'] = 1000000
        results = solver.solve(instance, tee=True)

    elif solver=='neos_couenne':
        results = solver_manager.solve(instance, solver="couenne")
        
    elif solver=='neos_knitro':
        results = solver_manager.solve(instance, solver="knitro")

    exec_time = time.time() - start_time
    
    return instance, results, exec_time

#%%

instance.Reservoir0s.W.pprint()
instance.Reservoir1s.W.pprint()
instance.Reservoir2s.W.pprint()

instance.Pump1s.Qout.pprint()
instance.Pump2s.Qout.pprint()
instance.Turb1s.Pe.pprint()


instance.Grids.Psell.pprint()
instance.Grids.Pbuy.pprint()

#%% GET RESULTS
from Utilities import get_results

file = './results/CEDER/Case2_24hWinter_ECP'
df_out, df_param, df_size = get_results(file=file, instance=instance, results=results, l_t=l_t, exec_time=exec_time)
#%%


#Plots

file = './results/CEDER/Case3_24hWinter_ECP'
df_out = pd.read_csv(file+'.csv').drop('Unnamed: 0',axis=1)
df_param = pd.read_csv(file+'_param.csv').drop('Unnamed: 0',axis=1)
df_size = pd.read_csv(file+'_size.csv').drop('Unnamed: 0',axis=1)


cbcolors = sns.color_palette('colorblind')

df_out['Loads.Pin'] = -df_out['Loads.Pout']

df_out['Pumps.Pe'] = df_out['Pump1s.Pe']+df_out['Pump2s.Pe']

df_melted = pd.melt(df_out, id_vars=['t'], value_vars=['Pumps.Pe', 'PVs.P', 'Loads.Pin', 'Turb1s.Pe', 'Grids.P'],
                    var_name='Variable', value_name='Value')
df_melted['Value'] = df_melted['Value'] / 1000

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Nimbus Roman No9 L"],
    "font.size": 9,
    'axes.spines.top': False,
    'axes.spines.right': False
})

labels_hours = ['0','','','','','','6','','','','','','12','','','','','','18','','','','','23']


variable_names = {
    'Pumps.Pe': 'Pumps',
    'PVs.P': 'PV',
    'Loads.Pin': 'Load',
    'Turb1s.Pe': 'Turbine',
    'Grids.P': 'Grid',
    'Bats.P': 'Battery'
}

df_melted['Variable'] = df_melted['Variable'].replace(variable_names)

colors_dict = {
    'PV': sns.color_palette("colorblind")[2],
    'Pumps': sns.color_palette("colorblind")[0],
    'Grid': sns.color_palette("colorblind")[7],
    'Battery': sns.color_palette("colorblind")[4],
    'Load': sns.color_palette("colorblind")[3],
    'Turbine': sns.color_palette("colorblind")[1]
}

import matplotlib.lines as mlines

fig, ax1 = plt.subplots(figsize=(3.4, 2.4))  

df_pivot = df_melted.pivot(index='t', columns='Variable', values='Value').fillna(0)

column_order = ['PV', 'Load','Pumps'] + [col for col in df_pivot.columns if col not in ['PV', 'Load','Pumps']]
df_pivot = df_pivot[column_order]

df_positive = df_pivot.clip(lower=0)
df_negative = df_pivot.clip(upper=0)

bottom_pos = np.zeros(len(df_positive))
for col in df_positive.columns:
    ax1.bar(df_positive.index, df_positive[col], bottom=bottom_pos, label=col, color=colors_dict[col])
    bottom_pos += df_positive[col]

bottom_neg = np.zeros(len(df_negative))
for col in df_negative.columns:
    ax1.bar(df_negative.index, df_negative[col], bottom=bottom_neg, color=colors_dict[col])
    bottom_neg += df_negative[col]

ax1.axhline(0, color='black', linewidth=1)
ax1.set_ylim(-45, 45)

ax1.set_ylabel('P (kW)', fontsize=8, loc='center', labelpad=2)
ax1.text(24.5,-55,'Time (h)')
ax1.set_xticks(range(0, 24))
ax1.set_xticklabels(labels_hours, fontsize=7, rotation=0)

ax1_twin = ax1.twinx()

# Graficar las líneas principales
ax1_twin.plot(df_out['t'], df_grid['PVPC'], color='tab:red', linestyle='solid', linewidth=1.5, label='$c_{buy,g}$')
ax1_twin.plot(df_out['t'], df_grid['Excedentes'], color='tab:red', linestyle='dashed', linewidth=1.5, label='$c_{sell,g}$')

# Agregar una línea blanca debajo de las líneas del lineplot para dar el efecto de borde
ax1_twin.plot(df_out['t'], df_grid['PVPC'], color='white', linestyle='solid', linewidth=2.25, zorder=-1)
ax1_twin.plot(df_out['t'], df_grid['Excedentes'], color='white', linestyle='dashed', linewidth=3, zorder=-1)

ax1_twin.set_ylabel('Price (€/MWh)', fontsize=8, color='black')
ax1_twin.tick_params(axis='y', labelcolor='black')

ax1_twin.spines["right"].set_visible(True)
ax1_twin.spines["right"].set_color("black")
ax1_twin.spines["right"].set_linewidth(1)
ax1_twin.set_ylim(45,300)

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax1_twin.get_legend_handles_labels()

handles = handles1 + handles2
labels = labels1 + labels2

ax1.legend(handles, labels, loc='upper center', fontsize=7, bbox_to_anchor=(0.5, -0.22),
           fancybox=False, shadow=False, ncol=4)

plt.tight_layout()
plt.rcParams['savefig.format'] = 'pdf'
plt.savefig(file + 'Electrical', dpi=300)
plt.show()



df = df_out
colors = sns.color_palette("colorblind", len(df_pivot.columns))



#----------------- 2nd Plot, Volume and Flow--------------------------

fig = plt.figure(figsize=(3.4, 2.4))

ax2 = fig.add_subplot(2, 1, 1)
df[['Pipe01s.Q', 'Pipe02s.Q', 'Turb1s.Qout']].plot(
    kind='bar', 
    stacked=False, 
    ylabel='Flow (m³/s)', 
    ax=ax2, 
    color=[cbcolors[2], cbcolors[1], cbcolors[0]], 
    sharex=True, 
    width=1.5
)

ax2.set_xlabel('', fontsize=8)
ax2.set_ylabel('Flow (m³/s)', fontsize=8)
ax2.set_xticklabels(labels_hours, fontsize=7, rotation=0) 
ax2.legend(['Pipe1', 'Pipe2', 'Turbine'], fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=3)
ax2.text(24.5,-0.005,'Time (h)')

fig.add_subplot(2, 1, 2)
df_out['Reservoir0s.W'] = df_out['Reservoir0s.W']/2500*100
df_out['Reservoir1s.W'] = df_out['Reservoir1s.W']/1500*100
df_out['Reservoir2s.W'] = df_out['Reservoir2s.W']/500*100

ax1 = sns.lineplot(data=df, x=df.index, y='Reservoir1s.W', label='R1', color=cbcolors[2], marker='^')
sns.lineplot(data=df, x=df.index, y='Reservoir0s.W', label='R0', color=cbcolors[0], marker='v')
sns.lineplot(data=df, x=df.index, y='Reservoir2s.W', label='R2', color=cbcolors[1], marker='x')

# ax1.set_title("Reservoirs Level", fontsize=10)
ax1.set_xticks(range(0, len(df)))
ax1.set_xticklabels(labels_hours, fontsize=7, rotation=0)  # Asigna las etiquetas de las horas
ax1.set_xlabel('', fontsize=8)
ax1.set_ylabel('Volume (\%)', fontsize=8)
ax1.text(24.5,-25,'Time (h)')
ax1.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=3)

plt.subplots_adjust(hspace=0.35)

plt.tight_layout()

plt.rcParams['savefig.format'] = 'pdf'
plt.savefig(file + '_VQ', dpi=300)
plt.show()


#%% battery plot
SOC = df_out['Bats.E']/400e3*100
fig, ax = plt.subplots(figsize=(3.4, 2.4))  


ax.axhline(80, color='black', linestyle='--', linewidth=1.5, zorder=5, clip_on=False)
ax.axhline(10, color='black', linestyle='--', linewidth=1.5, zorder=5, clip_on=False)

sns.lineplot(x=df_out['t'], y=SOC, color=cbcolors[4], ax=ax,marker='v')
ax.set_xlabel('', fontsize=8)
ax.set_ylabel('SOC (\%)', fontsize=8)
ax.set_xticks(range(0, 24))
ax.set_xticklabels(labels_hours, fontsize=8, rotation=0)
ax.axhline(0, color='black', linewidth=1)

ax.text(24.5,-10,'Time (h)',fontsize=8)


plt.tight_layout()
plt.subplots_adjust(bottom=0.4)
ax.set_ylim(0, max(SOC) * 1.05)  
plt.rcParams['savefig.format'] = 'pdf'
plt.savefig(file + 'Bats_Energy', dpi=300)

plt.show()


#%%

# Operating costs comparasison

# Colores colorblind

cbcolors = sns.color_palette('colorblind')

# Datos
df = pd.DataFrame({
    'Case': ['Case 1 Winter', 'Case 1 Summer', 'Case 2 Winter', 'Case 2 Summer', 'Case 3 Winter', 'Case 3 Summer'],
    'Value': [26.57, 21.84, 12.41, -7.78, -3.78, -21.83]
})

# Extraer Case y Season
df['Case_Number'] = df['Case'].str.extract(r'(Case \d+)')  
df['Season'] = df['Case'].str.extract(r'(Winter|Summer)')  

# Pivotear para formato adecuado de barplot
df_pivot = df.pivot(index='Case_Number', columns='Season', values='Value')

# Colores personalizados: Azul para Winter, Anaranjado para Summer
colors = {'Winter': cbcolors[3], 'Summer': cbcolors[0]}

# Crear figura
fig, ax = plt.subplots(figsize=(3.4, 2.4))

# Graficar con colores específicos
df_pivot.plot(kind='bar', ax=ax, color=[colors['Winter'], colors['Summer']])

# Línea en 0
ax.axhline(0, color='black', linewidth=1)

# Etiquetas y formato
ax.set_xlabel('', fontsize=12)
ax.set_ylabel('Cost (€/day)', fontsize=8)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title='Season', loc='lower left')

plt.tight_layout()

# Guardar y mostrar
plt.rcParams['savefig.format'] = 'pdf'
plt.savefig('./results/CEDER/' + 'CaseCostsComparison', dpi=300)

plt.show()

