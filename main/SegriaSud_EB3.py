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
from Devices.Pumps import Pump, RealPump, ReversiblePump, ReversibleRealPump, RealPumpS
from Devices.Turbines import Turbine, DiscreteTurbine
from Devices.EB import EB
from Devices.SolarPV import SolarPV
from Devices.MainGrid import Grid
from Devices.Batteries import Battery, NewBattery
from pyomo.contrib.gdpopt.enumerate import GDP_Enumeration_Solver

# Import useful functions
from Utilities import clear_clc, model_to_file

#clean console and variable pane
# clear_clc() #consider removing if you are not working with Spyder

# model
m = pyo.ConcreteModel()

df_meteo_aug = pd.read_csv('data/meteo/LesPlanes_meteo_hour_aug.csv')
df_cons_aug = pd.read_csv('data/irrigation/EB5_irrigation_s.csv')
df_grid_aug = pd.read_csv('data/costs/PVPC_aug.csv')

df_meteo_jan = pd.read_csv('data/meteo/LesPlanes_meteo_hour_jan.csv')
df_cons_jan = pd.read_csv('data/irrigation/EB5_irrigation_w.csv')
df_grid_jan = pd.read_csv('data/costs/PVPC_jan.csv')

df_grid_jan['Excedentes_cut'] = df_grid_jan['Excedentes']*(1-0.3*df_grid_jan['Hour'].apply(lambda x: 1 if (x in [8,9,10,11,12,13,14,15,16]) else 0))
# df_grid_jan['Excedentes'] = df_grid_jan['Excedentes_cut']
df_grid_aug['Excedentes_cut'] = df_grid_aug['Excedentes']*(1-0.3*df_grid_aug['Hour'].apply(lambda x: 1 if (x in [8,9,10,11,12,13,14,15,16]) else 0))
# df_grid_aug['Excedentes'] = df_grid_aug['Excedentes_cut']


# time
T = 24
l_t = list(range(T))
m.tw = pyo.Set(initialize=l_t)
m.ts = pyo.Set(initialize=l_t)

# electricity cost
l_costw = df_grid_jan['PVPC']
l_excw = df_grid_jan['Excedentes']
m.costw = pyo.Param(m.tw, initialize=l_costw)
m.excw = pyo.Param(m.tw, initialize=l_excw)

l_costs = df_grid_aug['PVPC']
l_excs = df_grid_aug['Excedentes']
m.costs = pyo.Param(m.ts, initialize=l_costs)
m.excs = pyo.Param(m.ts, initialize=l_excs)

cost_new_pv = 0.00126*T/1000
cp_bat = 0.00171*T/1000
ce_bat = 0.00856*T/1000
# Costos en €/(kwh·h). Considerant vida de 20 anys, del llibre de Oriol Gomis i Paco Diaz trobem:
#   - Ce = 750 €/kWh (10 anys de vida) -> x2 = 1500 €/kWh -> /20/365/24 = 0.00856 €/(kWh·h)

# ===== Create the system =====
m.B4w = pyo.Block()
m.B5w = pyo.Block()
m.Irrigation1w = pyo.Block()
m.Pump1w = pyo.Block()
m.Pump2w = pyo.Block()
# m.Pump3w = pyo.Block()
m.Turb1w = pyo.Block()
m.Pipe1w = pyo.Block()
m.PVw = pyo.Block()
m.Gridw = pyo.Block()
m.EBgw = pyo.Block()
m.EBpvw = pyo.Block()
m.Batw = pyo.Block()

# m.ReservoirEbres = pyo.Block()
# m.Reservoir1s = pyo.Block()
# m.Irrigation1s = pyo.Block()
# m.Pump1s = pyo.Block()
# m.Pump2s = pyo.Block()
# m.Pump3s = pyo.Block()
# m.Turb1s = pyo.Block()
# m.Pipe1s = pyo.Block()
# m.PVs = pyo.Block()
# m.Grids = pyo.Block()
# m.EBgs = pyo.Block()
# m.EBpvs = pyo.Block()
# m.Bats = pyo.Block()


data_irr = {'Q':df_cons_jan['Qirr']/3600} # irrigation
Source(m.Irrigation1w, m.tw, data_irr, {})
# data_irr = {'Q':df_cons_aug['Qirr']/3600} # irrigation
# Source(m.Irrigation1s, m.ts, data_irr, {})

data_B4 = {'dt':3600, 'W0':230e5, 'Wmin':0.9*230e5, 'Wmax':26948500, 'zmin':420.19+(431.06-420.19)*0.9*230e3/269485, 'zmax':431.06}
init_B4 = {'Q':[0]*T, 'W':[230e5]*T}
Reservoir(m.B4w, m.tw, data_B4, init_B4)
# Reservoir(m.B4s, m.ts, data_B4, init_B4)
data_B5 = {'dt':3600, 'W0':124e3, 'Wmin':0.9*124e3, 'Wmax':185814, 'zmin':444.1+(450.5-444.1)*0.9*124e3/185814, 'zmax':450.5, 'WT_min':0.98*124e3, 'WT_max':1.02*124e3}
init_B5 = {'Q':[0]*T, 'W':[124e3]*T}
Reservoir(m.B5w, m.tw, data_B5, init_B5)
# Reservoir(m.B5s, m.ts, data_B5, init_B5)

data_c1 = {'K':6.1042, 'Qmax':1.2} # canal
init_c1 = {'Q':[0]*T, 'H':[108]*T, 'H0':[108]*T, 'zlow':[30]*T, 'zhigh':[138]*T}
Pipe(m.Pipe1w, m.tw, data_c1, init_c1)
# Pipe(m.Pipe1s, m.ts, data_c1, init_c1)

data_p = {'A':37.84, 'B':38.88, 'n_n':998, 'nmax':1, 'eff':0.861, 'eff_t':0.5, 'S':0.24*0.24*3.14, 'Qmin':0.5947, 'Qmax':1.2436, 'Qnom':0.4691, 'Pmax':160e3} # pumps (both equal)
init_p = {'Q':[0]*T, 'H':[37]*T, 'n':[0.99]*T, 'Pe':[0]*T}
RealPump(m.Pump1w, m.tw, data_p, init_p)
# RealPump(m.Pump1s, m.ts, data_p, init_p)
data_p['eff']=0.86
RealPump(m.Pump2w, m.tw, data_p, init_p)
# RealPump(m.Pump2s, m.ts, data_p, init_p)
# data_p['eff']=0.859
# RealPump(m.Pump3w, m.tw, data_p, init_p)
# RealPump(m.Pump3s, m.ts, data_p, init_p)

data_t = {'eff':0.5, 'Pmax':160e3}
init_t = {'Q':[0]*T, 'H':[37]*T, 'Pe':[0]*T}
Turbine(m.Turb1w, m.tw, data_t, init_t)
# Turbine(m.Turb1s, m.ts, data_t, init_t)

data_pv = {'Pinst':236e3, 'Pmax':236e3, 'forecast':df_meteo_jan['Irr']/1000, 'eff':0.98} # PV
SolarPV(m.PVw, m.tw, data_pv)
# data_pv = {'Pinst':236e3e3, 'Pmax':236e3, 'forecast':df_meteo_aug['Irr']/1000, 'eff':0.98} # PV
# SolarPV(m.PVs, m.ts, data_pv)

data_bat = {'dt':3600, 'E0':0.05, 'Emax':200e3, 'Pmax':200e3, 'SOCmin':0.2, 'SOCmax':1.0, 'eff_ch':0.8, 'eff_dc':0.8,'Einst':0.1, 'Pinst':0}
init_bat = {'E':[0]*T, 'P':[0]*T}
NewBattery(m.Batw, m.tw, data_bat, init_bat)
# NewBattery(m.Bats, m.ts, data_bat, init_bat)

Grid(m.Gridw, m.tw, {'Pmax':100e6}) # grid
# Grid(m.Grids, m.ts, {'Pmax':100e6}) # grid

EB(m.EBgw, m.tw)
# EB(m.EBgs, m.ts)
# EB(m.EBpvw, m.tw)
# EB(m.EBpvs, m.ts)


# def ConstraintPumpTurbw(m, t):
#     return m.Turb1w.Qin[t] * (m.Pump1w.PumpOn[t] + m.Pump2w.PumpOn[t]) == 0
# m.Turb1_c_PumpTurbw = pyo.Constraint(m.tw, rule=ConstraintPumpTurbw)
# def ConstraintPumpTurbs(m, t):
#     return m.Turb1s.Qin[t] * (m.Pump1s.PumpOn[t] + m.Pump2s.PumpOn[t]) == 0
# m.Turb1_c_PumpTurbs = pyo.Constraint(m.ts, rule=ConstraintPumpTurbs)

# def ConstraintPump2w(m, t):
#     return m.Pump2w.Qin[t] * m.Pump1w.PumpOn[t] == 0
# m.Turb1_c_Pump2w = pyo.Constraint(m.tw, rule=ConstraintPump2w)
# def ConstraintPump2s(m, t):
#     return m.Pump2s.Qin[t] * m.Pump1s.PumpOn[t] == 0
# m.Turb1_c_Pump2s = pyo.Constraint(m.ts, rule=ConstraintPump2s)


# def ConstraintBatEws(m):
#     return m.Batw.Edim == m.Bats.Edim
# m.c_BatEws = pyo.Constraint(rule=ConstraintBatEws)
# def ConstraintBatPws(m):
#     return m.Batw.Pdim == m.Bats.Pdim
# m.c_BatPws = pyo.Constraint(rule=ConstraintBatPws)


# m.Batw.Edim.fix(0)
# m.Bats.Edim.fix(0)
# m.Batw.Pdim.fix(0)
# m.Bats.Pdim.fix(0)

# m.Turb1s.Pdim.fix(160e3)
m.Turb1w.Pdim.fix(110e3)

m.PVw.Pdim.fix(0)


# Connections
m.p1r0w = Arc(ports=(m.Pump1w.port_Qin, m.B4w.port_Q), directed=True)
m.p1c1_Qw = Arc(ports=(m.Pump1w.port_Qout, m.Pipe1w.port_Q), directed=True)
m.p1c1_Hw = Arc(ports=(m.Pump1w.port_H, m.Pipe1w.port_H), directed=True)
m.p1ebw = Arc(ports=(m.Pump1w.port_P, m.EBgw.port_P), directed=True)

m.p2r0w = Arc(ports=(m.Pump2w.port_Qin, m.B4w.port_Q), directed=True)
m.p2c1_Qw = Arc(ports=(m.Pump2w.port_Qout, m.Pipe1w.port_Q), directed=True)
m.p2c1_Hw = Arc(ports=(m.Pump2w.port_H, m.Pipe1w.port_H), directed=True)
m.p2ebw = Arc(ports=(m.Pump2w.port_P, m.EBgw.port_P), directed=True) # pv node

# m.p3r0w = Arc(ports=(m.Pump3w.port_Qin, m.B4w.port_Q), directed=True)
# m.p3c1_Qw = Arc(ports=(m.Pump3w.port_Qout, m.Pipe1w.port_Q), directed=True)
# m.p3c1_Hw = Arc(ports=(m.Pump3w.port_H, m.Pipe1w.port_H), directed=True)
# m.p3ebw = Arc(ports=(m.Pump3w.port_P, m.EBgw.port_P), directed=True) # pv node

m.t1r0w = Arc(ports=(m.Turb1w.port_Qout, m.B4w.port_Q), directed=True)
m.t1c1_Qw = Arc(ports=(m.Turb1w.port_Qin, m.Pipe1w.port_Q), directed=True)
m.t1c1_Hw = Arc(ports=(m.Turb1w.port_H, m.Pipe1w.port_H), directed=True)
m.t1ebw = Arc(ports=(m.Turb1w.port_P, m.EBgw.port_P), directed=True)

m.c1r1_Qw = Arc(ports=(m.Pipe1w.port_Q, m.B5w.port_Q), directed=True)
m.c1r1_zw = Arc(ports=(m.B5w.port_z, m.Pipe1w.port_zhigh), directed=True)
m.c1r0_zw = Arc(ports=(m.B4w.port_z, m.Pipe1w.port_zlow), directed=True)

m.r1i1w = Arc(ports=(m.Irrigation1w.port_Qin, m.B5w.port_Q), directed=True)

m.gridebw = Arc(ports=(m.Gridw.port_P, m.EBgw.port_P), directed=True)
m.pvebw = Arc(ports=(m.PVw.port_P, m.EBgw.port_P), directed=True) # pv node
m.batebw = Arc(ports=(m.Batw.port_P, m.EBgw.port_P), directed=True) # pv node

# Connections
# m.p1r0s = Arc(ports=(m.Pump1s.port_Qin, m.B4s.port_Q), directed=True)
# m.p1c1_Qs = Arc(ports=(m.Pump1s.port_Qout, m.Pipe1s.port_Q), directed=True)
# m.p1c1_Hs = Arc(ports=(m.Pump1s.port_H, m.Pipe1s.port_H), directed=True)
# m.p1ebs = Arc(ports=(m.Pump1s.port_P, m.EBgs.port_P), directed=True)

# m.p2r0s = Arc(ports=(m.Pump2s.port_Qin, m.B4s.port_Q), directed=True)
# m.p2c1_Qs = Arc(ports=(m.Pump2s.port_Qout, m.Pipe1s.port_Q), directed=True)
# m.p2c1_Hs = Arc(ports=(m.Pump2s.port_H, m.Pipe1s.port_H), directed=True)
# m.p2ebs = Arc(ports=(m.Pump2s.port_P, m.EBgs.port_P), directed=True) # pv node

# m.t1r0s = Arc(ports=(m.Turb1s.port_Qout, m.B4s.port_Q), directed=True)
# m.t1c1_Qs = Arc(ports=(m.Turb1s.port_Qin, m.Pipe1s.port_Q), directed=True)
# m.t1c1_Hs = Arc(ports=(m.Turb1s.port_H, m.Pipe1s.port_H), directed=True)
# m.t1ebs = Arc(ports=(m.Turb1s.port_P, m.EBgs.port_P), directed=True)

# m.c1r1_Qs = Arc(ports=(m.Pipe1s.port_Q, m.B5s.port_Q), directed=True)
# m.c1r1_zs = Arc(ports=(m.B5s.port_z, m.Pipe1s.port_zhigh), directed=True)
# m.c1r0_zs = Arc(ports=(m.B4s.port_z, m.Pipe1s.port_zlow), directed=True)

# m.r1i1s = Arc(ports=(m.Irrigation1s.port_Qin, m.B5s.port_Q), directed=True)

# m.gridebs = Arc(ports=(m.Grids.port_P, m.EBgs.port_P), directed=True)
# m.pvebs = Arc(ports=(m.PVs.port_P, m.EBgs.port_P), directed=True) # pv node
# m.batebs = Arc(ports=(m.Bats.port_P, m.EBgs.port_P), directed=True) # pv node

pyo.TransformationFactory("network.expand_arcs").apply_to(m) # apply arcs to model


#%% RUN THE OPTIMIZATION
from pyomo.environ import value
import os
import time

# Objective function
def obj_fun(m):
# 	return sum((m.Grid.Pbuy[t]*m.cost[t]/1e6 - m.Grid.Psell[t]*m.exc[t]/1e6) + 0*1/1e6*((m.PV.Pinst+m.PV.Pdim)*m.PV.forecast[t]*m.PV.eff + m.PV.P[t]) for t in l_t ) #+ (m.Bat.Pdim*cp_bat + m.Bat.Edim*ce_bat)/365/20#+ m.PV.Pdim*cost_new_pv
# 	return sum(( m.Gridw.Pbuy[t]*m.costw[t]/1e6 - m.Gridw.Psell[t]*m.excw[t]/1e6 + 
#              m.Grids.Pbuy[t]*m.costs[t]/1e6 - m.Grids.Psell[t]*m.excs[t]/1e6)/2  for t in l_t ) + (m.Batw.Pdim*cp_bat + m.Batw.Edim*ce_bat) #+ m.PV.Pdim*cost_new_pv
	return sum( m.Gridw.Pbuy[t]*m.costw[t]/1e6 - m.Gridw.Psell[t]*m.excw[t]/1e6 for t in l_t ) + (m.Batw.Pdim*cp_bat + m.Batw.Edim*ce_bat) #+ m.PV.Pdim*cost_new_pv
m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)

instance = m.create_instance()

start_time = time.time()


# os.environ['NEOS_EMAIL'] = 'sergi.costa.dilme@upc.edu'
# solver_manager = pyo.SolverManagerFactory('neos')
# # results = solver_manager.solve(instance, solver="knitro")
# results = solver_manager.solve(instance, solver="couenne")
# # results = solver_manager.solve(instance, solver="ipopt", options_string='max_iter=10000000')
# # results = solver_manager.solve(instance, solver="bonmin")
# # results = solver_manager.solve(instance, solver="minlp")
# results.write()

with open("couenne.opt", "w") as file:
    file.write('''time_limit 100000
                convexification_cuts 2
                convexification_points 2
                delete_redundant yes
                use_quadratic no
                feas_tolerance 1e-1
                ''')
solver = pyo.SolverFactory('asl:couenne')
results = solver.solve(instance, tee=True)
results.write()
os.remove('couenne.opt') #Delete options

# solver = pyo.SolverFactory('asl:SCIP')
# results = solver.solve(instance, tee=True)
# results.write()

# instance = m.create_instance()
# solver = pyo.SolverFactory('ipopt')
# solver.options['max_iter'] = 1000000
# results = solver.solve(instance, tee=True)

exec_time = time.time() - start_time

#%% GET RESULTS
from Utilities import get_results

file = './results/EB3/Preliminary'
df_out, df_param, df_size = get_results(file=file+'/EB3', instance=instance, results=results, l_t=l_t, exec_time=exec_time)

#%% PLOTS
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
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

file = 'Preliminary/EB3'

df_meteo_aug = pd.read_csv('data/meteo/LesPlanes_meteo_hour_aug.csv')
df_cons_aug = pd.read_csv('data/irrigation/EB5_irrigation_s.csv')
df_grid_aug = pd.read_csv('data/costs/PVPC_aug.csv')
df_meteo_jan = pd.read_csv('data/meteo/LesPlanes_meteo_hour_jan.csv')
df_cons_jan = pd.read_csv('data/irrigation/EB5_irrigation_w.csv')
df_grid_jan = pd.read_csv('data/costs/PVPC_jan.csv')



df_param = pd.read_csv('results/EB3/'+file+'_param.csv')
w_lim = df_param['Reservoir1w.Wmin'][0]

df = pd.read_csv('results/EB3/'+file+'.csv')
df['PVw.Pf'] = -df_meteo_jan['Irr']/1000*215.28e3*0.98
# df['PVs.Pf'] = -df_meteo_aug['Irr']/1000*215.28e3*0.98


if 'Turb1w.Qout' in df.columns:
    df['Rev1w.Qout'] = df['Pump1w.Qout'] - df['Turb1w.Qout']
    df['Rev1w.Pe'] = df['Pump1w.Pe'] + df['Turb1w.Pe']
    # df['Rev1s.Qout'] = df['Pump1s.Qout'] - df['Turb1s.Qout']
    # df['Rev1s.Pe'] = df['Pump1s.Pe'] + df['Turb1s.Pe']
else:
    df['Rev1w.Qout'] = df['Pump1w.Qout']
    df['Rev1w.Pe'] = df['Pump1w.Pe']
    # df['Rev1s.Qout'] = df['Pump1s.Qout']
    # df['Rev1s.Pe'] = df['Pump1s.Pe']

df['Irrigation1w.Qin'] = -df['Irrigation1w.Qout'] # no se per que passa pero Qin a vegades es reseteja a 0. La resta tot be
# df['Irrigation1s.Qin'] = -df['Irrigation1s.Qout']

for c in df.columns:
    df[c] = df[c].apply(lambda x: 0 if (x>-1e-5 and x<1e-5) else x)

df_W = df[[col for col in df.columns if 'w.' in col]]
df_S = df[[col for col in df.columns if 's.' in col]]
df_W['t'] = df_W.index
df_S['t'] = df_S.index
df_W['PVPC'] = df_grid_jan['PVPC']
# df_S['PVPC'] = df_grid_aug['PVPC']
df_W['Excedentes'] = df_grid_jan['Excedentes']
# df_S['Excedentes'] = df_grid_aug['Excedentes']

df_W.columns = [col.replace('w.','.') for col in df_W.columns]
# df_S.columns = [col.replace('s.','.') for col in df_W.columns]

#%%
i=0
# season=['S','W']
season=['W']
# for df in [df_S, df_W]:
for df in [df_W]:
    
    fig = plt.figure(figsize=(3.4, 1.8))
    plt.rcParams['axes.spines.right'] = True
    ax1 = fig.add_subplot(1,1,1)
    df.apply(lambda x: x/1000).plot(y=['PV.Pf'], kind='bar', ax=ax1, stacked=False, ylabel='P (kW)', 
                                color='#C0E3C0', alpha=1, edgecolor=None)
    df.apply(lambda x: x/1000).plot(y=['PV.P','Pump2.Pe','Rev1.Pe'], kind='bar', stacked=True, ax=ax1, ylabel='P (kW)',
                                color=[cbcolors[2],cbcolors[0],cbcolors[1]], edgecolor=None)
    ax1.axhline(0,color='k')
    ax1.set_ylim(-200,200)
    bars = ax1.patches
    patterns =(None, None, None, None)
    hatches = [p for p in patterns for i in range(len(df))]
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax1.legend(['_','$\hat{P}_{PV}$','$P_{PV}$','$P_{p,PV}$','$P_{p,g}$'],loc='upper center', bbox_to_anchor=(0.5, -0.18),
          fancybox=False, shadow=False, ncol=4)
    
    ax2 = plt.twinx()
    ax2.set_ylim(45,300)
    sns.lineplot(df,x='t' ,y='PVPC', ax=ax2, color='tab:red')#, label='Buy')
    sns.lineplot(df,x='t' ,y='Excedentes', ax=ax2, color='tab:red', linestyle='dashed')#, label='Buy')
    ax2.set_xticks(range(24), labels=labels_hours, rotation=90)
    ax2.text(24.5,0.1,'Time (h)')
    
    plt.ylabel('Price (€/MWh)')
    plt.xlabel('Hour')
    # plt.title('Power consumption')
    plt.subplots_adjust(left=0.17, right=0.82, top=0.97, bottom=0.3)
    plt.show()
    
    plt.rcParams['savefig.format']='pdf'
    plt.savefig('results/ISGT/' + file + season[i] + '_P', dpi=300)
    plt.rcParams['savefig.format']='svg'
    plt.savefig('results/ISGT/' + file + season[i] + '_P', dpi=300)
    
    
    
    fig = plt.figure(figsize=(3.4, 1.9))
    plt.rcParams['axes.spines.right'] = False
    gs = gridspec.GridSpec(2,1,height_ratios=[2,1])
    
    ax1 = fig.add_subplot(gs[0])
    ax1.set_ylim(-0.12,0.12)
    df.plot(y=['Pump2.Qout','Irrigation1.Qin','Rev1.Qout'], kind='bar', stacked=True, ax=ax1, ylabel='Q (m$^3$/s)',
            color=[cbcolors[0],cbcolors[2],cbcolors[1]], edgecolor=None)
    bars = ax1.patches
    patterns =(None,None,None)
    hatches = [p for p in patterns for i in range(len(df))]
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax1.legend(['$Q_{p,PV}$', '$Q_{irr}$', '$Q_{p,g}$'], ncols=3, loc='upper center', bbox_to_anchor=(0.5, -0.88),
          fancybox=False, shadow=False)
    ax1.axhline(0,color='k')
    ax1.set_xticks(range(24), labels=['']*24, rotation=90)
    ax1.tick_params(axis='x', labelbottom='off')
    
    ax2 = fig.add_subplot(gs[1],sharex=ax1)
    ax2.figsize=(3.4,2)
    ax2.set_ylim(7500,14000)
    ax2.set_yticks([8000,10000,13000])
    ax2.axhline(w_lim,color='#AFAFAF', alpha=1, linewidth=1)
    ax2.axhline(13000,color='#AFAFAF', alpha=1, linewidth=1)
    ax2.axhline(10000*0.95, color='#AFAFAF', linestyle='--', alpha=1, linewidth=1)
    ax2.axhline(10000*1.05, color='#AFAFAF', linestyle='--', alpha=1, linewidth=1)
    sns.lineplot(df, x='t', y='Reservoir1.W', ax=ax2, color='tab:red', linewidth=1.5)
    ax2.set_xticks(range(24), labels=labels_hours, rotation=90)
    ax2.text(1.02,-0.45,'Time (h)', transform=ax2.transAxes)
    
    plt.ylabel('$W_{R1}$ (m$^3$)')
    plt.xlabel(None)
    plt.setp(ax1.get_xticklabels(), visible=False)
    # plt.xticks(range(24),labels=labels_hours,rotation=90)
    plt.subplots_adjust(left=0.18, right=0.84, top=0.97, bottom=0.26)
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
# plt.legend(labels=['Winter','_','Spring','_','Summer','_','Fall','_'], ncol=1, loc='upper right')
# plt.xlabel('Hour')
# plt.ylabel('Irrigation demand (m$^3$/h)')
# plt.xticks(range(24),labels=labels_hours, rotation=90)
# plt.xlim([0,24])
# plt.ylim([0,250])
# plt.tight_layout()
# plt.subplots_adjust(left=0.15, right=1, top=0.95, bottom=0.18)
# plt.show()
# plt.rcParams['savefig.format']='pdf'
# plt.savefig('results/ISGT/Irrigation_season', dpi=300)

# df['Winter'] = df['date'].apply(lambda x: 0 if (x.month in [3,4,5,6,7,8]) else 1)
# fig = plt.figure(figsize=(3.4, 1.1))
# sns.lineplot(df, x='hour', y='Qreg', style='Winter', hue='Winter', palette=[cbcolors[1],cbcolors[0]], errorbar=('ci',90))
# plt.legend(labels=['S','_','W','_'], ncol=1)
# plt.text(24.5,-50,'Time (h)')
# plt.xlabel(None)
# plt.ylabel('Demand (m$^3$/h)')
# plt.xticks(range(24),labels=labels_hours, rotation=90)
# plt.yticks([0,100,200])
# # plt.ylim([0,200])
# plt.xlim([0,24])
# plt.tight_layout()
# plt.subplots_adjust(left=0.15, right=0.84, top=0.92, bottom=0.22)
# plt.show()
# plt.rcParams['savefig.format']='pdf'
# plt.savefig('results/ISGT/Irrigation_WS', dpi=300)
# plt.rcParams['savefig.format']='svg'
# plt.savefig('results/ISGT/Irrigation_WS', dpi=300)

#%% (PARALLEL PUMP)

i=0
season=['S','W']
for df in [df_S, df_W]:
     
    fig = plt.figure(figsize=(3.4, 1.8))
    plt.rcParams['axes.spines.right'] = True
    ax1 = fig.add_subplot(1,1,1)
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
    ax1.legend(['_','$\hat{P}_{PV}$','$P_{PV}$','$P_{p}$'],loc='upper center', bbox_to_anchor=(0.5, -0.18),
          fancybox=False, shadow=False, ncol=3)
    
    ax2 = plt.twinx()
    ax2.set_ylim(45,300)
    sns.lineplot(df,x='t' ,y='PVPC', ax=ax2, color='tab:red')#, label='Buy')
    sns.lineplot(df,x='t' ,y='Excedentes', ax=ax2, color='tab:red', linestyle='dashed')#, label='Buy')
    ax2.set_xticks(range(24), labels=labels_hours, rotation=90)
    ax2.text(24.5,0.1,'Time (h)')
    
    plt.ylabel('Price (€/MWh)')
    plt.xlabel('Hour')
    # plt.title('Power consumption')
    plt.subplots_adjust(left=0.17, right=0.82, top=0.97, bottom=0.3)
    plt.show()
    
    plt.rcParams['savefig.format']='pdf'
    plt.savefig('results/ISGT/' + file + season[i] + '_P', dpi=300)
    plt.rcParams['savefig.format']='svg'
    plt.savefig('results/ISGT/' + file + season[i] + '_P', dpi=300)
    
    
    
    fig = plt.figure(figsize=(3.4, 1.9))
    plt.rcParams['axes.spines.right'] = False
    gs = gridspec.GridSpec(2,1,height_ratios=[2,1])
    
    ax1 = fig.add_subplot(gs[0])
    ax1.set_ylim(-0.12,0.12)
    df.plot(y=['Irrigation1.Qin','Rev1.Qout'], kind='bar', stacked=True, ax=ax1, ylabel='Q (m$^3$/s)',
            color=[cbcolors[2],cbcolors[1]], edgecolor=None)
    bars = ax1.patches
    patterns =(None,None,None)
    hatches = [p for p in patterns for i in range(len(df))]
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax1.legend(['$Q_{irr}$','$Q_{p}$'], ncols=2, loc='upper center', bbox_to_anchor=(0.5, -0.88),
          fancybox=False, shadow=False)
    ax1.axhline(0,color='k')
    ax1.set_xticks(range(24), labels=['']*24, rotation=90)
    ax1.tick_params(axis='x', labelbottom='off')
    
    ax2 = fig.add_subplot(gs[1],sharex=ax1)
    ax2.figsize=(3.4,2)
    ax2.set_ylim(7500,14000)
    ax2.set_yticks([8000,10000,13000])
    ax2.axhline(w_lim,color='#AFAFAF', alpha=1, linewidth=1)
    ax2.axhline(13000,color='#AFAFAF', alpha=1, linewidth=1)
    ax2.axhline(10000*0.95, color='#AFAFAF', linestyle='--', alpha=1, linewidth=1)
    ax2.axhline(10000*1.05, color='#AFAFAF', linestyle='--', alpha=1, linewidth=1)
    sns.lineplot(df, x='t', y='Reservoir1.W', ax=ax2, color='tab:red', linewidth=1.5)
    ax2.set_xticks(range(24), labels=labels_hours, rotation=90)
    ax2.text(1.02,-0.45,'Time (h)', transform=ax2.transAxes)
    
    plt.ylabel('$W_{R1}$ (m$^3$)')
    plt.xlabel(None)
    plt.setp(ax1.get_xticklabels(), visible=False)
    # plt.xticks(range(24),labels=labels_hours,rotation=90)
    plt.subplots_adjust(left=0.18, right=0.84, top=0.97, bottom=0.26)
    plt.show()

    plt.rcParams['savefig.format']='pdf'
    plt.savefig('results/ISGT/' + file + season[i] + '_Q', dpi=300)
    plt.rcParams['savefig.format']='svg'
    plt.savefig('results/ISGT/' + file + season[i] + '_Q', dpi=300)
    
    
    i=i+1


    



#%%
logging.INFO = 2000
log_infeasible_constraints(instance)

# with open('eq', 'w') as f:
#     instance.pprint(f)