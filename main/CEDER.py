"""
    AGISTIN - EXAMPLE CEDER
    
    3 Reservoirs wihout irrigation connected to a battery and a PV system.
    Giving services to loads.
    
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

#%% Data analysis
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

#%%

date = '2023-07-02'
ho = 0
hf = 24
df_meteo = pd.DataFrame()
# df_meteo ['Irr'] = df_data['Irr'][(df_data.index == date) & (df_data['Hour'] >= ho) & (df_data['Hour'] <= hf)]
# df_load = - df_data['CT_Edificio1'][(df_data.index == date) & (df_data['Hour'] >= ho) & (df_data['Hour'] <= hf)]
# df_grid = pd.read_csv('data/costs/PVPC_jul.csv').iloc[ho:(hf)].reset_index(drop=True)
df_grid =  pd.read_csv('data/costs/PVPC_jan.csv').iloc[ho:(hf)].reset_index(drop=True)





# df_concat = pd.concat([summer_day_mean, winter_day_mean])
# df_meteo = pd.DataFrame()

# df_meteo['Irr'] = df_concat['Irr']
# df_load = df_concat['CT_Edificio1']
# df_grid = pd.concat([df_grid_Jul,df_grid_Jan])
df_meteo['Irr']  = winter_day_mean['Irr']
df_load = - winter_day_mean['CT_Edificio1']

df_meteo.reset_index(drop=True, inplace=True)
df_load.reset_index(drop=True, inplace=True)
df_grid.reset_index(drop=True, inplace=True)

# time
T = 24
l_t = list(range(T))
# m.tw = pyo.Set(initialize=l_t)
m.ts = pyo.Set(initialize=l_t)

# electricity cost
# demands = - summer_day_mean['CT_Edificio1']
# demandw = - winter_day_mean['CT_Edificio1']
demands = df_load

l_costs = df_grid['PVPC']
# l_costw = df_grid_Jan['PVPC']

l_excs = df_grid['Excedentes']
# l_excw = df_grid_Jan['Excedentes']

# l_cost = [650,600,700,800,1000]
# l_exc = [x/3 for x in [650,600,700,800,1000]]

# m.costw = pyo.Param(m.tw, initialize=l_costw)
# m.excw = pyo.Param(m.tw, initialize=l_excw)
m.costs = pyo.Param(m.ts, initialize=l_costs)
m.excs = pyo.Param(m.ts, initialize=l_excs)

cost_new_pv = 0.00126*T/1000
cp_bat = 0.00171*T/1000
ce_bat = 0.00856*T/1000

# Costos en €/(kwh·h). Considerant vida de 20 anys, del llibre de Oriol Gomis i Paco Diaz trobem:
#   - Ce = 750 €/kWh (10 anys de vida) -> x2 = 1500 €/kWh -> /20/365/24 = 0.00856 €/(kWh·h)


# ===== Create the system =====
# m.Reservoir0w = pyo.Block()
# m.Reservoir1w = pyo.Block()
# m.Reservoir2w = pyo.Block()
# m.Pump1w = pyo.Block()
# m.Pump2w = pyo.Block()
# m.Turb1w = pyo.Block()
# m.Pipe01w = pyo.Block()
# m.Pipe02w = pyo.Block()
# m.VarSourcew = pyo.Block()
# m.Pipe01Tw = pyo.Block()
# m.PVw = pyo.Block()
# m.Gridw = pyo.Block()
# m.EBw = pyo.Block()
# m.Batw = pyo.Block()
# m.Loadw = pyo.Block()

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
m.Bats = pyo.Block()
m.Loads = pyo.Block()




data_R0 = {'dt':3600, 'W0':2.5e3, 'Wmin':0, 'Wmax':2.5e3, 'zmin':1, 'zmax':4}
init_R0 = {'Q':[0]*T, 'W':[2.5e3]*T}
# Reservoir(m.Reservoir0w, m.tw, data_R0, init_R0)
Reservoir(m.Reservoir0s, m.ts, data_R0, init_R0)


data_R1 = {'dt':3600, 'W0':0e3, 'Wmin':0e3, 'Wmax':1.5e3, 'zmin':67, 'zmax':69}
init_R1 = {'Q':[0]*T, 'W':[0.5e3]*T}
Reservoir(m.Reservoir1s, m.ts, data_R1, init_R1)
# Reservoir(m.Reservoir1w, m.tw, data_R1, init_R1)

data_R2 = {'dt':3600, 'W0':0e3, 'Wmin':0e3, 'Wmax':0.5e3, 'zmin':77, 'zmax':79}
init_R2 = {'Q':[0]*T, 'W':[0.25e3]*T}
Reservoir(m.Reservoir2s, m.ts, data_R2, init_R2)
# Reservoir(m.Reservoir2w, m.tw, data_R2, init_R2)

data_c1 = {'K':300, 'Qmax':0.5} # canal
init_c1 = {'Q':[0]*T, 'H':[77]*T, 'H0':[77]*T, 'zlow':[0]*T, 'zhigh':[77]*T}
Pipe(m.Pipe01s, m.ts, data_c1, init_c1)
Pipe(m.Pipe01Ts, m.ts, data_c1, init_c1)
# Pipe(m.Pipe01w, m.tw, data_c1, init_c1)
# Pipe(m.Pipe01Tw, m.tw, data_c1, init_c1)


data_c2 = {'K':300, 'Qmax':0.5} # canal
init_c2 = {'Q':[0]*T, 'H':[67]*T, 'H0':[67]*T, 'zlow':[2]*T, 'zhigh':[67]*T}
Pipe(m.Pipe02s, m.ts, data_c2, init_c2)
# Pipe(m.Pipe02w, m.tw, data_c2, init_c2)


data_Source = {'Qmax':1}
init_Source = {'Q':[0.0]*T,'Qin':[0]*T,'Qout':[0]*T}
VarSource(m.VarSources, m.ts, data_Source, init_Source )
# VarSource(m.VarSourcew, m.tw, data_Source, init_Source )


data_p1 = {'A':121.54, 'B':0.0003, 'n_n':2900, 'eff':0.8, 'S':0.1*0.1*3.14,'Pn':7.5e3, 'Npumps':4,'Qnom':0.05,'Qmax':2,'Qmin':0,'Pmax':30e3} # pumps (both equal)
init_p1 = {'Q':[0]*T, 'H':[67]*T, 'n':[2900]*T, 'Pe':[7.5e3]*T}
DiscretePump(m.Pump1s, m.ts, data_p1, init_p1)
DiscretePump(m.Pump2s, m.ts, data_p1, init_p1)
# DiscretePump(m.Pump1w, m.tw, data_p1, init_p1)
# DiscretePump(m.Pump2w, m.tw, data_p1, init_p1)


data_t = {'eff':0.8, 'Pmax':40e3}
init_t = {'Q':[0]*T, 'H':[67]*T, 'Pe':[-40e3*0.8]*T}
Turbine(m.Turb1s, m.ts, data_t, init_t)
# Turbine(m.Turb1w, m.tw, data_t, init_t)

# data_pvs = {'Pinst':16e3, 'Pmax':66e3, 'forecast':summer_day_mean['Irr']/1000, 'eff':0.98} # PV
# data_pvw = {'Pinst':16e3, 'Pmax':66e3, 'forecast':winter_day_mean['Irr']/1000, 'eff':0.98} # PV
data_pvs = {'Pinst':16e3, 'Pmax':66e3, 'forecast':df_meteo['Irr']/1000, 'eff':0.98} # PV

SolarPV(m.PVs, m.ts, data_pvs)
# SolarPV(m.PVw, m.tw, data_pvw)

data_bat = {'dt':3600, 'E0':40e3, 'Emax':800e3, 'Pmax':180e3, 'SOCmin':0.1, 'SOCmax':0.80, 'eff_ch':0.9, 'eff_dc':0.9,'Einst':400e3, 'Pinst':90e3}
init_bat = {'E':[200e3]*T, 'P':[0]*T}
NewBattery(m.Bats, m.ts, data_bat, init_bat)
# NewBattery(m.Batw, m.tw, data_bat, init_bat)

Grid(m.Grids, m.ts, {'Pmax':100e3}) # grid
# Grid(m.Gridw, m.tw, {'Pmax':100e3}) # grid


data_loads = {'P':demands} # Electrical Demand
# data_loadw = {'P':demandw} # Electrical Demand

Load(m.Loads, m.ts, data_loads, {})
# Load(m.Loadw, m.tw, data_loadw, {})


# EB(m.EBw, m.tw)
EB(m.EBs, m.ts)


# def ConstraintPumpOnOffs (m,ts):
#     return m.Pump1s.Pe[ts] * m.Pump2s.Pe[ts] == 0
# m.p1p2s = pyo.Constraint(m.ts, rule = ConstraintPumpOnOffs)

# def ConstraintPumpOnOffw (m,tw):
#     return m.Pump1w.Pe[tw] * m.Pump2w.Pe[tw] == 0
# m.p1p2w = pyo.Constraint(m.tw, rule = ConstraintPumpOnOffw)


# def ConstraintBatEws(m):
#     return m.Batw.Edim == m.Bats.Edim
# m.Reservoir1_c_BatEws = pyo.Constraint(rule=ConstraintBatEws)

# def ConstraintBatPws(m):
#     return m.Batw.Pdim == m.Bats.Pdim
# m.Reservoir1_c_BatPws = pyo.Constraint(rule=ConstraintBatPws)

# def ConstraintPVsw(m):
#     return m.PVw.Pdim == m.PVs.Pdim
# m.cPVsw = pyo.Constraint(rule=ConstraintPVsw)


# def ConstraintWinterSummerMin(m,t):
#     return (m.Reservoir1.W[23] + m.Reservoir2.W[23]) <= 100
# m.r1r2 = pyo.Constraint(m.t, rule = ConstraintWinterSummerMin)



#Energy constraint

# def ConstraintW1min(m):
#     return m.Reservoir0.W[T-1]*m.Reservoir0.z[T-1] + m.Reservoir1.W[T-1]*m.Reservoir1.z[T-1] + m.Reservoir2.W[T-1]*m.Reservoir2.z[T-1]  >= (m.Reservoir0.W0*m.Reservoir0.z[0] + m.Reservoir1.W0*m.Reservoir1.z[0] + m.Reservoir2.W0*m.Reservoir2.z[0])*0.9
# m.Reservoir1_c_W1min = pyo.Constraint(rule=ConstraintW1min)


# def ConstraintW1max(m):
#     return m.Reservoir0.W[T-1]*m.Reservoir0.z[T-1] + m.Reservoir1.W[T-1]*m.Reservoir1.z[T-1] + m.Reservoir2.W[T-1]*m.Reservoir2.z[T-1]  >= (m.Reservoir0.W0*m.Reservoir0.z[0] + m.Reservoir1.W0*m.Reservoir1.z[0] + m.Reservoir2.W0*m.Reservoir2.z[0])*1.1
# m.Reservoir1_c_W1max = pyo.Constraint(rule=ConstraintW1max)



# Connections

#Pump1 to R1 and Pipe12 and EB
# m.p1r0w = Arc(ports=(m.Pump1w.port_Qin, m.Reservoir0w.port_Q), directed=True)
# m.p1ebw = Arc(ports=(m.Pump1w.port_P, m.EBw.port_P), directed=True)
# m.p1h1_Qw = Arc(ports=(m.Pump1w.port_Qout, m.Pipe01w.port_Q), directed=True)
# m.p1h1_Hw = Arc(ports=(m.Pump1w.port_H, m.Pipe01w.port_H), directed=True)

m.p1r0s = Arc(ports=(m.Pump1s.port_Qin, m.Reservoir0s.port_Q), directed=True)
m.p1ebs = Arc(ports=(m.Pump1s.port_P, m.EBs.port_P), directed=True)
m.p1h1_Qs = Arc(ports=(m.Pump1s.port_Qout, m.Pipe01s.port_Q), directed=True)
m.p1h1_Hs = Arc(ports=(m.Pump1s.port_H, m.Pipe01s.port_H), directed=True)



m.p2r0s = Arc(ports=(m.Pump2s.port_Qin, m.Reservoir0s.port_Q), directed=True)
m.p2ebs = Arc(ports=(m.Pump2s.port_P, m.EBs.port_P), directed=True)
m.p2h2_Qs = Arc(ports=(m.Pump2s.port_Qout, m.Pipe02s.port_Q), directed=True)
m.p2h2_Hs = Arc(ports=(m.Pump2s.port_H, m.Pipe02s.port_H), directed=True)

# m.p2r0w = Arc(ports=(m.Pump2w.port_Qin, m.Reservoir0w.port_Q), directed=True)
# m.p2ebw = Arc(ports=(m.Pump2w.port_P, m.EBw.port_P), directed=True)
# m.p2h2_Qw = Arc(ports=(m.Pump2w.port_Qout, m.Pipe02w.port_Q), directed=True)
# m.p2h2_Hw = Arc(ports=(m.Pump2w.port_H, m.Pipe02w.port_H), directed=True)



# m.p1h1_Q = Arc(ports=(m.Pump1.port_Qout, m.HydroSwitch.port_Qout), directed=True)
# m.p1h1_H = Arc(ports=(m.Pump1.port_H, m.HydroSwitch.port_Hout), directed=True)

# m.HSp01 = Arc(ports=(m.HydroSwitch.port_Qin0, m.Pipe01.port_Q), directed=True)
# m.HSp02 = Arc(ports=(m.HydroSwitch.port_Qin1, m.Pipe02.port_Q), directed=True)
# m.HSp01_H = Arc(ports=(m.HydroSwitch.port_Hin0, m.Pipe01.port_H), directed=True)
# m.HSp02_H = Arc(ports=(m.HydroSwitch.port_Hin1, m.Pipe02.port_H), directed=True)

#Turbine to Reservoir1
m.t1r0s = Arc(ports=(m.Turb1s.port_Qout, m.Reservoir0s.port_Q), directed=True)
m.t1c1_Qs = Arc(ports=(m.Turb1s.port_Qin, m.Pipe01Ts.port_Q), directed=True)
m.t1c1_Hs = Arc(ports=(m.Turb1s.port_H, m.Pipe01Ts.port_H), directed=True)
m.t1ebs = Arc(ports=(m.Turb1s.port_P, m.EBs.port_P), directed=True)

# m.t1r0w = Arc(ports=(m.Turb1w.port_Qout, m.Reservoir0w.port_Q), directed=True)
# m.t1c1_Qw = Arc(ports=(m.Turb1w.port_Qin, m.Pipe01Tw.port_Q), directed=True)
# m.t1c1_Hw = Arc(ports=(m.Turb1w.port_H, m.Pipe01Tw.port_H), directed=True)
# m.t1ebw = Arc(ports=(m.Turb1w.port_P, m.EBw.port_P), directed=True)


#Pipe 01T
# m.c01tr1w = Arc(ports=(m.Pipe01Tw.port_Q, m.Reservoir1w.port_Q),directed=True)
# m.c01tr1_zw = Arc(ports=(m.Reservoir1w.port_z, m.Pipe01Tw.port_zhigh), directed=True)
# m.c01tr0_zw = Arc(ports=(m.Reservoir0w.port_z, m.Pipe01Tw.port_zlow), directed=True)

m.c01tr1s = Arc(ports=(m.Pipe01Ts.port_Q, m.Reservoir1s.port_Q),directed=True)
m.c01tr1_zs = Arc(ports=(m.Reservoir1s.port_z, m.Pipe01Ts.port_zhigh), directed=True)
m.c01tr0_zs = Arc(ports=(m.Reservoir0s.port_z, m.Pipe01Ts.port_zlow), directed=True)


#Pipe 01 to Reservoir 1
m.c01r1_Qs = Arc(ports=(m.Pipe01s.port_Q, m.Reservoir1s.port_Q), directed=True)
m.c01r1_zs = Arc(ports=(m.Reservoir1s.port_z, m.Pipe01s.port_zhigh), directed=True)
m.c01r0_zs = Arc(ports=(m.Reservoir0s.port_z, m.Pipe01s.port_zlow), directed=True)

# m.c01r1_Qw = Arc(ports=(m.Pipe01w.port_Q, m.Reservoir1w.port_Q), directed=True)
# m.c01r1_zw = Arc(ports=(m.Reservoir1w.port_z, m.Pipe01w.port_zhigh), directed=True)
# m.c01r0_zw = Arc(ports=(m.Reservoir0w.port_z, m.Pipe01w.port_zlow), directed=True)


#Pipe 02 to R2 and R0
m.c02r2_Qs = Arc(ports=(m.Pipe02s.port_Q, m.Reservoir2s.port_Q), directed=True)
m.c02r2_zs = Arc(ports=(m.Reservoir2s.port_z, m.Pipe02s.port_zhigh), directed=True)
m.c02r0_zs = Arc(ports=(m.Reservoir0s.port_z, m.Pipe02s.port_zlow), directed=True)

# m.c02r2_Qw = Arc(ports=(m.Pipe02w.port_Q, m.Reservoir2w.port_Q), directed=True)
# m.c02r2_zw = Arc(ports=(m.Reservoir2w.port_z, m.Pipe02w.port_zhigh), directed=True)
# m.c02r0_zw = Arc(ports=(m.Reservoir0w.port_z, m.Pipe02w.port_zlow), directed=True)


m.c21r2_Qs = Arc(ports=(m.VarSources.port_Qin, m.Reservoir2s.port_Q), directed=True)
m.c21r1_Qs = Arc(ports=(m.VarSources.port_Qout, m.Reservoir1s.port_Q), directed=True)

# m.c21r2_Qw = Arc(ports=(m.VarSourcew.port_Qin, m.Reservoir2w.port_Q), directed=True)
# m.c21r1_Qw = Arc(ports=(m.VarSourcew.port_Qout, m.Reservoir1w.port_Q), directed=True)

# m.c21r2_z = Arc(ports=(m.Reservoir2.port_z, m.VarSource.port_zhigh), directed=True)
# m.c21r1_z = Arc(ports=(m.Reservoir1.port_z, m.VarSource.port_zlow), directed=True)



#Electrical Node

m.pvebs = Arc(ports=(m.PVs.port_P, m.EBs.port_P), directed=True)
m.gridebs = Arc(ports=(m.Grids.port_P, m.EBs.port_P), directed=True)
m.batebs = Arc(ports=(m.Bats.port_P, m.EBs.port_P), directed = True)
m.loadebs = Arc(ports=(m.Loads.port_Pin, m.EBs.port_P), directed=True)

# m.pvebw = Arc(ports=(m.PVw.port_P, m.EBw.port_P), directed=True)    
# m.gridebw = Arc(ports=(m.Gridw.port_P, m.EBw.port_P), directed=True)
# m.batebw = Arc(ports=(m.Batw.port_P, m.EBw.port_P), directed = True)
# m.loadebw = Arc(ports=(m.Loadw.port_Pin, m.EBw.port_P), directed=True)



pyo.TransformationFactory("network.expand_arcs").apply_to(m) # apply arcs to model

#%%

from pyomo.environ import value
import os
import time

# Objective function
def obj_fun(m):
    return sum((m.Grids.Pbuy[t]*m.costs[t]/1e6 - m.Grids.Psell[t]*m.excs[t]/1e6) for t in l_t )+ m.PVs.Pdim*cost_new_pv  + (m.Bats.Pdim*cp_bat + m.Bats.Edim*ce_bat)

# def obj_fun(m):
#     return sum(( m.Gridw.Pbuy[t]*m.costw[t]/1e6 - m.Gridw.Psell[t]*m.excw[t]/1e6 + 
#              m.Grids.Pbuy[t]*m.costs[t]/1e6 - m.Grids.Psell[t]*m.excs[t]/1e6)/2 for t in l_t ) + (m.Batw.Pdim*cp_bat + m.Batw.Edim*ce_bat) #+ m.PV.Pdim*cost_new_pv


m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)
instance = m.create_instance()

start_time = time.time()

# solver = pyo.SolverFactory('asl:couenne') #ipopt asl:couenne gdpopt.enumerate
# solver.options['branch_fbbt'] = 'no'
# solver.solve(instance, tee=True)

os.environ['NEOS_EMAIL'] = 'pau.garcia.motilla@upc.edu'
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
#                 feas_tolerance 1e-2
#                 ''')
# solver = pyo.SolverFactory('asl:couenne')
# results = solver.solve(instance, tee=True)
# results.write()
# os.remove('couenne.opt') #Delete options
 
exec_time = time.time() - start_time

#%%

instance.Reservoir0.W.pprint()
instance.Reservoir1.W.pprint()
instance.Reservoir2.W.pprint()

instance.Pump1.Qout.pprint()
instance.Pump2.Qout.pprint()
instance.Turb1.Pe.pprint()

instance.Pipe02.Q.pprint()
instance.Pipe01.Q.pprint()

instance.Grid.Psell.pprint()
instance.Grid.Pbuy.pprint()

#%% GET RESULTS
from Utilities import get_results

file = './results/CEDER/Case3_24hWinter'
df_out, df_param, df_size = get_results(file=file, instance=instance, results=results, l_t=l_t, exec_time=exec_time)
#%%


#Plots

file = './results/CEDER/Case3_24hWinter'
df_out = pd.read_csv(file+'.csv').drop('Unnamed: 0',axis=1)
df_param = pd.read_csv(file+'_param.csv').drop('Unnamed: 0',axis=1)
df_size = pd.read_csv(file+'_size.csv').drop('Unnamed: 0',axis=1)


cbcolors = sns.color_palette('colorblind')

l_costs = df_grid['PVPC']

df_melted = pd.melt(df_out, id_vars=['t'], value_vars=['Pump1s.Pe', 'Pump2s.Pe', 'PVs.P', 'Loads.Pin', 'Turb1s.Pe', 'Grids.P','Bats.P'],
                    var_name='Variable', value_name='Value')
df_melted['Value'] = df_melted['Value'] / 1000

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]})

# Primer subplot: gráfico de barras apiladas
df_pivot = df_melted.pivot(index='t', columns='Variable', values='Value').fillna(0)
colors = sns.color_palette("colorblind", len(df_pivot.columns))

# Dividir los datos positivos y negativos
df_positive = df_pivot.clip(lower=0)
df_negative = df_pivot.clip(upper=0)

# Plotear los valores positivos
bottom_pos = np.zeros(len(df_positive))
for i, col in enumerate(df_positive.columns):
    ax1.bar(df_positive.index, df_positive[col], bottom=bottom_pos, label=col, color=colors[i])
    bottom_pos += df_positive[col]

# Plotear los valores negativos
bottom_neg = np.zeros(len(df_negative))
for i, col in enumerate(df_negative.columns):
    ax1.bar(df_negative.index, df_negative[col], bottom=bottom_neg, color=colors[i])
    bottom_neg += df_negative[col]

# Línea horizontal en y=0
ax1.axhline(0, color='black', linewidth=1)

# Configurar el título y las etiquetas de los ejes del primer subplot
ax1.set_title('Generated and Consumed Power', fontsize=18)
ax1.set_xlabel('Time', fontsize=14)
ax1.set_ylabel('Power [kW]', fontsize=14)

# Segundo subplot: gráfico de línea de costos
ax2.plot(df_out['t'], l_costs, color='black', marker='o', linestyle='dashed', linewidth=2, label='PVPC')
ax2.plot(df_out['t'], df_grid['Excedentes'], color='red', marker='o', linestyle='dashed', linewidth=2, label='Selling price')

ax2.set_ylabel('Price [€]', color='black', fontsize=14)
ax2.set_xlabel('Time', fontsize=14)
ax2.tick_params(axis='y', labelcolor='black')

# Obtener los handles y labels de cada eje
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

# Crear las leyendas y posicionarlas adecuadamente
legend1 = ax1.legend(handles1[:len(handles1)//2], labels1[:len(labels1)//2], loc='lower right', fontsize=10, bbox_to_anchor=(0.18, 0))
legend2 = ax1.legend(handles1[len(handles1)//2:], labels1[len(labels1)//2:], loc='lower right', fontsize=10, bbox_to_anchor=(0.38, 0))
legend3 = ax2.legend(handles2, labels2, loc='lower right', fontsize=10, bbox_to_anchor=(1, 1))

# Añadir las leyendas al primer subplot
ax1.add_artist(legend1)
ax1.add_artist(legend2)
ax1.set_ylim(-45, 45)

# Ajustar el espacio entre los subplots
plt.tight_layout()

# Mostrar el gráfico
plt.show()

# Guardar el gráfico
plt.rcParams['savefig.format']='pdf'
plt.savefig(file + 'Electrical', dpi=300)
   
df = df_out
colors = sns.color_palette("colorblind", len(df_pivot.columns))

fig = plt.figure(figsize=(6, 4))
fig.add_subplot(2,1,1)
ax1 = sns.lineplot(data=df, x=df.index,y='Reservoir1s.W', label='R1', color=cbcolors[2], marker='^')
sns.lineplot(data=df, x=df.index,y='Reservoir0s.W', label='R0', color=cbcolors[3], marker='v')
sns.lineplot(data=df, x=df.index,y='Reservoir2s.W', label='R2', color=cbcolors[0], marker='x')

plt.xlabel('Time',fontsize=14)
plt.ylabel('Volume[m3]',fontsize=14)
ax2 = fig.add_subplot(2,1,2)
df.plot(y=['Pump1s.Qout', 'Turb1s.Qout', 'Pump2s.Qout'], kind='bar', stacked=False, ylabel='Q', ax=ax2, 
        color=[cbcolors[2],cbcolors[1],cbcolors[0]], edgecolor='dimgray', label=['P1','T1','P2'], sharex=True)
ax2.legend()
plt.xlabel('Time',fontsize = 12)
plt.ylabel('Flow [m3/s]',fontsize = 12)
plt.tight_layout()
plt.show()

plt.rcParams['savefig.format']='pdf'
plt.savefig(file + '_VQ', dpi=300)



#%%
#Power 

fig = plt.figure(figsize=(5, 3))
ax1 = fig.add_subplot(1,1,1)

# Gráfico de barras en el eje izquierdo (ax1)
df['Load.P'] = -df['Load.P']
df.plot(y=['Pump1.Pe', 'PV.P', 'Grid.P', 'Turb1.Pe','Load.P','Pump2.Pe'], kind='bar', stacked=True, ax=ax1,
        color=['lightgrey', 'lightgrey', 'dimgrey', 'k', 'lightgrey','lightgrey'], edgecolor='dimgray')

bars = ax1.patches
patterns = ('////', '\\\\', None, None, 'xxxxx', None,'oooo')
hatches = [p for p in patterns for i in range(len(df))]
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)

ax1.legend(['$P_{p1}$', '$P_{PV}$', '$P_{g}$', '$P_{t1}$', '$P_{load}$','$P_{p2}$'], loc='upper center', bbox_to_anchor=(0.525, -0.2),
           fancybox=False, shadow=False, ncol=4)
ax1.axhline(0, color='k')
# Crear el segundo eje y gráfico de línea en el eje derecho (ax2)
ax2 = plt.twinx()
sns.lineplot(df, ax=ax2, x=range(hf-ho), y='l_cost', color='blue', linestyle='-', marker='o')  # Ajusta 'l_cost' según tus datos
ax2.set_xticks(range(hf-ho), labels=df.index, rotation=0)
ax2.set_ylabel('Cost')
ax2.legend(['Cost'], loc='upper center', bbox_to_anchor=(0.525, -0.25), fancybox=False, shadow=False)

# ax2 = plt.twinx()
# sns.lineplot(df ,x=df.index,y='l_cost', color='k', marker='.',ax=ax2)
# ax2.set_xticks(df.index, labels=df.index, rotation=0)

plt.ylabel('Price')
plt.xlabel('Time')
plt.title('Power consumption')
plt.tight_layout()
plt.show()
plt.rcParams['savefig.format']='pdf'
plt.savefig(file + '_P', dpi=300)



#%% PLOTS
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec


labels_hours = ['0','','','','','','6','','','','','','12','','','','','','18','','','','','23']

cbcolors = sns.color_palette('colorblind')

file = 'Case2_tstw/ISGT'

df_meteo_aug = pd.read_csv('data/meteo/LesPlanes_meteo_hour_aug.csv').head(24)
df_cons_aug = pd.read_csv('data/irrigation/LesPlanes_irrigation_aug.csv').head(24)
df_grid_aug = pd.read_csv('data/costs/PVPC_aug.csv').head(24)
df_meteo_jan = pd.read_csv('data/meteo/LesPlanes_meteo_hour_jan.csv').head(24)
df_cons_jan = pd.read_csv('data/irrigation/LesPlanes_irrigation_jan.csv').head(24)
df_grid_jan = pd.read_csv('data/costs/PVPC_jan.csv').head(24)


df_param = pd.read_csv('results/ISGT/'+file+'_param.csv')
w_lim = df_param['Reservoir1w.Wmin']

df = pd.read_csv('results/ISGT/'+file+'.csv')
df['PVw.Pf'] = -df_meteo_jan['Irr']/1000*215.28e3*0.98
df['PVs.Pf'] = -df_meteo_aug['Irr']/1000*215.28e3*0.98


if 'Turb1s.Qout' in df.columns:
    df['Rev1w.Qout'] = df['Pump1w.Qout'] - df['Turb1w.Qout']
    df['Rev1w.Pe'] = df['Pump1w.Pe'] + df['Turb1w.Pe']
    df['Rev1s.Qout'] = df['Pump1s.Qout'] - df['Turb1s.Qout']
    df['Rev1s.Pe'] = df['Pump1s.Pe'] + df['Turb1s.Pe']
else:
    df['Rev1w.Qout'] = df['Pump1w.Qout']
    df['Rev1w.Pe'] = df['Pump1w.Pe']
    df['Rev1s.Qout'] = df['Pump1s.Qout']
    df['Rev1s.Pe'] = df['Pump1s.Pe']

df['Irrigation1w.Qin'] = -df['Irrigation1w.Qout'] # no se per que passa pero Qin a vegades es reseteja a 0. La resta tot be
df['Irrigation1s.Qin'] = -df['Irrigation1s.Qout']

for c in df.columns:
    df[c] = df[c].apply(lambda x: 0 if (x>-1e-5 and x<1e-5) else x)

df_W = df[[col for col in df.columns if 'w.' in col]]
df_S = df[[col for col in df.columns if 's.' in col]]
df_W['t'] = df_W.index
df_S['t'] = df_S.index
df_W['PVPC'] = df_grid_jan['PVPC']
df_S['PVPC'] = df_grid_aug['PVPC']
df_W['Excedentes'] = df_grid_jan['Excedentes']
df_S['Excedentes'] = df_grid_aug['Excedentes']

df_W.columns = [col.replace('w.','.') for col in df_W.columns]
df_S.columns = [col.replace('s.','.') for col in df_W.columns]

#%%
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
    df.apply(lambda x: x/1000).plot(y=['PV.P','Pump2.Pe','Pump1.Pe','Turb1.Pe'], kind='bar', stacked=True, ax=ax1, ylabel='P (kW)',
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
# fig = plt.figure(figsize=(3.4, 1.3))
# sns.lineplot(df, x='hour', y='Qreg', style='Winter', hue='Winter', palette=[cbcolors[1],cbcolors[0]], errorbar=('ci',90))
# plt.legend(labels=['S','_','W','_'], ncol=1)
# plt.xlabel('Time (h)')
# plt.ylabel('Demand (m$^3$/h)')
# plt.xticks(range(24),labels=labels_hours, rotation=90)
# plt.yticks([0,100,200])
# # plt.ylim([0,200])
# plt.xlim([0,24])
# plt.tight_layout()
# plt.subplots_adjust(left=0.15, right=1, top=0.95, bottom=0.3)
# plt.show()
# plt.rcParams['savefig.format']='pdf'
# plt.savefig('results/ISGT/Irrigation_WS', dpi=300)
# plt.rcParams['savefig.format']='svg'
# plt.savefig('results/ISGT/Irrigation_WS', dpi=300)

#%%
df = df_melted
df['PVPC'] = l_costs
df['Excedentes'] = l_costs/3 
labels_hours = ['0','','','','','','6','','','','','','12','','','','','','18','','','','22']

fig = plt.figure(figsize=(3.4, 1.8))
plt.rcParams['axes.spines.right'] = True
ax1 = fig.add_subplot(1,1,1)

df.apply(lambda x: x/1000).plot(y=['PV.P','Turb.Pe','Grid.P'], kind='bar', stacked=True, ax=ax1, ylabel='P (kW)',
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
ax2.set_xticks(range(23), labels=labels_hours, rotation=90)
ax2.text(24.5,0.1,'Time (h)')

plt.ylabel('Price (€/MWh)')
plt.xlabel('Hour')
# plt.title('Power consumption')
plt.subplots_adjust(left=0.17, right=0.82, top=0.97, bottom=0.3)
plt.show()
