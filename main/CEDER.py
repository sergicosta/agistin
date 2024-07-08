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
from Devices.Pumps import Pump, RealPump, ReversiblePump, DiscretePump
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


date = '2023-07-02'
ho = 8
hf = 20
df_meteo = pd.DataFrame()
df_meteo ['Irr'] = df_data['Irr'][(df_data.index == date) & (df_data['Hour'] > ho) & (df_data['Hour'] <= hf)]
df_load = df_data['CT_Edificio1'][(df_data.index == date) & (df_data['Hour'] > ho) & (df_data['Hour'] <= hf)]
df_grid = pd.read_csv('data/costs/PVPC_jul.csv').iloc[ho:(hf)].reset_index(drop=True)

df_meteo.reset_index(drop=True, inplace=True)
df_load.reset_index(drop=True, inplace=True)
df_grid.reset_index(drop=True, inplace=True)


# time
T = hf-ho
l_t = list(range(T))
m.t = pyo.Set(initialize=l_t)

# electricity cost
demand = -df_load

l_cost = df_grid['PVPC']
l_exc = df_grid['Excedentes']
# l_cost = [650,600,700,800,1000]
# l_exc = [x/3 for x in [650,600,700,800,1000]]

m.cost = pyo.Param(m.t, initialize=l_cost)
m.exc = pyo.Param(m.t, initialize=l_exc)
cost_new_pv = 0.00126*T/1000
cp_bat = 0.00171*T/1000
ce_bat = 0.00856*T/1000

# Costos en €/(kwh·h). Considerant vida de 20 anys, del llibre de Oriol Gomis i Paco Diaz trobem:
#   - Ce = 750 €/kWh (10 anys de vida) -> x2 = 1500 €/kWh -> /20/365/24 = 0.00856 €/(kWh·h)


# ===== Create the system =====
m.Reservoir0 = pyo.Block()
m.Reservoir1 = pyo.Block()
m.Reservoir2 = pyo.Block()
m.Pump1 = pyo.Block()
m.Turb1 = pyo.Block()
m.Pipe01 = pyo.Block()
m.Pipe02 = pyo.Block()
m.VarSource = pyo.Block()
m.Pipe01T = pyo.Block()
m.PV = pyo.Block()
m.Grid = pyo.Block()
m.EB = pyo.Block()
m.HydroSwitch = pyo.Block()
m.Bat = pyo.Block()
m.Load = pyo.Block()



data_R0 = {'dt':3600, 'W0':1.5e3, 'Wmin':0, 'Wmax':2.5e3, 'zmin':1, 'zmax':4}
init_R0 = {'Q':[0]*T, 'W':[1.5e3]*T}
Reservoir(m.Reservoir0, m.t, data_R0, init_R0)

data_R1 = {'dt':3600, 'W0':0.5e3, 'Wmin':0e3, 'Wmax':1.5e3, 'zmin':67, 'zmax':69}
init_R1 = {'Q':[0]*T, 'W':[0.5e3]*T}
Reservoir(m.Reservoir1, m.t, data_R1, init_R1)

data_R2 = {'dt':3600, 'W0':0.25e3, 'Wmin':0e3, 'Wmax':0.5e3, 'zmin':77, 'zmax':79}
init_R2 = {'Q':[0]*T, 'W':[0.25e3]*T}
Reservoir(m.Reservoir2, m.t, data_R2, init_R2)

data_c1 = {'K':0.02, 'Qmax':10} # canal
init_c1 = {'Q':[0]*T, 'H':[77]*T, 'H0':[77]*T, 'zlow':[0]*T, 'zhigh':[77]*T}
Pipe(m.Pipe01, m.t, data_c1, init_c1)
Pipe(m.Pipe01T, m.t, data_c1, init_c1)

data_c2 = {'K':0.02, 'Qmax':10} # canal
init_c2 = {'Q':[0]*T, 'H':[67]*T, 'H0':[67]*T, 'zlow':[2]*T, 'zhigh':[67]*T}
Pipe(m.Pipe02, m.t, data_c2, init_c2)

# data_c3 = {'K':0.02, 'Qmax':10} # canal
# init_c3 = {'Q':[0]*T, 'H':[10]*T, 'H0':[10]*T, 'zlow':[67]*T, 'zhigh':[77]*T}
# Pipe(m.Pipe21, m.t, data_c3, init_c3)

data_Source = {'S':0.2}
init_Source = {'Q':[0.0]*T,'Qin':[0]*T,'Qout':[0]*T,'Qmax':[0.0]*T,'zhigh':[77]*T, 'zlow': [66]*T, 'H':[10]*T}
VarSource(m.VarSource, m.t, data_Source, init_Source )



data_p1 = {'A':121.54, 'B':0.0003, 'n_n':2900, 'eff':0.8, 'S':0.1*0.1*3.14,'Pn':7.5e3, 'Npumps':4,'Qnom':0.05,'Qmax':2,'Qmin':0,'Pmax':30e3} # pumps (both equal)
init_p1 = {'Q':[0]*T, 'H':[67]*T, 'n':[2900]*T, 'Pe':[7.5e3]*T}
DiscretePump(m.Pump1, m.t, data_p1, init_p1)


data_t = {'eff':0.8, 'Pmax':40e3}
init_t = {'Q':[0]*T, 'H':[67]*T, 'Pe':[-40e3*0.9]*T}
Turbine(m.Turb1, m.t, data_t, init_t)

# data_tDumb = {'eff':0.0, 'Pmax':0}
# init_tDumb = {'Q':[0]*T, 'H':[67]*T, 'Pe':[0]*T}
# Turbine(m.DumbTurb, m.t, data_tDumb, init_tDumb)

data_pv = {'Pinst':16e3, 'Pmax':16e3, 'forecast':df_meteo['Irr']/1000, 'eff':0.98} # PV
SolarPV(m.PV, m.t, data_pv)

data_bat = {'dt':3600, 'E0':0.05, 'Emax':200e3, 'Pmax':200e3, 'SOCmin':0, 'SOCmax':1.0, 'eff_ch':0.8, 'eff_dc':0.8,'Einst':0.1, 'Pinst':0}
init_bat = {'E':[0.5]*T, 'P':[0]*T}
NewBattery(m.Bat, m.t, data_bat, init_bat)

Grid(m.Grid, m.t, {'Pmax':100e3}) # grid

# Grid(m.DumbGrid, m.t, {'Pmax':100e3})

data_load = {'P':demand} # Electrical Demand
Load(m.Load, m.t, data_load, {})


EB(m.EB, m.t)

HydroSwitch(m.HydroSwitch, m.t)

#Energy constraint

def ConstraintW1min(m):
    return m.Reservoir0.W[T-1]*m.Reservoir0.z[T-1] + m.Reservoir1.W[T-1]*m.Reservoir1.z[T-1] + m.Reservoir2.W[T-1]*m.Reservoir2.z[T-1]  >= (m.Reservoir0.W0*m.Reservoir0.z[0] + m.Reservoir1.W0*m.Reservoir1.z[0] + m.Reservoir2.W0*m.Reservoir2.z[0])*0.95
m.Reservoir1_c_W1min = pyo.Constraint(rule=ConstraintW1min)


def ConstraintW1max(m):
    return m.Reservoir0.W[T-1]*m.Reservoir0.z[T-1] + m.Reservoir1.W[T-1]*m.Reservoir1.z[T-1] + m.Reservoir2.W[T-1]*m.Reservoir2.z[T-1]  >= (m.Reservoir0.W0*m.Reservoir0.z[0] + m.Reservoir1.W0*m.Reservoir1.z[0] + m.Reservoir2.W0*m.Reservoir2.z[0])*1.05
m.Reservoir1_c_W1max = pyo.Constraint(rule=ConstraintW1max)



# Connections

#Pump1 to R1 and Pipe12 and EB
m.p1r0 = Arc(ports=(m.Pump1.port_Qin, m.Reservoir0.port_Q), directed=True)
m.p1eb = Arc(ports=(m.Pump1.port_P, m.EB.port_P), directed=True)


m.p1h1_Q = Arc(ports=(m.Pump1.port_Qout, m.HydroSwitch.port_Qout), directed=True)
m.p1h1_H = Arc(ports=(m.Pump1.port_H, m.HydroSwitch.port_Hout), directed=True)


m.HSp01 = Arc(ports=(m.HydroSwitch.port_Qin0, m.Pipe01.port_Q), directed=True)
m.HSp02 = Arc(ports=(m.HydroSwitch.port_Qin1, m.Pipe02.port_Q), directed=True)
m.HSp01_H = Arc(ports=(m.HydroSwitch.port_Hin0, m.Pipe01.port_H), directed=True)
m.HSp02_H = Arc(ports=(m.HydroSwitch.port_Hin1, m.Pipe02.port_H), directed=True)

#Turbine to Reservoir1
m.t1r0 = Arc(ports=(m.Turb1.port_Qout, m.Reservoir0.port_Q), directed=True)
m.t1c1_Q = Arc(ports=(m.Turb1.port_Qin, m.Pipe01T.port_Q), directed=True)
m.t1c1_H = Arc(ports=(m.Turb1.port_H, m.Pipe01T.port_H), directed=True)
m.t1eb = Arc(ports=(m.Turb1.port_P, m.EB.port_P), directed=True)

#Pipe 01T
m.c01tr1 = Arc(ports=(m.Pipe01T.port_Q, m.Reservoir1.port_Q),directed=True)
m.c01tr1_z = Arc(ports=(m.Reservoir1.port_z, m.Pipe01T.port_zhigh), directed=True)
m.c01tr0_z = Arc(ports=(m.Reservoir0.port_z, m.Pipe01T.port_zlow), directed=True)


#Pipe 01 to Reservoir 1
m.c01r1_Q = Arc(ports=(m.Pipe01.port_Q, m.Reservoir1.port_Q), directed=True)
m.c01r1_z = Arc(ports=(m.Reservoir1.port_z, m.Pipe01.port_zhigh), directed=True)
m.c01r0_z = Arc(ports=(m.Reservoir0.port_z, m.Pipe01.port_zlow), directed=True)


#Pipe 02 to R2 and R0
m.c02r2_Q = Arc(ports=(m.Pipe02.port_Q, m.Reservoir2.port_Q), directed=True)
m.c02r2_z = Arc(ports=(m.Reservoir2.port_z, m.Pipe02.port_zhigh), directed=True)
m.c02r0_z = Arc(ports=(m.Reservoir0.port_z, m.Pipe02.port_zlow), directed=True)


# Pipe 21 to Reservoir2 via turbine (valve)
# m.c21r2_Q = Arc(ports=(m.Pipe21.port_Q, m.Reservoir2.port_Q), directed=True)
# m.c21r1_Q = Arc(ports=(m.Pipe21.port_Q, m.Reservoir1.port_Q), directed=True)
# m.c21r2_z = Arc(ports=(m.Reservoir2.port_z, m.Pipe21.port_zhigh), directed=True)
# m.c21r0_z = Arc(ports=(m.Reservoir1.port_z, m.Pipe21.port_zlow), directed=True)

# m.t2r0 = Arc(ports=(m.DumbTurb.port_Qout, m.Reservoir1.port_Q), directed=True)
# m.t2c1_Q = Arc(ports=(m.DumbTurb.port_Qin, m.Pipe21.port_Q), directed=True)
# m.t2c1_H = Arc(ports=(m.DumbTurb.port_H, m.Pipe21.port_H), directed=True)
# m.t2eb = Arc(ports=(m.DumbTurb.port_P, m.DumbGrid.port_P), directed=True)

m.c21r2_Q = Arc(ports=(m.VarSource.port_Qin, m.Reservoir2.port_Q), directed=True)
m.c21r1_Q = Arc(ports=(m.VarSource.port_Qout, m.Reservoir1.port_Q), directed=True)

m.c21r2_z = Arc(ports=(m.Reservoir2.port_z, m.VarSource.port_zhigh), directed=True)
m.c21r1_z = Arc(ports=(m.Reservoir1.port_z, m.VarSource.port_zlow), directed=True)

#Electrical Node

m.pveb = Arc(ports=(m.PV.port_P, m.EB.port_P), directed=True)
m.grideb = Arc(ports=(m.Grid.port_P, m.EB.port_P), directed=True)
m.bateb = Arc(ports=(m.Bat.port_P, m.EB.port_P), directed = True)
m.loadeb = Arc(ports=(m.Load.port_Pin, m.EB.port_P), directed=True)



pyo.TransformationFactory("network.expand_arcs").apply_to(m) # apply arcs to model

#%%

from pyomo.environ import value
import os
import time

# Objective function
def obj_fun(m):
	return sum((m.Grid.Pbuy[t]*m.cost[t]/1e6 - m.Grid.Psell[t]*m.exc[t]/1e6) for t in l_t ) + m.PV.Pdim*cost_new_pv + m.Bat.Pdim*cp_bat + m.Bat.Edim*ce_bat
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
instance.Turb1.Pe.pprint()

instance.Pipe02.Q.pprint()
instance.Pipe01.Q.pprint()

instance.Grid.Psell.pprint()
instance.Grid.Pbuy.pprint()

#%% GET RESULTS
from Utilities import get_results

file = './results/CEDER/12h_20h_Ctr_4Pumps_commonBus_16kWmax_CEDERDATA'
df_out, df_param, df_size = get_results(file=file, instance=instance, results=results, l_t=l_t, exec_time=exec_time)
#%%


#Plots

file = 'results/CEDER/12h_20h_Ctr_4Pumps_commonBus_16kWmax_CEDERDATA'
df_out = pd.read_csv(file+'.csv').drop('Unnamed: 0',axis=1)
df_param = pd.read_csv(file+'_param.csv').drop('Unnamed: 0',axis=1)
df_size = pd.read_csv(file+'_size.csv').drop('Unnamed: 0',axis=1)

#Reservoir levels
df_melted = pd.melt(df_out, id_vars=['t'], value_vars=['Reservoir0.W', 'Reservoir1.W', 'Reservoir2.W'],
                    var_name='Variable', value_name='Value')

plt.figure(figsize=(10, 6))
sns.lineplot(data=df_melted, x='t', y='Value', hue='Variable', marker='o')
plt.title('Reservoir levels')
plt.xlabel('Time')
plt.ylabel('Volume [m3]')
plt.ylim(0,None)
plt.legend()
plt.show()


#Power Flow
# df_out['Pumps.Pe'] = df_out['Pump1.Pe'] + df_out['Pump2.Pe']
df_out['Grid.P'] = -df_out['Grid.P']

# Transformar el DataFrame
df_melted = pd.melt(df_out, id_vars=['t'], value_vars=['Pump1.Pe', 'PV.P', 'Load.Pin','Turb1.Pe','Grid.P'],
                    var_name='Variable', value_name='Value')

# Crear la figura y los ejes
fig, ax1 = plt.subplots(figsize=(12, 8))

# Crear el gráfico de barras en el eje principal
sns.barplot(data=df_melted, x='t', y='Value', hue='Variable', ax=ax1)

# Añadir la línea negra en y=0
ax1.axhline(0, color='black', linewidth=1)

# Crear un segundo eje y
ax2 = ax1.twinx()

# Añadir la línea que sigue la variable 'Cost' en el segundo eje
ax2.plot(df_out['t'], l_cost, color='black', marker='o', linestyle='-', linewidth=2, label='Cost')
ax2.set_ylabel('Cost', color='black')
ax2.tick_params(axis='y', labelcolor='black')

# Configurar el título y las etiquetas del eje x y el eje y principal
ax1.set_title('Generated and consumed Powers')
ax1.set_xlabel('Time')
ax1.set_ylabel('Power [kW]')
ax1.legend(title='Variable', loc='upper left')

# Ajustar la leyenda para incluir ambas
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left')

# Mostrar el gráfico
plt.show()

#%%


file = 'results/CEDER/12h_20h_Ctr_4Pumps_commonBus_16kWmax_CEDERDATA'
df_out = pd.read_csv(file+'.csv').drop('Unnamed: 0',axis=1)
df_param = pd.read_csv(file+'_param.csv').drop('Unnamed: 0',axis=1)
df_size = pd.read_csv(file+'_size.csv').drop('Unnamed: 0',axis=1)


df = pd.read_csv(file+'.csv')
df['PV.Pf'] = -df['PV.forecast']*(df_param['PV.Pinst'].iloc[0] + df_size['PV.Pdim'])


df['l_cost'] = df_grid['PVPC']
df['l_exc'] = l_exc


df['hour'] = list(range(ho+1,hf+1))

df = df.set_index('hour')


    

fig = plt.figure(figsize=(3.4, 2.3))
fig.add_subplot(2,1,1)
ax1 = sns.lineplot(data=df, x=df.index,y='Reservoir1.W', label='R1', color='grey', marker='^')
sns.lineplot(data=df, x=df.index,y='Reservoir0.W', label='R0', color='k', marker='v')
sns.lineplot(data=df, x=df.index,y='Reservoir2.W', label='R2', color='dimgray', marker='x')

plt.xlabel('Time')
plt.ylabel('Volume')
ax2 = fig.add_subplot(2,1,2)
df.plot(y=['Pump1.Qout', 'Turb1.Qout', 'HydroSwitch.Qin0', 'HydroSwitch.Qin1'], kind='bar', stacked=False, ylabel='Q', ax=ax2, 
        color=['lightgrey','k','lightgrey', 'lightgrey'], edgecolor='dimgray', label=['P1','T1','HS0','HS1'], sharex=True)
bars = ax2.patches
patterns =('xxxx',None, "////", "\\\\")
hatches = [p for p in patterns for i in range(len(df))]
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)
ax2.legend()
plt.xlabel('Time')
plt.ylabel('Flow')
plt.tight_layout()
plt.show()
plt.rcParams['savefig.format']='pdf'
plt.savefig(file + '_VQ', dpi=300)


#Power 

fig = plt.figure(figsize=(5, 3))
ax1 = fig.add_subplot(1,1,1)

# Gráfico de barras en el eje izquierdo (ax1)
df['Load.P'] = -df['Load.P']
df.plot(y=['Pump1.Pe', 'PV.P', 'Grid.P', 'Turb1.Pe','Load.P'], kind='bar', stacked=True, ax=ax1,
        color=['lightgrey', 'lightgrey', 'dimgrey', 'k', 'lightgrey'], edgecolor='dimgray')

bars = ax1.patches
patterns = ('////', '\\\\', None, None, 'xxxxx', None)
hatches = [p for p in patterns for i in range(len(df))]
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)

ax1.legend(['$P_{p1}$', '$P_{PV}$', '$P_{g}$', '$P_{t1}$', '$P_{load}$'], loc='upper center', bbox_to_anchor=(0.525, -0.2),
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


