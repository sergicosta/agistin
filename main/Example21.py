  
 #   importation of battery model block to the example number 2 for testing
"""
 Optimization usage example of the systema with batteries. It considers the same system as
 in Example 2, but adding the block of the battery and the battery sizing costs
 to the objective function.
""" 
 
# Import pyomo
import pyomo.environ as pyo
from pyomo.network import *

# Import devices
from Devices.Reservoirs import Reservoir_Ex0
from Devices.Sources import Source
from Devices.Pipes import Pipe_Ex0
from Devices.Pumps import Pump
from Devices.EB import EB
from Devices.SolarPV import SolarPV
from Devices.MainGrid import Grid
from Devices.Batteries import Battery_Ex0


# model
m = pyo.ConcreteModel()


# time
l_t = list(range(5))
m.t = pyo.Set(initialize=l_t)

# electricity cost
l_cost = [5,10,15,10,5]
m.cost = pyo.Param(m.t, initialize=l_cost)
cost_new_pv = 10
cost_new_battery = 1

# ===== Create the system =====
m.Reservoir1 = pyo.Block()
m.Reservoir0 = pyo.Block()
m.Irrigation1 = pyo.Block()
m.Pump1 = pyo.Block()
m.Pump2 = pyo.Block()
m.Pipe1 = pyo.Block()
m.PV = pyo.Block()
m.Grid = pyo.Block()
m.EB = pyo.Block()
m.Battery = pyo.Block()


data_irr = {'Q':[2,0,0,0,2]} # irrigation
Source(m.Irrigation1, m.t, data_irr, {})

data_res0 = {'W0':15, 'Wmin':0, 'Wmax':20}
data_res1 = {'W0':1, 'Wmin':0, 'Wmax':10}
init_res = {'Q':[0,0,0,0,0], 'W':[5,5,5,5,5]}

Reservoir_Ex0(m.Reservoir1, m.t, data_res1, init_res)
Reservoir_Ex0(m.Reservoir0, m.t, data_res0, init_res)

data_c1 = {'H0':20, 'K':0.05, 'Qmax':50} # canal
init_c1 = {'Q':[0,0,0,0,0], 'H':[20,20,20,20,20]}
Pipe_Ex0(m.Pipe1, m.t, data_c1, init_c1)

data_p = {'A':50, 'B':0.1, 'n_n':1450, 'eff':0.9, 'Qmax':20, 'Qnom':5, 'Pmax':9810*50*20} # pumps (both equal)
init_p = {'Q':[0,0,0,0,0], 'H':[20,20,20,20,20], 'n':[1450,1450,1450,1450,1450], 'Pe':[9810*5*20,9810*5*20,9810*5*20,9810*5*20,9810*5*20]}
Pump(m.Pump1, m.t, data_p, init_p)
Pump(m.Pump2, m.t, data_p, init_p)

data_pv = {'Pinst':50e3, 'Pmax':100e3, 'forecast':[0.4,1,1.7,1.2,0.5]} # PV
SolarPV(m.PV, m.t, data_pv)

Grid(m.Grid, m.t, {'Pmax':480e3}) # grid

EB(m.EB, m.t)

#Battery data
data = {'E0':0.1e3,'SOCmax':1,
        'SOCmin':0.0,'Pmax':100e3,
        'Einst':80e3,'Pinst':80e3,
        'Emax':100e3,'rend_ch':0.9,
        'rend_disc':1.1}

init_data = {'E':19e3,'P':19e3}
Battery_Ex0(m.Battery, m.t, data, init_data)


# Connections
m.p1r0 = Arc(ports=(m.Pump1.port_Qin, m.Reservoir0.port_Q), directed=True)
m.p1c1_Q = Arc(ports=(m.Pump1.port_Qout, m.Pipe1.port_Q), directed=True)
m.p1c1_H = Arc(ports=(m.Pump1.port_H, m.Pipe1.port_H), directed=True)
m.p2r0 = Arc(ports=(m.Pump2.port_Qin, m.Reservoir0.port_Q), directed=True)
m.p2c1_Q = Arc(ports=(m.Pump2.port_Qout, m.Pipe1.port_Q), directed=True)
m.p2c1_H = Arc(ports=(m.Pump2.port_H, m.Pipe1.port_H), directed=True)
m.c1r1 = Arc(ports=(m.Pipe1.port_Q, m.Reservoir1.port_Q), directed=True)
m.r1i1 = Arc(ports=(m.Irrigation1.port_Qin, m.Reservoir1.port_Q), directed=True)
m.ebp1 = Arc(ports=(m.Pump1.port_P, m.EB.port_P), directed=True)
m.ebp2 = Arc(ports=(m.Pump2.port_P, m.EB.port_P), directed=True)
m.grideb = Arc(ports=(m.Grid.port_P, m.EB.port_P), directed=True)
m.pveb = Arc(ports=(m.PV.port_P, m.EB.port_P), directed=True)
m.bat_eb = Arc(ports=(m.Battery.port_P, m.EB.port_P), directed=True)

pyo.TransformationFactory("network.expand_arcs").apply_to(m) # apply arcs to model


#%% RUN THE OPTIMIZATION

# Objective function
def obj_fun(m):
	return sum((m.Grid.Pbuy[t]*m.cost[t] - m.Grid.Psell[t]*m.cost[t]/2) for t in l_t ) + m.PV.Pdim*cost_new_pv + (m.Battery.Edim + m.Battery.Pdim)*cost_new_battery
m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)

instance = m.create_instance()
solver = pyo.SolverFactory('ipopt')
solver.solve(instance, tee=False)

instance.Reservoir1.W.pprint()
instance.Reservoir0.W.pprint()
instance.Grid.P.pprint()
instance.PV.Pdim.pprint()
instance.Battery.Edim.pprint()
instance.Battery.Pdim.pprint()


#%%Battery test

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np


P_grid = instance.Grid.P.get_values()
P_Pump1 = instance.Pump1.Pe.get_values()
P_Pump2 = instance.Pump2.Pe.get_values()
P_PV = instance.PV.P.get_values()
W_R0 = instance.Reservoir0.W.get_values()
W_R1 = instance.Reservoir1.W.get_values()
cost = instance.cost.extract_values()
E_bat = instance.Battery.E.get_values()
SOC = instance.Battery.SOC.get_values()
P_bat = instance.Battery.P.get_values()



df = pd.DataFrame.from_dict(P_grid, orient='index', columns=['P_grid'])
df = pd.concat([df, pd.DataFrame.from_dict(P_Pump1, orient='index', columns=['P_Pump1'])], axis=1)
df = pd.concat([df, pd.DataFrame.from_dict(P_Pump2, orient='index', columns=['P_Pump2'])], axis=1)
df = pd.concat([df, pd.DataFrame.from_dict(P_PV, orient='index', columns=['P_PV'])], axis=1)
df = pd.concat([df, pd.DataFrame.from_dict(W_R0, orient='index', columns=['W_R0'])], axis=1)
df = pd.concat([df, pd.DataFrame.from_dict(W_R1, orient='index', columns=['W_R1'])], axis=1)
df = pd.concat([df, pd.DataFrame.from_dict(cost, orient='index', columns=['cost'])], axis=1)
df = pd.concat([df, pd.DataFrame.from_dict(SOC, orient='index', columns=['SOC'])], axis=1)
df = pd.concat([df, pd.DataFrame.from_dict(P_bat, orient='index', columns=['P_bat'])], axis=1)
df = pd.concat([df, pd.DataFrame.from_dict(E_bat, orient='index', columns=['E_bat'])], axis=1)



df['t'] = df.index
df = df.set_index('t', drop=False)


fig = plt.figure(1)

ax=sns.lineplot(data=df, x='t',y='W_R0', label='Reservoir0 (m3)', color='tab:blue', marker='o')
sns.lineplot(data=df, x='t',y='W_R1', label='Reservoir1 (m3)', color='tab:red', marker='o')
plt.axhline(y=0, color='k')
plt.xlabel('Time')
plt.ylabel('Volume')
plt.tight_layout()
plt.show()


fig = plt.figure(3)
ax = sns.barplot(data=df, x='t',y=(df['P_Pump1'])/1e3+df['P_Pump2']/1e3)
plt.ylabel('Pump Power [kW]')
plt.xlabel('time')
plt.show()

fig = plt.figure(4)
ax = sns.barplot(data=df, x='t',y=(df['SOC']))
plt.show()

fig = plt.figure (5)
data = np.array([-df['P_grid'] / 1e3, -df['P_PV'] / 1e3, -df['P_bat'] / 1e3])

data_shape = np.shape(data)

# Take negative and positive data apart and cumulate
def get_cumulated_array(data, **kwargs):
    cum = data.clip(**kwargs)
    cum = np.cumsum(cum, axis=0)
    d = np.zeros(np.shape(data))
    d[1:] = cum[:-1]
    return d

cumulated_data = get_cumulated_array(data, min=0)
cumulated_data_neg = get_cumulated_array(data, max=0)

# Re-merge negative and positive data.
row_mask = (data < 0)
cumulated_data[row_mask] = cumulated_data_neg[row_mask]
data_stack = cumulated_data

cols = ['sienna', 'seagreen', 'steelblue']

fig, ax1 = plt.subplots(1, 1)  # Create a single subplot
ax2 = ax1.twinx()
lab = ['Grid', 'PV', 'Battery']

for i in np.arange(0, data_shape[0]):
    ax1.bar(np.arange(data_shape[1]), data[i], bottom=data_stack[i], color=cols[i], label=lab[i])

ax1.set_ylabel('Power Consumption (kW)')  # Set the label for the first Y-axis

sns.lineplot(data=df, x='t', y='cost', label='Cost (€/Wh)', color='tab:red', marker='o', ax=ax2)  # Use ax2 for the line plot
    
ax2.set_ylabel('Cost (€/Wh)')  # Set the label for the second Y-axis

ax1.legend(loc=(1.01,1.01))
ax1.set_xlabel('Time')

plt.show()