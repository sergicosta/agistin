#Real Example. Testing model for 1 day with Planes i Aixalelles real Data.

# Import pyomo
import pyomo.environ as pyo
from pyomo.network import *
import os
# Import builder
from Builder import data_parser, builder
from pyomo.opt import SolverStatus, TerminationCondition


# Import devices
from Devices.Reservoirs import Reservoir
from Devices.Sources import Source
from Devices.Pipes import Pipe
from Devices.Pumps import ReversibleRealPump
from Devices.EB import EB
from Devices.SolarPV import SolarPV
from Devices.MainGrid import Grid
from Devices.Batteries import Battery_Ex0

data_parser("ExampleReal", dt=3600) # dt = value of each timestep (if using SI this is seconds)

m = pyo.ConcreteModel()

# time
days=1
l_t = list(range(24*days))
m.t = pyo.Set(initialize=l_t)

# electricity cost
l_cost = [x/1000 for x in [0.04996,0.0513,0.05426,
                           0.05348,0.05361,0.05106,0.05038,
                           0.05130,0.07652,0.07118,0.11813,
                           0.12021,0.12026,0.11986,0.0699,
                           0.07047,0.07117,0.072279,0.1285,
                           0.144,0.19378,0.1745,0.08832,0.077]] * days

m.cost = pyo.Param(m.t, initialize=l_cost)
# cost_new_pv = 1000
# cost_new_bat = 1000

builder(m,'ExampleReal')
#%%
def obj_fun(m):
	return sum((m.Grid.Pbuy[t]*m.cost[t] - m.Grid.Psell[t]*m.cost[t]/2) for t in l_t )
with open("couenne.opt", "w") as file:
    file.write('''
               convexification_cuts 7
               convexification_points 6
               log_num_abt_per_level 1
               disj_depth_stop 2
               log_num_local_optimization_per_level 2
               display_stats yes
               ''')
    
m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)
instance = m.create_instance()
solver = pyo.SolverFactory('asl:couenne')
results = solver.solve(instance, tee=True)
results.write()
os.remove('couenne.opt') #Delete options


# os.environ['NEOS_EMAIL'] = 'pau.garcia.motilla@upc.edu'
# opt = pyo.SolverFactory("knitro")
# solver_manager = pyo.SolverManagerFactory('neos')
# results = solver_manager.solve(instance, opt=opt)
# results.write()




#%%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np


P_grid = instance.Grid.P.get_values()
P_Pump1 = instance.Pump1.Pe.get_values()
# P_Pump2 = instance.Pump2.Pe.get_values()
P_PV = instance.PV.P.get_values()
W_R0 = instance.Reservoir0.W.get_values()
W_R1 = instance.Reservoir1.W.get_values()
cost = instance.cost.extract_values()
E_bat = instance.Battery.E.get_values()
SOC = instance.Battery.SOC.get_values()
P_bat = instance.Battery.P.get_values()




df = pd.DataFrame.from_dict(P_grid, orient='index', columns=['P_grid'])
df = pd.concat([df, pd.DataFrame.from_dict(P_Pump1, orient='index', columns=['P_Pump1'])], axis=1)
# df = pd.concat([df, pd.DataFrame.from_dict(P_Pump2, orient='index', columns=['P_Pump2'])], axis=1)
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

# ax=sns.lineplot(data=df, x='t',y='W_R0', label='Reservoir0 (m3)', color='tab:blue', marker='o')
sns.lineplot(data=df, x='t',y='W_R1', label='Reservoir1 (m3)', color='tab:red', marker='o')
plt.axhline(y=0, color='k')
plt.xlabel('Time')
plt.ylabel('Volume')
plt.tight_layout()
plt.show()


fig = plt.figure(3)
ax = sns.barplot(data=df, x='t',y=(df['P_Pump1'])/1e3)
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



