  
 #   importation of battery model block to the example number 2 for testing
"""
 Optimization usage example of the systema with batteries. It considers the same system as
 in Example 2, but adding the block of the battery and the battery sizing costs
 to the objective function.
""" 
 
# Import pyomo
import pyomo.environ as pyo
from pyomo.network import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns

# Import devices
from Devices.Reservoirs import Reservoir
from Devices.Sources import Source
from Devices.Pipes import Pipe
from Devices.Pumps import RealPump
from Devices.EB import EB
from Devices.SolarPV import SolarPV
from Devices.MainGrid import Grid
from Devices.Batteries import Battery


# model
m = pyo.ConcreteModel()


# time
T = 10
l_t = list(range(T))
m.t = pyo.Set(initialize=l_t)

# electricity cost
l_cost = [10,10,5,5,30]*2
m.cost = pyo.Param(m.t, initialize=l_cost)
cost_new_pv = 10
cost_new_battery = 2

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


data_irr = {'Q':[2,0,0,0,0.6]*2} # irrigation
Source(m.Irrigation1, m.t, data_irr, {})

data_res0 = {'dt':1, 'W0':5, 'Wmin':0, 'Wmax':10, 'zmin':90, 'zmax':95}
data_res1 = {'dt':1, 'W0':5, 'Wmin':0, 'Wmax':10, 'zmin':100, 'zmax':105}
init_res = {'Q':[0]*T, 'W':[5]*T}

Reservoir(m.Reservoir1, m.t, data_res1, init_res)
Reservoir(m.Reservoir0, m.t, data_res0, init_res)

data_c1 = {'K':0.02, 'Qmax':50} # canal
init_c1 = {'Q':[0]*T, 'H':[20]*T, 'H0':[20]*T, 'zlow':[90]*T, 'zhigh':[100]*T}
Pipe(m.Pipe1, m.t, data_c1, init_c1)

data_p = {'A':50, 'B':0.01, 'n_n':1450, 'eff':0.9, 'Qmax':20, 'Qnom':5, 'Pmax':9810*50*20, 'Qmin':0.5, 'Qmax':1.1} # pumps (both equal)
init_p = {'Q':[0]*T, 'H':[20]*T, 'n':[1450]*T, 'Pe':[9810*5*20]*T}
RealPump(m.Pump1, m.t, data_p, init_p)
RealPump(m.Pump2, m.t, data_p, init_p)

data_pv = {'Pinst':50e3, 'Pmax':100e3, 'forecast':[1,1,1.7,0.5,0]*2, 'eff': 0.98} # PV
SolarPV(m.PV, m.t, data_pv)

Grid(m.Grid, m.t, {'Pmax':1e9}) # grid

EB(m.EB, m.t)

#Battery data
data = {'E0':0.1e3,'SOCmax':1,
        'SOCmin':0.0,'Pmax':100e3,
        'Einst':10e3,'Pinst':10e3,
        'Emax':100e3,'rend_ch':0.9,
        'rend_dc':0.9}

init_data = {'E':[19e3]*T,'P':19e3}
Battery(m.Battery, m.t, data, init_data)


# Connections
m.p1r0 = Arc(ports=(m.Pump1.port_Qin, m.Reservoir0.port_Q), directed=True)
m.p1c1_Q = Arc(ports=(m.Pump1.port_Qout, m.Pipe1.port_Q), directed=True)
m.p1c1_H = Arc(ports=(m.Pump1.port_H, m.Pipe1.port_H), directed=True)
m.ebp1 = Arc(ports=(m.Pump1.port_P, m.EB.port_P), directed=True)

m.p2r0 = Arc(ports=(m.Pump2.port_Qin, m.Reservoir0.port_Q), directed=True)
m.p2c1_Q = Arc(ports=(m.Pump2.port_Qout, m.Pipe1.port_Q), directed=True)
m.p2c1_H = Arc(ports=(m.Pump2.port_H, m.Pipe1.port_H), directed=True)
m.ebp2 = Arc(ports=(m.Pump2.port_P, m.EB.port_P), directed=True)

m.c1r1 = Arc(ports=(m.Pipe1.port_Q, m.Reservoir1.port_Q), directed=True)
m.c1r1_z = Arc(ports=(m.Reservoir1.port_z, m.Pipe1.port_zhigh), directed=True)
m.c1r0_z = Arc(ports=(m.Reservoir0.port_z, m.Pipe1.port_zlow), directed=True)

m.r1i1 = Arc(ports=(m.Irrigation1.port_Qin, m.Reservoir1.port_Q), directed=True)

m.grideb = Arc(ports=(m.Grid.port_P, m.EB.port_P), directed=True)
m.pveb = Arc(ports=(m.PV.port_P, m.EB.port_P), directed=True)
m.bat_eb = Arc(ports=(m.Battery.port_P, m.EB.port_P), directed=True)

pyo.TransformationFactory("network.expand_arcs").apply_to(m) # apply arcs to model


#%% RUN THE OPTIMIZATION
from pyomo.environ import value
import os


# Objective function
def obj_fun(m):
	return sum((m.Grid.Pbuy[t]*m.cost[t] - m.Grid.Psell[t]*m.cost[t]/2) for t in l_t ) + m.PV.Pdim*cost_new_pv + (m.Battery.Edim + m.Battery.Pdim)*cost_new_battery
m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)

instance = m.create_instance()
# solver = pyo.SolverFactory('asl:couenne') #ipopt asl:couenne gdpopt.enumerate
# solver.options['branch_fbbt'] = 'no'
# solver.solve(instance, tee=True)

# os.environ['NEOS_EMAIL'] = 'sergi.costa.dilme@upc.edu'
# opt = pyo.SolverFactory("knitro")
# solver_manager = pyo.SolverManagerFactory('neos')
# results = solver_manager.solve(instance, opt=opt)
# results.write()

with open("couenne.opt", "w") as file:
    file.write('''time_limit 100000
                convexification_cuts 4
                convexification_points 3
                delete_redundant yes
                use_quadratic no
                feas_tolerance 1e-5
                ''')
solver = pyo.SolverFactory('asl:couenne')
results = solver.solve(instance, tee=True)
results.write()
os.remove('couenne.opt') #Delete options

# instance = m.create_instance()
# solver = pyo.SolverFactory('ipopt')
# results = solver.solve(instance, tee=True)

#%%

from pyomo.environ import value

df_out = pd.DataFrame(l_t, columns=['t'])
df_param = pd.DataFrame()
for i in range(len(instance._decl_order)):
    e = instance._decl_order[i][0]
    if e is None:
        continue
    name = e.name
    
    if "pyomo.core.base.block.ScalarBlock" not in str(e.type):
        continue
    
    for ii in range(len(e._decl_order)):
        v = e._decl_order[ii][0]
        vals = 0
        
        if "pyomo.core.base.var.IndexedVar" in str(v.type): #Var(t)
            vals = v.get_values()
        elif "pyomo.core.base.param.IndexedParam" in str(v.type): #Param(t)
            vals = v.extract_values()
        elif "pyomo.core.base.var.ScalarVar" in str(v.type): #Var
            vals = v.get_values()
        elif "pyomo.core.base.param.ScalarParam" in str(v.type): #Param
            vals = v.extract_values()
            df_param = pd.concat([df_param, pd.DataFrame.from_dict(vals, orient='index', columns=[v.name])], axis=1)
            continue
        else:
            continue
        
        df_out = pd.concat([df_out, pd.DataFrame.from_dict(vals, orient='index', columns=[v.name])], axis=1)


file = './results/Example21'
df_out.to_csv(file+'.csv')
results.write(filename=file+'_results.txt')
with open(file+'_results.txt','a') as f:
    f.write('\nGOAL VALUE:\n'+str(value(instance.goal))+'\n')
    f.close()

#%%Battery test

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

df = pd.read_csv(file+'.csv')
df['PV.Pf'] = -df['PV.forecast']*(df_param['PV.Pinst'].iloc[0] + df['PV.Pdim'].iloc[T])

# Hydro
fig = plt.figure()
fig.add_subplot(2,1,1)
sns.lineplot(data=df, x='t',y='Reservoir0.W', label='Reservoir0 (m$^3$)', color='tab:blue', marker='o')
sns.lineplot(data=df, x='t',y='Reservoir1.W', label='Reservoir1 (m$^3$)', color='tab:red', marker='o')
plt.axhline(y=df_param['Reservoir0.Wmin'].iloc[0], color='k')
plt.axhline(y=df_param['Reservoir1.Wmin'].iloc[0], color='k')
plt.xlabel('Time')
plt.ylabel('Volume')
ax2 = fig.add_subplot(2,1,2)
df.plot(y=['Pump1.Qout','Pump2.Qout'], kind='bar', stacked=False, ylabel='Q', ax=ax2, color=['tab:blue','tab:red'], label=['Pump1','Pump2'])
plt.xlabel('Time')
plt.ylabel('Flow')
plt.tight_layout()
plt.show()

# Power
fig = plt.figure(figsize=(3.4, 2.5))
plt.rc('font',size=9)
# plt.suptitle('Power consumption')
ax1 = fig.add_subplot(1,1,1)
df.apply(lambda x: x/1000).plot(y=['Pump1.Pe','Pump2.Pe','PV.P', 'Grid.P'], kind='bar', stacked=True, ax=ax1, ylabel='P (kW)')
df.apply(lambda x: x/1000).plot(y=['PV.Pf'], kind='bar', ax=ax1, stacked=False, ylabel='P (kW)', color='tab:green', alpha=0.3)
ax1.legend(['$P_{p1}$','$P_{p2}$','$P_{PV}$', '$P_{g}$', '$\hat{P}_{PV}$'],loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=False, shadow=False, ncol=5)
ax1.axhline(0,color='k')
# ax1.set_xticklabels([])
# ax1.set_xticks(range(24), labels=range(24), rotation=90)
# ax2 = fig.add_subplot(2,1,2, sharex=ax1)
ax2 = plt.twinx()
sns.lineplot(df.iloc[0:T],x='t',y=l_cost, ax=ax2, color='tab:red')#, label='Buy')
# sns.lineplot(df_grid,x='Hour',y='Excedentes', ax=ax2, color='tab:blue', label='Sell')
ax2.set_xticks(range(T), labels=range(T), rotation=90)
plt.ylabel('Price (€/MWh)')
plt.xlabel('Hour')
plt.title('Power consumption')
plt.show()
plt.tight_layout()
# plt.rcParams['savefig.format']='pdf'
# plt.savefig('results/CIRED/' + file + '_P', dpi=300)


# Battery
fig = plt.figure()
sns.lineplot(data=df, x='t',y='Battery.E', label='Energy', color='tab:blue', marker='o')
sns.barplot(data=df, x='t', y='Battery.P', label='Power', color='tab:red')
plt.axhline(0,color='k')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.tight_layout()
plt.show()