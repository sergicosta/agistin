
"""
 Optimization usage for testing the RealPump model, where it's introduced the working limits of it.
 It considers the same system as in Example 22, but adding the constraints of the pump.
""" 
 
# Import pyomo
import pyomo.environ as pyo
from pyomo.network import *

# Import devices
from Devices.Reservoirs import Reservoir_Ex0
from Devices.Sources import Source
from Devices.Pipes import Pipe_Ex0
from Devices.Pumps import RealPump
from Devices.EB import EB
# from Devices.SolarPV import SolarPV
from Devices.MainGrid import Grid
# from Devices.Batteries import Battery_Ex0

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
# m.Pump2 = pyo.Block()
m.Pipe1 = pyo.Block()
# m.PV = pyo.Block()
m.Grid = pyo.Block()
m.EB = pyo.Block()
# m.Battery = pyo.Block()

Q_init = [0,0,0,0,0]
data_irr = {'Q':[0,0,0,0,4]} # irrigation
Source(m.Irrigation1, m.t, data_irr, {})

data_res0 = {'W0':20, 'Wmin':0, 'Wmax':20}
data_res1 = {'W0':0, 'Wmin':0, 'Wmax':20}
init_res = {'Q':Q_init, 'W':[15,15,15,15,15]}
init_res1 = {'Q':Q_init, 'W':[0,0,0,0,0]}
Reservoir_Ex0(m.Reservoir1, m.t, data_res1, init_res1)
Reservoir_Ex0(m.Reservoir0, m.t, data_res0, init_res)

data_c1 = {'H0':20, 'K':0.05, 'Qmax':50} # canal
init_c1 = {'Q':Q_init, 'H':[20,20,20,20,20]}   
Pipe_Ex0(m.Pipe1, m.t, data_c1, init_c1)

data_p = {'A':50, 'B':0.1, 'n_n':1450, 'eff':0.9, 'Qnom':5, 'Pmax':9810*50*20,
          'Qmin':0.8, 'Qmax':1.3} # pumps (both equal)
init_p = {'Q':Q_init, 'H':[20,20,20,20,20], 'n':[1450,1450,1450,1450,1450], 'Pe':[9810*5*20,9810*5*20,9810*5*20,9810*5*20,9810*5*20],}
RealPump(m.Pump1, m.t, data_p, init_p)

# RealPump(m.Pump2, m.t, data_p, init_p)

# data_pv = {'Pinst':50e3, 'Pmax':100e3, 'forecast':[0.4,1,1.7,1.2,0.5]} # PV
# SolarPV(m.PV, m.t, data_pv)

Grid(m.Grid, m.t, {'Pmax':9000e4}) # grid

EB(m.EB, m.t)

#Battery data
# data = {'E0':15e3,'SOCmax':1,'SOCmin':0.05,'Pmax':100e3,'Einst':50e3,'Pinst':50e3,
#          'Emax':100e3,'rend_ch':0.9,'rend_disc':1.1}
# init_data = {'E':[19e3,19e3,19e3,19e3,19e3],'P':19e3}
# Battery_Ex0(m.Battery, m.t, data, init_data)


# Connections
m.p1r0 = Arc(ports=(m.Pump1.port_Qin, m.Reservoir0.port_Q), directed=True)
m.p1c1_Q = Arc(ports=(m.Pump1.port_Qout, m.Pipe1.port_Q), directed=True)
m.p1c1_H = Arc(ports=(m.Pump1.port_H, m.Pipe1.port_H), directed=True)
# m.p2r0 = Arc(ports=(m.Pump2.port_Qin, m.Reservoir0.port_Q), directed=True)
# m.p2c1_Q = Arc(ports=(m.Pump2.port_Qout, m.Pipe1.port_Q), directed=True)
# m.p2c1_H = Arc(ports=(m.Pump2.port_H, m.Pipe1.port_H), directed=True)
m.c1r1 = Arc(ports=(m.Pipe1.port_Q, m.Reservoir1.port_Q), directed=True)
m.r1i1 = Arc(ports=(m.Irrigation1.port_Qin, m.Reservoir1.port_Q), directed=True)
m.ebp1 = Arc(ports=(m.Pump1.port_P, m.EB.port_P), directed=True)
# m.ebp2 = Arc(ports=(m.Pump2.port_P, m.EB.port_P), directed=True)
m.grideb = Arc(ports=(m.Grid.port_P, m.EB.port_P), directed=True)
# m.pveb = Arc(ports=(m.PV.port_P, m.EB.port_P), directed=True)
# m.bat_eb = Arc(ports=(m.Battery.port_P, m.EB.port_P), directed=True)

pyo.TransformationFactory("network.expand_arcs").apply_to(m) # apply arcs to model


#%% RUN THE OPTIMIZATION

# Objective function
def obj_fun(m):
	return sum((m.Grid.Pbuy[t]*m.cost[t] - m.Grid.Psell[t]*m.cost[t]/2) for t in l_t )
m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)

instance = m.create_instance()
solver = pyo.SolverFactory('asl:couenne')
solver.solve(instance, tee=True)

#%%
instance.Reservoir1.W.pprint()
instance.Reservoir0.W.pprint()
instance.Grid.P.pprint()
# instance.PV.Pdim.pprint()
# instance.Battery.Edim.pprint()
# instance.Battery.Pdim.pprint()
# instance.Battery.SOC.pprint()
# instance.Pump2.Pe.pprint()
instance.Pump1.Qout.pprint()
# instance.Pump2.Qout.pprint()
instance.Pump1.beta.pprint()
instance.Pump1.alpha.pprint()
instance.goal.pprint()
# instance.Pump2.beta.pprint()


