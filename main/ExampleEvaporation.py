"""
    AGISTIN - EXAMPLE 1
    
    Optimization usage example of two reservoirs, considering irragtion consumption,
    one pumop connected to the electrical grid and considering evaporation and raining
    in the reservoirs.
    
    Authors: Sergi Costa Dilmé (CITCEA-UPC), Juan Carlos Olives-Camps (CITCEA-UPC), Paula Muñoz Peña (CITCEA-UPC)
"""

# Import pyomo
import pyomo.environ as pyo
from pyomo.network import Arc, Port

# Import devices
from Devices.Reservoirs import Reservoir_Ex0
from Devices.Sources import Source
from Devices.Pipes import Pipe_Ex0
from Devices.Pumps import Pump
from Devices.MainGrid import Grid
from Devices.EB import EB
from Devices.Evaporation import Evaporation

# Import useful functions
from Utilities import clear_clc

#clean console and variable pane
#clear_clc() #consider removing if you are not working with Spyder

# model
m = pyo.ConcreteModel()

# time
l_t = list(range(6))
m.t = pyo.Set(initialize=l_t)

# electricity cost
l_cost = [10,5,1,1,5,10]
m.cost = pyo.Param(m.t, initialize=l_cost)

# ===== Create the system =====
m.Reservoir1 = pyo.Block()
m.Reservoir0 = pyo.Block()
m.Irrigation1 = pyo.Block()
m.Pump1 = pyo.Block()
m.Pipe1 = pyo.Block()
m.Grid = pyo.Block()
m.EB = pyo.Block()
m.Evap = pyo.Block()


data_irr = {'Q':[2,0,0,1,1,1]} # irrigation
Source(m.Irrigation1, m.t, data_irr, None)

data_res = {'W0':10, 'Wmin':1, 'Wmax':20} # reservoirs (both equal)
init_res = {'Q':[0,0,0,0,0,0], 'W':[5,5,5,5,5,5]}
Reservoir_Ex0(m.Reservoir1, m.t, data_res, init_res)
Reservoir_Ex0(m.Reservoir0, m.t, data_res, init_res)

data_c1 = {'H0':20, 'K':0.05, 'Qmax':50} # canal
init_c1 = {'Q':[0,0,0,0,0,0], 'H':[20,20,20,20,20,20]}
Pipe_Ex0(m.Pipe1, m.t, data_c1, init_c1)

data_p = {'A':50, 'B':0.1, 'n_n':1450, 'eff':0.9, 'Qmax':20, 'Qnom':5, 'Pmax':9810*50*20} # pumps (both equal)
init_p = {'Q':[0,0,0,0,0,0], 'H':[20,20,20,20,20,20], 'n':[1450,1450,1450,1450,1450,1450], 'Pe':[9810*5*20,9810*5*20,9810*5*20,9810*5*20,9810*5*20,9810*5*20]}
Pump(m.Pump1, m.t, data_p, init_p)

Grid(m.Grid, m.t, {'Pmax':100e3}, None) # grid
EB(m.EB, m.t, None, None) # node

data_evap = {'Wind':[2,2,2,2,2,2],'Temperature':[12,13,14,15,14,14],'Radiation':[300,500,600,700,600,500],
             'Pressure':[1028,1028,1028,1028,1028,1028],'Humitat':[78,76,74,72,74,77],
             'Rain':[0,0,3,0,0,0],'Area':100,'Latent':2256/1000,'hsum':6}
init_evap = {'Q':[0,0,0,0,0,0,]}
Evaporation(m.Evap, m.t,data_evap,init_evap)


# Connections
m.p1r0 = Arc(ports=(m.Pump1.port_Qin, m.Reservoir0.port_Q), directed=True)
m.p1c1_Q = Arc(ports=(m.Pump1.port_Qout, m.Pipe1.port_Q), directed=True)
m.p1c1_H = Arc(ports=(m.Pump1.port_H, m.Pipe1.port_H), directed=True)
m.c1r1 = Arc(ports=(m.Pipe1.port_Q, m.Reservoir1.port_Q), directed=True)
m.r1i1 = Arc(ports=(m.Irrigation1.port_Qin, m.Reservoir1.port_Q), directed=True)
m.EBp1 = Arc(ports=(m.Pump1.port_P, m.EB.port_P), directed=True)
m.EBgrid = Arc(ports=(m.Grid.port_P, m.EB.port_P), directed=True)
m.evapr0 = Arc(ports=(m.Evap.port_Qout,m.Reservoir0.port_Q), directed=True)


pyo.TransformationFactory("network.expand_arcs").apply_to(m) # apply arcs to model


#%% RUN THE OPTIMIZATION

# Objective function
def obj_fun(m):
	return sum(-m.Grid.P[t]*m.cost[t] for t in l_t)
m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)

instance = m.create_instance()
solver = pyo.SolverFactory('asl:couenne')
solver.solve(instance, tee=True)

instance.Reservoir1.W.pprint()
instance.Reservoir0.W.pprint()
instance.Evap.Qout.pprint()
