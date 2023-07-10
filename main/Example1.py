"""
    AGISTIN - EXAMPLE 1
    
    Optimization usage example of two reservoirs, considering irragtion consumption,
    and two pumps connected to the electrical grid.
    
    Authors: Sergi Costa Dilmé (CITCEA-UPC), Juan Carlos Olives-Camps (CITCEA-UPC), Paula Muñoz Peña (CITCEA-UPC)
"""

# Import pyomo
import pyomo.environ as pyo
from pyomo.network import *

# Import devices
from Devices.Reservoirs import Reservoir
from Devices.Sources import Source
from Devices.Pipes import Pipe
from Devices.Pumps import Pump
from Devices.MainGrid import Grid


# model
m = pyo.ConcreteModel()


# time
l_t = list(range(5))
m.t = pyo.Set(initialize=l_t)

# electricity cost
l_cost = [10,5,1,5,10]
m.cost = pyo.Param(m.t, initialize=l_cost)


# ===== Create the system =====

m.Source0 = pyo.Block()
m.Reservoir1 = pyo.Block()
m.Reservoir0 = pyo.Block()
m.Irrigation1 = pyo.Block()
m.Pump1 = pyo.Block()
m.Pump2 = pyo.Block()
m.Pipe1 = pyo.Block()
m.Grid = pyo.Block()


data_smain = {'Q':[0,0,0,0,0]}
Source(m.Source0, m.t, data_smain)

data_irr = {'Q':[2,1,1,1,1]} # irrigation
Source(m.Irrigation1, m.t, data_irr)

data_res = {'W0':5, 'Wmin':0, 'Wmax':20} # reservoirs (both equal)
init_res = {'Q':[0,0,0,0,0], 'W':[5,5,5,5,5]}
Reservoir(m.Reservoir1, m.t, data_res, init_res)
Reservoir(m.Reservoir0, m.t, data_res, init_res)

data_c1 = {'H0':20, 'K':0.05, 'Qmax':50} # canal
init_c1 = {'Q':[0,0,0,0,0], 'H':[20,20,20,20,20]}
Pipe(m.Pipe1, m.t, data_c1, init_c1)

data_p = {'A':50, 'B':0.1, 'n_n':1450, 'eff':0.9, 'Qmax':20, 'Qnom':5, 'Pmax':9810*50*20} # pumps (both equal)
init_p = {'Q':[0,0,0,0,0], 'H':[20,20,20,20,20], 'n':[1450,1450,1450,1450,1450], 'Pe':[9810*5*20,9810*5*20,9810*5*20,9810*5*20,9810*5*20]}
Pump(m.Pump1, m.t, data_p, init_p)
Pump(m.Pump2, m.t, data_p, init_p)

Grid(m.Grid, m.t, {'P_max':100e3}) # grid



# Connections
m.p1r0 = Arc(ports=(m.Pump1.port_Qin, m.Reservoir0.port_Qout), directed=True)
m.p1c1_Q = Arc(ports=(m.Pump1.port_Qout, m.Pipe1.port_Q), directed=True)
m.p1c1_H = Arc(ports=(m.Pump1.port_H, m.Pipe1.port_H), directed=True)
m.p2r0 = Arc(ports=(m.Pump2.port_Qin, m.Reservoir0.port_Qout), directed=True)
m.p2c1_Q = Arc(ports=(m.Pump2.port_Qout, m.Pipe1.port_Q), directed=True)
m.p2c1_H = Arc(ports=(m.Pump2.port_H, m.Pipe1.port_H), directed=True)
m.c1r1 = Arc(ports=(m.Pipe1.port_Q, m.Reservoir1.port_Qin), directed=True)
m.r0s0 = Arc(ports=(m.Source0.port_Qout, m.Reservoir0.port_Qin), directed=True)
m.r1i1 = Arc(ports=(m.Irrigation1.port_Qin, m.Reservoir1.port_Qout), directed=True)
m.gridp1 = Arc(ports=(m.Pump1.port_P, m.Grid.port_P), directed=True)
m.gridp2 = Arc(ports=(m.Pump2.port_P, m.Grid.port_P), directed=True)

pyo.TransformationFactory("network.expand_arcs").apply_to(m) # apply arcs to model


#%% RUN THE OPTIMIZATION

# Objective function
def obj_fun(m):
	return sum(m.Grid.P[t]*m.cost[t] for t in l_t)
m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)

instance = m.create_instance()
solver = pyo.SolverFactory('ipopt')
solver.solve(instance, tee=False)

instance.Reservoir1.W.pprint()
instance.Reservoir0.W.pprint()
instance.Grid.P.pprint()

# RESULTS
    # W : Size=5, Index=t
    #     Key : Lower : Value              : Upper : Fixed : Stale : Domain
    #       0 :     0 : 2.9999999999999103 :    20 : False : False : NonNegativeReals
    #       1 :     0 : 2.2707626582039224 :    20 : False : False : NonNegativeReals
    #       2 :     0 : 1.7292373317963907 :    20 : False : False : NonNegativeReals
    #       3 :     0 : 0.9999999900000922 :    20 : False : False : NonNegativeReals
    #       4 :     0 :                0.0 :    20 : False : False : NonNegativeReals
    # W : Size=5, Index=t
    #     Key : Lower : Value             : Upper : Fixed : Stale : Domain
    #       0 :     0 :  5.00000000000009 :    20 : False : False : NonNegativeReals
    #       1 :     0 : 4.729237341796077 :    20 : False : False : NonNegativeReals
    #       2 :     0 : 4.270762668203609 :    20 : False : False : NonNegativeReals
    #       3 :     0 : 4.000000009999908 :    20 : False : False : NonNegativeReals
    #       4 :     0 : 4.000000009999998 :    20 : False : False : NonNegativeReals
    # P : Size=5, Index=t
    #     Key : Lower     : Value                   : Upper    : Fixed : Stale : Domain
    #       0 : -100000.0 : -1.9496166613012406e-08 : 100000.0 : False : False :  Reals
    #       1 : -100000.0 :       59037.07788286004 : 100000.0 : False : False :  Reals
    #       2 : -100000.0 :                100000.0 : 100000.0 : False : False :  Reals
    #       3 : -100000.0 :       59037.07788279228 : 100000.0 : False : False :  Reals
    #       4 : -100000.0 :  -1.958746603332127e-08 : 100000.0 : False : False :  Reals