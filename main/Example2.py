"""
    AGISTIN - EXAMPLE 2
    
    Optimization usage example of two reservoirs, considering irrigation consumption,
    and two pumps. The pumping station gathers the pumps' power consumption, a solar
    PV plant and a connection point to the public grid.
    The PV plant can be upgraded to double its size (from 50e3 to 100e3) with a cost of 10 per power unit.
    The price of selling power to the grid is half of the cost of buying it.
    
    Authors: Sergi Costa Dilmé (CITCEA-UPC), Juan Carlos Olives-Camps (CITCEA-UPC), Paula Muñoz Peña (CITCEA-UPC)
"""

# Import pyomo
import pyomo.environ as pyo
from pyomo.network import Arc

# Import devices
from Devices.Reservoirs import Reservoir_Ex0
from Devices.Sources import Source
from Devices.Pipes import Pipe_Ex0
from Devices.Pumps import Pump
from Devices.EB import EB
from Devices.SolarPV import SolarPV
from Devices.MainGrid import Grid

# Import useful functions
from Utilities import clear_clc

#clean console and variable pane
clear_clc() #consider removing if you are not working with Spyder

# model
m = pyo.ConcreteModel()


# time
l_t = list(range(5))
m.t = pyo.Set(initialize=l_t)

# electricity cost
l_cost = [10,5,1,5,10]
m.cost = pyo.Param(m.t, initialize=l_cost)
cost_new_pv = 10


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


data_irr = {'Q':[2,1,1,1,1]} # irrigation
Source(m.Irrigation1, m.t, data_irr, {})

data_res = {'W0':5, 'Wmin':0, 'Wmax':20} # reservoirs (both equal)
init_res = {'Q':[0,0,0,0,0], 'W':[5,5,5,5,5]}
Reservoir_Ex0(m.Reservoir1, m.t, data_res, init_res)
Reservoir_Ex0(m.Reservoir0, m.t, data_res, init_res)

data_c1 = {'H0':20, 'K':0.05, 'Qmax':50} # canal
init_c1 = {'Q':[0,0,0,0,0], 'H':[20,20,20,20,20]}
Pipe_Ex0(m.Pipe1, m.t, data_c1, init_c1)

data_p = {'A':50, 'B':0.1, 'n_n':1450, 'eff':0.9, 'Qmax':20, 'Qnom':5, 'Pmax':9810*50*20} # pumps (both equal)
init_p = {'Q':[0,0,0,0,0], 'H':[20,20,20,20,20], 'n':[1450,1450,1450,1450,1450], 'Pe':[9810*5*20,9810*5*20,9810*5*20,9810*5*20,9810*5*20]}
Pump(m.Pump1, m.t, data_p, init_p)
Pump(m.Pump2, m.t, data_p, init_p)

data_pv = {'Pinst':50e3, 'Pmax':100e3, 'forecast':[0.0,0.2,0.8,1.0,0.1]} # PV
SolarPV(m.PV, m.t, data_pv)

Grid(m.Grid, m.t, {'Pmax':100e3}) # grid

EB(m.EB, m.t)



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

pyo.TransformationFactory("network.expand_arcs").apply_to(m) # apply arcs to model


#%% RUN THE OPTIMIZATION

# Objective function
def obj_fun(m):
	return sum((m.Grid.Pbuy[t]*m.cost[t] - m.Grid.Psell[t]*m.cost[t]/2) for t in l_t ) + m.PV.Pdim*cost_new_pv
m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)

instance = m.create_instance()
solver = pyo.SolverFactory('ipopt')
solver.solve(instance, tee=False)

instance.Reservoir1.W.pprint()
instance.Reservoir0.W.pprint()
instance.Grid.P.pprint()
instance.PV.Pdim.pprint()

# RESULTS
    # W : Size=5, Index=t
    #     Key : Lower : Value              : Upper : Fixed : Stale : Domain
    #       0 :     0 : 3.0000000000000004 :    20 : False : False : NonNegativeReals
    #       1 :     0 :  2.051624535242702 :    20 : False : False : NonNegativeReals
    #       2 :     0 : 1.7161061718196515 :    20 : False : False : NonNegativeReals
    #       3 :     0 : 0.9741875933941936 :    20 : False : False : NonNegativeReals
    #       4 :     0 :                0.0 :    20 : False : False : NonNegativeReals
    # W : Size=5, Index=t
    #     Key : Lower : Value             : Upper : Fixed : Stale : Domain
    #       0 :     0 : 4.999999999999999 :    20 : False : False : NonNegativeReals
    #       1 :     0 : 4.948375464757298 :    20 : False : False : NonNegativeReals
    #       2 :     0 : 4.283893828180348 :    20 : False : False : NonNegativeReals
    #       3 :     0 : 4.025812406605806 :    20 : False : False : NonNegativeReals
    #       4 :     0 : 4.000000009999997 :    20 : False : False : NonNegativeReals
    # P : Size=5, Index=t
    #     Key : Lower     : Value                  : Upper    : Fixed : Stale : Domain
    #       0 : -100000.0 :  9.444173782089441e-09 : 100000.0 : False : False :  Reals
    #       1 : -100000.0 : -8.895566856661031e-09 : 100000.0 : False : False :  Reals
    #       2 : -100000.0 :              -100000.0 : 100000.0 : False : False :  Reals
    #       3 : -100000.0 : -8.894453079458348e-09 : 100000.0 : False : False :  Reals
    #       4 : -100000.0 :  9.520949369890758e-09 : 100000.0 : False : False :  Reals
    # Pdim : Size=1, Index=None
    #     Key  : Lower : Value             : Upper   : Fixed : Stale : Domain
    #     None :     0 : 6271.117831362812 : 50000.0 : False : False : NonNegativeReals