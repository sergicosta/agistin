"""
    AGISTIN - EXAMPLE 1
    
    Optimization usage example of two reservoirs, considering irragtion consumption,
    and two pumps connected to the electrical grid.
        
    Authors: Sergi Costa Dilmé (CITCEA-UPC), Juan Carlos Olives-Camps (CITCEA-UPC), Paula Muñoz Peña (CITCEA-UPC)
"""

# Import pyomo
import pyomo.environ as pyo
from pyomo.network import Arc, Port

# Import devices
from Devices.Reservoirs import Reservoir_Ex0
from Devices.Sources import Source
from Devices.Pipes import Pipe_Ex0
from Devices.Pumps import ReversiblePump, Pump
from Devices.MainGrid import Grid
from Devices.EB import EB
from Devices.Turbines import Turbine

# Import useful functions
from Utilities import clear_clc, model_to_file

#clean console and variable pane
#clear_clc() #consider removing if you are not working with Spyder

# model
m = pyo.ConcreteModel()

# time
l_t = list(range(5))
m.t = pyo.Set(initialize=l_t)

# electricity cost
l_cost = [10,5,1,5,10]
m.cost = pyo.Param(m.t, initialize=l_cost)

# ===== Create the system =====
m.Reservoir1 = pyo.Block()
m.Reservoir0 = pyo.Block()
m.Irrigation1 = pyo.Block()
m.Pump1 = pyo.Block()
m.Pump2 = pyo.Block()
# m.Turb1 = pyo.Block()
m.Pipe1 = pyo.Block()
m.Grid = pyo.Block()
m.EB = pyo.Block()



data_irr = {'Q':[2,1,0,0,0]} # irrigation
Source(m.Irrigation1, m.t, data_irr, None)

data_res = {'W0':10, 'Wmin':0, 'Wmax':20} # reservoirs (both equal)
init_res = {'Q':[0,0,0,0,0], 'W':[5,5,5,5,5]}
Reservoir_Ex0(m.Reservoir1, m.t, data_res, init_res)
data_res = {'W0':10, 'Wmin':0, 'Wmax':20} # reservoirs (both equal)
init_res = {'Q':[0,0,0,0,0], 'W':[5,5,5,5,5]}
Reservoir_Ex0(m.Reservoir0, m.t, data_res, init_res)

data_c1 = {'H0':20, 'K':0.05, 'Qmax':50} # canal
init_c1 = {'Q':[0,0,0,0,0], 'H':[20,20,20,20,20]}
Pipe_Ex0(m.Pipe1, m.t, data_c1, init_c1)

data_p = {'A':50, 'B':0.1, 'n_n':1450, 'eff':0.9, 'Qmax':20, 'Qnom':5, 'Pmax':9810*50*20} # pumps (both equal)
init_p = {'Q':[0,0,0,0,0], 'H':[20,20,20,20,20], 'n':[1450,1450,1450,1450,1450], 'Pe':[9810*5*20,9810*5*20,9810*5*20,9810*5*20,9810*5*20]}
# Pump(m.Pump1, m.t, data_p, init_p)

data_p = {'A':50, 'B':0.1, 'n_n':1450, 'eff':0.9, 'eff_t':0.5, 'Qmax':20, 'Qnom':5, 'Pmax':9810*50*20, 'S':0.05, 'H0':[20, 20, 20, 20, 20]} # pumps (both equal)
init_p = {'Q':[1e-6,1e-6,1e-6,1e-6,1e-6], 'H':[20,20,20,20,20], 'n':[1450,1450,1450,1450,1450], 'Pe':[9810*5*20,9810*5*20,9810*5*20,9810*5*20,9810*5*20]}
ReversiblePump(m.Pump1, m.t, data_p, init_p)
ReversiblePump(m.Pump2, m.t, data_p, init_p)

data_t = {'eff':0.8}
init_t = {'Q':[3,1,0,0,0], 'H':[20,20,20,20,20], 'Pe':[-9810*20*1,-9810*20*1,-9810*20*1,-9810*20*1,-9810*20*1]}
# Turbine(m.Turb1, m.t, data_t, init_t)

Grid(m.Grid, m.t, {'Pmax':100e6}, None) # grid
EB(m.EB, m.t, None, None) # node

# Connections

# m.p1r0 = Arc(ports=(m.Turb1.port_Qout, m.Reservoir0.port_Q), directed=True)
# m.p1c1_Q = Arc(ports=(m.Turb1.port_Qin, m.Pipe1.port_Q), directed=True)
# m.p1c1_H = Arc(ports=(m.Turb1.port_H, m.Pipe1.port_H), directed=True)
# m.EBp1 = Arc(ports=(m.Turb1.port_P, m.EB.port_P), directed=True)

m.p1r0 = Arc(ports=(m.Pump1.port_Qin, m.Reservoir0.port_Q), directed=True)
m.p1c1_Q = Arc(ports=(m.Pump1.port_Qout, m.Pipe1.port_Q), directed=True)
m.p1c1_H = Arc(ports=(m.Pump1.port_H, m.Pipe1.port_H), directed=True)
m.p2r0 = Arc(ports=(m.Pump2.port_Qin, m.Reservoir0.port_Q), directed=True)
m.p2c1_Q = Arc(ports=(m.Pump2.port_Qout, m.Pipe1.port_Q), directed=True)
m.p2c1_H = Arc(ports=(m.Pump2.port_H, m.Pipe1.port_H), directed=True)
m.c1r1 = Arc(ports=(m.Pipe1.port_Q, m.Reservoir1.port_Q), directed=True)
m.r1i1 = Arc(ports=(m.Irrigation1.port_Qin, m.Reservoir1.port_Q), directed=True)
m.EBp1 = Arc(ports=(m.Pump1.port_P, m.EB.port_P), directed=True)
m.EBp2 = Arc(ports=(m.Pump2.port_P, m.EB.port_P), directed=True)
m.EBgrid = Arc(ports=(m.Grid.port_P, m.EB.port_P), directed=True)

pyo.TransformationFactory("network.expand_arcs").apply_to(m) # apply arcs to model


#%% RUN THE OPTIMIZATION

# Objective function
def obj_fun(m):
	return sum(-m.Grid.P[t]*m.cost[t] for t in l_t)
m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)

instance = m.create_instance()
# solver = pyo.SolverFactory('asl:couenne')
# solver.solve(instance, tee=True)



import os
os.environ['NEOS_EMAIL'] = 'pau.garcia.motilla@upc.edu'
opt = pyo.SolverFactory("knitro")
solver_manager = pyo.SolverManagerFactory('neos')
results = solver_manager.solve(instance, opt=opt)


instance.Reservoir1.W.pprint()
instance.Reservoir1.Q.pprint()
instance.Reservoir0.W.pprint()
instance.Reservoir0.Q.pprint()
instance.Grid.P.pprint()
instance.Pump1.ModePump.pprint()
instance.Pump1.Qoutt.pprint()
instance.Pump1.Qoutp.pprint()
instance.Pump1.Pe.pprint()
instance.Pipe1.signQ.pprint()
instance.Pipe1.Q.pprint()
# instance.Turb1.Qout.pprint()

model_to_file(instance, 'ExampleRev.txt')

# RESULTS
    # W : Size=5, Index=t
    #     Key : Lower : Value              : Upper : Fixed : Stale : Domain
    #       0 :     0 :  2.999999999999957 :    20 : False : False : NonNegativeReals
    #       1 :     0 :  2.270762658203932 :    20 : False : False : NonNegativeReals
    #       2 :     0 : 1.7292373317963943 :    20 : False : False : NonNegativeReals
    #       3 :     0 : 0.9999999900000451 :    20 : False : False : NonNegativeReals
    #       4 :     0 :                0.0 :    20 : False : False : NonNegativeReals
    # W : Size=5, Index=t
    #     Key : Lower : Value             : Upper : Fixed : Stale : Domain
    #       0 :     0 : 5.000000000000043 :    20 : False : False : NonNegativeReals
    #       1 :     0 : 4.729237341796067 :    20 : False : False : NonNegativeReals
    #       2 :     0 : 4.270762668203605 :    20 : False : False : NonNegativeReals
    #       3 :     0 : 4.000000009999955 :    20 : False : False : NonNegativeReals
    #       4 :     0 : 4.000000009999997 :    20 : False : False : NonNegativeReals
    # P : Size=5, Index=t
    #     Key : Lower     : Value                 : Upper    : Fixed : Stale : Domain
    #       0 : -100000.0 :  9.31716873473775e-09 : 100000.0 : False : False :  Reals
    #       1 : -100000.0 :    -59037.07788285201 : 100000.0 : False : False :  Reals
    #       2 : -100000.0 :             -100000.0 : 100000.0 : False : False :  Reals
    #       3 : -100000.0 :   -59037.077882781225 : 100000.0 : False : False :  Reals
    #       4 : -100000.0 : 9.317168818435555e-09 : 100000.0 : False : False :  Reals