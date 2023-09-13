"""
    AGISTIN - EXAMPLE 3
    
    Optimization usage example of the data parser. It considers the same system as
    in Example 2, but reads the data from an external spreadsheet file.
    
    Authors: Sergi Costa Dilmé (CITCEA-UPC), Juan Carlos Olives-Camps (CITCEA-UPC), Paula Muñoz Peña (CITCEA-UPC)
"""

# Import pyomo
import pyomo.environ as pyo
from pyomo.network import *

# Import builder
from Builder import data_parser, builder

# Import devices
from Devices.Reservoirs import Reservoir_Ex0
from Devices.Sources import Source
from Devices.Pipes import Pipe_Ex0
from Devices.Pumps import Pump
from Devices.EB import EB
from Devices.SolarPV import SolarPV
from Devices.MainGrid import Grid


# generate system json file
data_parser("Test1")


m = pyo.ConcreteModel()

# time
l_t = list(range(5))
m.t = pyo.Set(initialize=l_t)

# electricity cost
l_cost = [10,5,1,5,10]
m.cost = pyo.Param(m.t, initialize=l_cost)
cost_new_pv = 2, 10

builder(m,"Test1")


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