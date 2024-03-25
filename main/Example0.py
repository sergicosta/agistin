"""
    AGISTIN - EXAMPLE 0
    
    Basic usage example of two reservoirs connected by a waterflow source
    
    Authors: Sergi Costa Dilmé (CITCEA-UPC), Juan Carlos Olives-Camps (CITCEA-UPC), Paula Muñoz Peña (CITCEA-UPC)
"""

# Import pyomo
import pyomo.environ as pyo
from pyomo.network import Arc, Port

# Import devices
from Devices.Reservoirs import Reservoir_Ex0
from Devices.Sources import Source

# Import useful functions
from Utilities import clear_clc

#clean console and variable pane
clear_clc() #consider removing if you are not working with Spyder

# model
m = pyo.ConcreteModel()


# time
l_t = list(range(5))
m.t = pyo.Set(initialize=l_t)


# ===== Create the system =====

# Source 1
m.Source1 = pyo.Block()
data_s1 = {'Q':[1,1,1,1,1]}
Source(m.Source1, m.t, data_s1)

# Reservoir0
m.Reservoir0 = pyo.Block()
data_r0 = {'W0':10, 'Wmin':0, 'Wmax':20}
init_r0 = {'Q':[0,0,0,0,0], 'W':[5,5,5,5,5]}
Reservoir_Ex0(m.Reservoir0, m.t, data_r0, init_r0)

# Reservoir1
m.Reservoir1 = pyo.Block()
data_r1 = {'W0':5, 'Wmin':0, 'Wmax':20}
init_r1 = {'Q':[0,0,0,0,0], 'W':[5,5,5,5,5]}
Reservoir_Ex0(m.Reservoir1, m.t, data_r1, init_r1)


# Connections
m.s1r0 = Arc(ports=(m.Source1.port_Qin, m.Reservoir0.port_Q), directed=True)
m.s1r1 = Arc(ports=(m.Source1.port_Qout, m.Reservoir1.port_Q), directed=True)

pyo.TransformationFactory("network.expand_arcs").apply_to(m) # apply arcs to model

# Print variables and constraints:
# m.pprint()



#%% RUN THE OPTIMIZATION

# Objective function (Feasibility problem in this example)
def obj_fun(m):
	return 0
m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)

instance = m.create_instance()
solver = pyo.SolverFactory('ipopt')
solver.solve(instance, tee=False)

instance.Reservoir1.W.pprint()
instance.Reservoir0.W.pprint()

# RESULTS:
    # W : Size=5, Index=t
    #     Key : Lower : Value : Upper : Fixed : Stale : Domain
    #       0 :     0 :   6.0 :    20 : False : False : NonNegativeReals
    #       1 :     0 :   7.0 :    20 : False : False : NonNegativeReals
    #       2 :     0 :   8.0 :    20 : False : False : NonNegativeReals
    #       3 :     0 :   9.0 :    20 : False : False : NonNegativeReals
    #       4 :     0 :  10.0 :    20 : False : False : NonNegativeReals
    # W : Size=5, Index=t
    #     Key : Lower : Value : Upper : Fixed : Stale : Domain
    #       0 :     0 :   9.0 :    20 : False : False : NonNegativeReals
    #       1 :     0 :   8.0 :    20 : False : False : NonNegativeReals
    #       2 :     0 :   7.0 :    20 : False : False : NonNegativeReals
    #       3 :     0 :   6.0 :    20 : False : False : NonNegativeReals
    #       4 :     0 :   5.0 :    20 : False : False : NonNegativeReals