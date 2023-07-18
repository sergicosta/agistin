"""
AGISTIN project 

.\Builder.py

Builder functions generate un complete pyomo model from a .json file.
"""

import pyomo.environ as pyo
from pyomo.network import *
import json
    
def builder(m):
  
    from Devices.EB import EB
    from Devices.MainGrid import Grid
    from Devices.Pipes import Pipe
    from Devices.Pumps import Pump
    from Devices.Reservoirs import Reservoir
    from Devices.SolarPV import SolarPV
    from Devices.Sources import Source

    with open('Cases\TestCase.json', 'r') as jfile:
        system = json.load(jfile)

    for it in list(system.keys()):
        setattr(m, it, pyo.Block())
    
    for it in list(system.keys()):
        s = system[it]['data']['type']
        create = locals()[s]
        create(getattr(m, it), m.t, system[it]['data'], system[it]['init_data'])
    
    val = 0
    
    for it in list(system.keys()):
        for j in list(system[it]['conns'].keys()):
            setattr(m, f'arc_{val}', Arc(ports=(getattr(getattr(m, it), f'port_{j}'), getattr(getattr(m, system[it]['conns'][j][0]), f'port_{system[it]["conns"][j][1]}')), directed=True))
            val += 1
    
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)
    
    # return m



m = pyo.ConcreteModel()

# time
l_t = list(range(5))
m.t = pyo.Set(initialize=l_t)

# electricity cost
l_cost = [10,5,1,5,10]
m.cost = pyo.Param(m.t, initialize=l_cost)
cost_new_pv = 10

builder(m)

def obj_fun(m):
 	return sum(-m.MainGrid.P[t]*m.cost[t] for t in l_t) + m.PV.Pdim*cost_new_pv
m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)

instance = m.create_instance()
solver = pyo.SolverFactory('ipopt')
solver.solve(instance, tee=False)

instance.Reservoir0.W.pprint()
instance.Reservoir1.W.pprint()
instance.MainGrid.P.pprint()
instance.PV.Pdim.pprint()