"""
AGISTIN project 

.\Devices\Reservoirs.py

Builder functions generate un complete pyomo model from a .json file.
"""

import pyomo.environ as pyo
from pyomo.network import *
from Devices.EB import EB
from Devices.MainGrid import Grid
from Devices.Pipes import Pipe
from Devices.Pumps import Pump
from Devices.Reservoirs import Reservoir
from Devices.Sources import Source
import json

with open('Cases\TestCase.json', 'r') as jfile:
    system = json.load(jfile)

m = pyo.ConcreteModel()

# time
l_t = list(range(5))
m.t = pyo.Set(initialize=l_t)

# electricity cost
l_cost = [10,5,1,5,10]
m.cost = pyo.Param(m.t, initialize=l_cost)

for it in list(system.keys()):
    setattr(m, it, pyo.Block())

for it in list(system.keys()):
    s = system[it]['data']['type']
    create = locals()[s]
    create(getattr(m, it), m.t, system[it]['data'], system[it]['init_data'])

val = 0

for it in list(system.keys()):
    
    if system[it]['data']['type'] == 'Grid':
        for conn in system[it]['to']:
            setattr(m, f'arc_{val}', Arc(ports=(getattr(getattr(m, conn), 'port_P'), getattr(getattr(m, it), 'port_P')), directed=True))
            val += 1
    
    if system[it]['data']['type'] == 'Pipe':
        for conn in system[it]['to']:
            setattr(m, f'arc_{val}', Arc(ports=(getattr(getattr(m, it), 'port_Q'), getattr(getattr(m, conn), 'port_Qin')), directed=True))
            val += 1
    
    if system[it]['data']['type'] == 'Pump':
        for conn in system[it]['to']:
            setattr(m, f'arc_{val}Q', Arc(ports=(getattr(getattr(m, it), 'port_Qout'), getattr(getattr(m, conn), 'port_Q')), directed=True))
            setattr(m, f'arc_{val}H', Arc(ports=(getattr(getattr(m, it), 'port_H'   ), getattr(getattr(m, conn), 'port_H')), directed=True))
            val += 1
    
    if system[it]['data']['type'] == 'Reservoir':
        for conn in system[it]['to']:
            setattr(m, f'arc_{val}', Arc(ports=(getattr(getattr(m, conn), 'port_Qin'), getattr(getattr(m, it), 'port_Qout')), directed=True))
            val += 1
    
    if system[it]['data']['type'] == 'Source':
        for conn in system[it]['to']:
            setattr(m, f'arc_{val}', Arc(ports=(getattr(getattr(m, it), 'port_Qout'), getattr(getattr(m, conn), 'port_Qin')), directed=True))
            val += 1

pyo.TransformationFactory("network.expand_arcs").apply_to(m)

def obj_fun(m):
	return sum(m.MainGrid.P[t]*m.cost[t] for t in l_t)
m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)

instance = m.create_instance()
solver = pyo.SolverFactory('ipopt')
solver.solve(instance, tee=False)

instance.Reservoir1.W.pprint()
instance.Reservoir2.W.pprint()
instance.MainGrid.P.pprint()