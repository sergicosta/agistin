"""
AGISTIN project

.\Builder.py

Builder functions generate a complete pyomo model from a .json file.
"""

import pyomo.environ as pyo
from pyomo.network import *
import json
    
def builder(m, test_case):
  
    from Devices.EB import EB
    from Devices.HydroSwitch import HydroSwitch
    from Devices.MainGrid import Grid
    from Devices.NewPumps import NewPump
    from Devices.Pipes import Pipe
    from Devices.Pumps import Pump
    from Devices.Reservoirs import Reservoir
    from Devices.SolarPV import SolarPV
    from Devices.Sources import Source
    from Devices.Switch import Switch
    from Devices.Turbines import Turbine

    with open(f'Cases\{test_case}.json', 'r') as jfile:
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


if __name__ == '__main__':

    m = pyo.ConcreteModel()
    
    # time
    l_t = list(range(5))
    m.t = pyo.Set(initialize=l_t)
    
    # electricity cost
    l_cost = [1,1,1,1,1]
    m.cost = pyo.Param(m.t, initialize=l_cost)
    cost_new_turb, cost_new_pump = 2, 10
    
    builder(m,'TestAll')
    
    def obj_fun(m):
     	return sum(-m.MainGrid.P[t]*m.cost[t] for t in l_t) + m.Turb1.Pdim*cost_new_turb + m.PumpNew.Pdim*cost_new_pump
    m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)
    
    # with open('model','w') as f:
    #     m.pprint(f)
    
    instance = m.create_instance()
    solver = pyo.SolverFactory('ipopt')
    solver.solve(instance, tee=False)
    
    # instance.Turb1.Qin.pprint()
    # instance.Pump1.Qout.pprint()
    # instance.PumpNew.Qout.pprint()
    print('------------------------- Reservoirs -------------------------')
    instance.Reservoir0.W.pprint()
    instance.Reservoir1.W.pprint()
    print('--------------------------- Powers ---------------------------')
    instance.Turb1.Pe.pprint()
    instance.Pump1.Pe.pprint()
    instance.PumpNew.Pe.pprint()
    instance.PV1.P.pprint()
    instance.MainGrid.P.pprint()
    print('--------------------------- Sizing ---------------------------')
    instance.Turb1.Pdim.pprint()
    instance.PumpNew.Pdim.pprint()