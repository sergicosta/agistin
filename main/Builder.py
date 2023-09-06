"""
AGISTIN project

.\Builder.py

Builder functions generate a complete pyomo model from a .json file.
"""

import pyomo.environ as pyo
from pyomo.network import *
import json
import pandas as pd


def data_parser(NameTest):
    T = 5
    dt = 1
    
    df = pd.read_excel(f'Cases/{NameTest}.xlsx', sheet_name=None)
    
    with open(f'Cases/{NameTest}.json', 'w') as f:
        f.write('{\n')
        for k in df.keys():
            for val in range(len(df[k])):
                f.write(f'"{df[k]["Name"][val]}":{{\n')
                f.write(f'\t "data":{{"type":"{k}",')
                for it in df[k].columns.values:
                    if it in ('Name','CONNECTION'):
                        pass
                    else:
                        f.write(f'"{it}":{df[k][it][val]},')
                if k == 'Reservoir':
                    f.write(f'"dt":{dt}')
                f.write('},\n')
                f.write('\t "init_data":{},\n')
                f.write('\t "conns":{')
                try:
                    con = df[k]['CONNECTION'][val]
                    cons = con.split(';')
                    for aux in cons:
                        if len(aux) == 0:
                            pass
                        else:
                            trp = aux.split(',')
                            f.write(f'"{trp[0]}":["{trp[1]}","{trp[2]}"],')
                except KeyError: # no CONNECTION
                    pass
                except AttributeError: # CONNECTION is NaN
                    pass 
                f.write('}\n')
                f.write('\t },\n')
        f.write('}\n')

 
def builder(m, test_case):
  
    from Devices.EB import EB
    from Devices.HydroSwitch import HydroSwitch
    from Devices.MainGrid import Grid
    from Devices.NewPumps import NewPump
    from Devices.Pipes import Pipe
    from Devices.Pumps import Pump
    from Devices.Reservoirs import Reservoir
    from Devices.Reservoirs import Reservoir_Ex0
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