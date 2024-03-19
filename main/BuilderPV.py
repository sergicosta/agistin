"""
AGISTIN project

.\Builder.py

Builder functions generate a complete pyomo model from a .json file.
"""

import pyomo.environ as pyo
from pyomo.network import Arc, Port
import json
import pandas as pd

def write_list(f, df, df_time, k, val):
    """
    Writes to the file f (type json), which is being created, a list of 
    initialization values for the element "val" of type "k".
    
    Inputs:

    :param f: .json file being created
    :param df: dataframe regarding static data
    :param df_time: dataframe regarding initializing data
    :param k: defines the type of element (not the name of the element) [key]
    :param val: counter to differentiate elements of the same type
    
    Outputs:
        - None. The file is updated.

    """
    first = True
    for it in df_time[k]:
        aux = it.split('_')
        if aux[0] == df[k]["Name"][val]:
            if first:
                f.write(f'"{aux[1]}":{list(df_time[k][it])}')
                first = False
            else:
                f.write(f',"{aux[1]}":{list(df_time[k][it])}')


def data_parser(NameTest, dt):
    """
    Converting several excel files, with static data and time-based datasets,
    into a .json format file, which is called by the builder.
    
    data_parser requieres the following:
        
    :param NameTest: name of the excel file with the static information of the plant ``str``
    :param dt: time interval for numerical integration :math:`\Delta t` ``int``
    
    It is required that 3 excel files exists:
        - 'NameTest.xlsx': plant parameters
        - 'NameTest_time.xlsx': devices initialization variables for optimization solver
        - 'NameTest_cost.xlsx': economic cost values for cost function formulation
    
    The output .json file:
        - 'NameTest.json': containing `data`, `init_data`, and `conns` of each device.

    """
    df = pd.read_excel(f'Cases/{NameTest}.xlsx', sheet_name=None)
    df_time = pd.read_excel(f'Cases/{NameTest}_time.xlsx', sheet_name=None)
    df_cost = pd.read_excel(f'Cases/{NameTest}_cost.xlsx', sheet_name=None)
    special = ['SolarPV','Source']
    T = df_time['Reservoir'].shape[0]
    
    with open(f'Cases/{NameTest}.json', 'w') as f:
        first = True
        f.write('{\n')
        for k in df.keys(): # type of element
            for val in range(len(df[k])): # for each element of type k
                if first:
                    first = False
                else:
                    f.write(',\n')
                f.write(f'"{df[k]["Name"][val]}":{{\n')
                f.write(f'\t "data":{{"type":"{k}"')
                for it in df[k].columns.values: # for each characteristic of val
                    if it in ('Name','CONNECTION'):
                        pass
                    else:
                        f.write(f',"{it}":{df[k][it][val]}')
                        
                if k in ('Reservoir'): # Elements that have constraints modelled as differential equations
                    f.write(f',"dt":{dt}')
                if k in special: # Elements with parameters that change during the simulation
                    f.write(',')
                    write_list(f, df, df_time, k, val)
                f.write('},\n')
                # Initialization values for decision variables
                f.write('\t "init_data":{')
                if k not in special:
                    write_list(f, df, df_time, k, val)
                f.write('},\n')
                #  CONNECTIONS
                f.write('\t "conns":{')
                try:
                    con = df[k]['CONNECTION'][val]
                    cons = con.split(';')
                    for aux in cons:
                        if len(aux) == 0:
                            pass
                        else:
                            trp = aux.split(',')
                            f.write(f'"{trp[0]}":["{trp[1]}","{trp[2]}"]')
                            if aux != cons[-2]:
                                f.write(',')
                except KeyError: # no CONNECTION
                    pass
                except AttributeError: # CONNECTION is NaN
                    pass 
                f.write('}\n')
                f.write('\t }')
        f.write('\n}\n')
        
    with open(f'Cases/{NameTest}_cost.json', 'w') as f:
        first = True
        f.write('{')
        for k in df_cost.keys():
            if not df_cost[k].empty:
                for it in df_cost[k]:
                    if first:
                        f.write(f'\n"{it}":{list(df_cost[k][it])}')
                        first = False
                    else:
                        f.write(f',\n"{it}":{list(df_cost[k][it])}')
        f.write('\n}\n')
        
    return T 

 
def builder(m, test_case):
    """
    Generate a complete pyomo model from a .json file.
    It provides a flexible solution to create an optimization problem in a pyomo environment using object-oriented programming.
    
    builder requieres the following:
           
    :param m: concrete pyomo model ``pyomo.core.base.PyomoModel.ConcreteModel``
    :param test_case: must match the name of the .json source file ``str``
    
    """
    

    from Devices.EB import EB
    #from Devices.HydroSwitch import HydroSwitch
    from Devices.MainGrid import Grid
    #from Devices.NewPumps import NewPump
    #from Devices.Pipes import Pipe
    #from Devices.Pipes import Pipe_Ex0
    #from Devices.Pumps import Pump
    #from Devices.Reservoirs import Reservoir
    #from Devices.Reservoirs import Reservoir_Ex0
    from Devices.SolarPV import SolarPV
    #from Devices.Sources import Source
    #from Devices.Switch import Switch
    #from Devices.Turbines import Turbine
    #from Devices.Batteries import Battery
    #from Devices.Batteries import Battery_MV

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
    
    with open(f'Cases\{test_case}_cost.json', 'r') as jfile:
        cost = json.load(jfile)
        
    for it in cost.keys():
        setattr(m, it, cost[it])


def run(name, dt):
    
    m = pyo.ConcreteModel()
    T = data_parser(name, dt)
    m.t = pyo.Set(initialize=list(range(T)))
    
    builder(m, name)
    """
    def obj_fun(m):
        return sum(-m.MainGrid.P[t]*m.cost_MainGrid[t] for t in list(range(T))) + m.Turb1.Pdim*m.cost_Turb1[0] + m.PumpNew.Pdim*m.cost_PumpNew[0]
    m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)
    
    instance = m.create_instance()
    solver = pyo.SolverFactory('ipopt')
    solver.options['tol'] = 1e-6
    # print(solver.options['tol'])
    solver.solve(instance, tee=False)
    
    return instance
"""


# if __name__ == '__main__':

#     # electricity cost
#     l_cost = [1,1,1,1,1]
#     m.cost = pyo.Param(m.t, initialize=l_cost)
#     cost_new_turb, cost_new_pump = 2, 10
    
#     builder(m,'Test1')
    
#     def obj_fun(m):
#      	return sum(-m.MainGrid.P[t]*m.cost[t] for t in l_t) + m.Turb1.Pdim*cost_new_turb + m.PumpNew.Pdim*cost_new_pump
#     m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)
    
#     # with open('model','w') as f:
#     #     m.pprint(f)
    
#     instance = m.create_instance()
#     solver = pyo.SolverFactory('ipopt')
#     solver.solve(instance, tee=False)
    
#     # instance.Turb1.Qin.pprint()
#     # instance.Pump1.Qout.pprint()
#     # instance.PumpNew.Qout.pprint()
#     print('------------------------- Reservoirs -------------------------')
#     instance.Reservoir0.W.pprint()
#     instance.Reservoir1.W.pprint()
#     print('--------------------------- Powers ---------------------------')
#     instance.Turb1.Pe.pprint()
#     instance.Pump1.Pe.pprint()
#     instance.PumpNew.Pe.pprint()
#     instance.PV1.P.pprint()
#     instance.MainGrid.P.pprint()
#     print('--------------------------- Sizing ---------------------------')
#     instance.Turb1.Pdim.pprint()
#     instance.PumpNew.Pdim.pprint()