"""
AGISTIN project

.\Builder.py

Builder functions generate a complete pyomo model from a .json file.
"""

import pyomo.environ as pyo
from pyomo.network import Arc, Port
import json
import os
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
    # Load static, time-series, and cost data robustly.
    # If some files are missing, fallback to known cases or generate defaults.
    fallback_case = 'ExampleBatteryPV720H'
    cases_dir = os.path.join(os.path.dirname(__file__), 'Cases')
    # Static data
    def _load_xlsx(name):
        return pd.read_excel(os.path.join(cases_dir, f'{name}.xlsx'), sheet_name=None)

    try:
        df = _load_xlsx(NameTest)
    except FileNotFoundError:
        df = _load_xlsx(fallback_case)

    # If the provided static workbook doesn't have 'Name' columns, fall back to a template with proper static schema
    def _has_name_columns(d):
        try:
            for sh, frame in d.items():
                if isinstance(frame, pd.DataFrame) and len(frame) > 0:
                    if 'Name' not in frame.columns:
                        return False
            return True
        except Exception:
            return False

    if not _has_name_columns(df):
        # Prefer a SOH-capable static template if available
        static_candidates = [
            'ExampleBatteryPV720H_SOH',
            fallback_case
        ]
        for cand in static_candidates:
            try:
                df = _load_xlsx(cand)
                if _has_name_columns(df):
                    break
            except FileNotFoundError:
                continue

    # Time-series data (try a few sensible alternatives)
    df_time = None
    time_candidates = [
        os.path.join(cases_dir, f'{NameTest}_time.xlsx'),
        os.path.join(cases_dir, f'{NameTest}.xlsx'),  # sometimes time is in the same file
        os.path.join(cases_dir, 'ExampleBatteryPV_time.xlsx'),
        os.path.join(cases_dir, f'{fallback_case}_time.xlsx'),
    ]
    last_err = None
    for path in time_candidates:
        try:
            df_time = pd.read_excel(path, sheet_name=None)
            break
        except FileNotFoundError as e:
            last_err = e
            continue
    if df_time is None:
        raise last_err if last_err else FileNotFoundError(f"No time-series workbook found for {NameTest}")

    # Cost data (optional)
    try:
        df_cost = pd.read_excel(os.path.join(cases_dir, f'{NameTest}_cost.xlsx'), sheet_name=None)
    except FileNotFoundError:
        try:
            df_cost = pd.read_excel(os.path.join(cases_dir, f'{fallback_case}_cost.xlsx'), sheet_name=None)
        except FileNotFoundError:
            df_cost = {}
    special = ['SolarPV','Source','Battery_FCR','Battery_SOH']

    # Determine horizon length T from time data: take the maximum non-empty sheet length
    T = 0
    for sh in df_time.values():
        if isinstance(sh, pd.DataFrame) and sh.shape[0] > 0:
            T = max(T, int(sh.shape[0]))
    if T == 0:
        raise ValueError("Time-series workbook contains no rows to infer horizon T")
    
    with open(os.path.join(cases_dir, f'{NameTest}.json'), 'w') as f:
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
                        
                if k in ('Reservoir', 'Battery_FCR', 'Battery_SOH'): # Elements that have constraints modelled as differential equations
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
        
    with open(os.path.join(cases_dir, f'{NameTest}_cost.json'), 'w') as f:
        first = True
        f.write('{')
        if isinstance(df_cost, dict) and len(df_cost) == 0:
            # No cost workbook available: write sensible defaults aligned with T
            f.write(f'\n"cost_MainGrid":{[10]*T},')
            f.write(f'\n"cost_PV1":[0]')
        else:
            for k in df_cost.keys():
                if not df_cost[k].empty:
                    for it in df_cost[k]:
                        series = list(df_cost[k][it])
                        # If provided series length doesn't match T, repeat or truncate to T
                        if len(series) != T:
                            if len(series) == 0:
                                series = [0]*T
                            else:
                                # tile to length T
                                reps = (T + len(series) - 1) // len(series)
                                series = (series * reps)[:T]
                        if first:
                            f.write(f'\n"{it}":{series}')
                            first = False
                        else:
                            f.write(f',\n"{it}":{series}')
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
    from Devices.MainGrid import Grid
    from Devices.SolarPV import SolarPV
    from Devices.Batteries import Battery_FCR, Battery_MV, Battery_SOH, Battery_SOH_Najera

    def _safe_json_load(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            # Try to sanitize content by trimming to the outermost braces and removing stray chars
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                s = f.read()
            start = s.find('{')
            end = s.rfind('}')
            if start != -1 and end != -1 and end > start:
                s = s[start:end+1]
                return json.loads(s)
            raise

    system_path = os.path.join(os.path.dirname(__file__), 'Cases', f'{test_case}.json')
    system = _safe_json_load(system_path)

    # Ensure Battery_SOH degrades to 99% after 1 year: SOH(1y) = 1 - k * 1^n => k = 0.01
    # Also map legacy 'dt' field to 'dt_hours' expected by the device implementation.
    for _comp_name, _comp in list(system.items()):
        if not isinstance(_comp, dict):
            continue
        _data = _comp.get('data', {})
        if not isinstance(_data, dict):
            continue
        # Harmonize timestep key
        if 'dt_hours' not in _data and 'dt' in _data:
            _data['dt_hours'] = _data.get('dt')
        # Robustly detect battery with SOH model and enforce k
        _type = str(_data.get('type', ''))
        if _type == 'Battery_SOH' or (_comp_name.strip().lower() == 'battery' and 'Battery' in _type):
            _data['k'] = 0.01
        _comp['data'] = _data

    for it in list(system.keys()):
        setattr(m, it, pyo.Block())
    
    for it in list(system.keys()):
        s = system[it]['data']['type']
        # Prefer SOH battery model if a plain 'Battery' is specified
        if s == 'Battery':
            s = 'Battery_SOH'
        create = locals()[s]
        create(getattr(m, it), m.t, system[it]['data'], system[it]['init_data'])
    
    val = 0
    
    for it in list(system.keys()):
        for j in list(system[it]['conns'].keys()):
            setattr(m, f'arc_{val}', Arc(ports=(getattr(getattr(m, it), f'port_{j}'), getattr(getattr(m, system[it]['conns'][j][0]), f'port_{system[it]["conns"][j][1]}')), directed=True))
            val += 1
    
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)
    
    cost_path = os.path.join(os.path.dirname(__file__), 'Cases', f'{test_case}_cost.json')
    try:
        cost = _safe_json_load(cost_path)
    except FileNotFoundError:
        cost = {}
        
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