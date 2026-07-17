# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:07:16 2024

@author: mvalois
"""

"""
    AGISTIN - ExampleBattery
    
    Optimization usage example of the data parser. It considers only a battery model connected to the main grid, and
    reads the data from an external spreadsheet file.
    
    Authors: Manuel Valois (Uni-Kassel), Nishat Jillani (Uni-Kassel)
"""
# Import pyomo
import pyomo.environ as pyo
from pyomo.network import Arc, Port
import json
import os
import pandas as pd

# Import builder
from BuilderBatteryPV import data_parser, builder

# Import devices
from Devices.MainGrid import Grid
from Devices.EB import EB
# from Devices.Batteries import Battery_MV
from Devices.SolarPV import SolarPV
#from Devices.Pumps import Pump


# Import useful functions

#clean console and variable pane
# clear_clc() #consider removing if you are not working with Spyder
from Utilities import clear_clc

"""
Select the 2-year (17518-hour) example dataset (base name, without _time/_cost).
This is the 1-year ExampleBatteryPV8760H input data repeated twice.
"""
data_filename = "ExampleBatteryPV8760H_2Y"

"""
Generate JSON from Excel and infer horizon T. dt=1 means 1 hour/step for this case.
"""
T = data_parser(data_filename, dt=1)

"""
Switch the battery block to the combined calendar + cycling ageing model
(Battery_SOH_Najera), based on Najera et al. 2023 (LFP/NMC semi-empirical
ageing model). The static Excel data still labels the battery as
'Battery_SOH', so the JSON type is patched here before building the model.
"""
_json_path = os.path.join(os.path.dirname(__file__), 'Cases', f'{data_filename}.json')
with open(_json_path, 'r') as _f:
    _system = json.load(_f)
for _name, _comp in _system.items():
    if isinstance(_comp, dict) and _comp.get('data', {}).get('type') in ('Battery_SOH', 'Battery_SOH_Cycle'):
        _comp['data']['type'] = 'Battery_SOH_Najera'
with open(_json_path, 'w') as _f:
    json.dump(_system, _f)

m = pyo.ConcreteModel()

# time from parsed horizon
l_t = list(range(T))
m.t = pyo.Set(initialize=l_t)

builder(m, data_filename)

"""
# Connections

m.grideb = Arc(ports=(m.Grid.port_P, m.EB.port_P), directed=True)
m.pveb = Arc(ports=(m.PV.port_P, m.EB.port_P), directed=True)
m.batteryebeb = Arc(ports=(m.Battery.port_P, m.EB.port_P), directed=True)

pyo.TransformationFactory("network.expand_arcs").apply_to(m) # apply arcs to model
"""

#%% RUN THE OPTIMIZATION
def obj_fun(m):
    # Basic energy cost objective; if cost arrays missing, default to zero
    cost = getattr(m, 'cost_MainGrid', [0]*len(l_t))
    cost_pv = getattr(m, 'cost_PV1', [0])
    return sum((m.Grid.Pbuy[t]*cost[t] - m.Grid.Psell[t]*cost[t]/2) for t in l_t) + m.PV.Pdim*cost_pv[0]

m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)

instance = m.create_instance()
solver = pyo.SolverFactory('ipopt')
solver.solve(instance, tee=False)


#instance.Reservoir1.W.pprint()
#instance.Reservoir0.W.pprint()
#instance.Grid.Pbuy.pprint()
#instance.Grid.Psell.pprint()
instance.Grid.P.pprint()

total_cost = pyo.value(instance.goal)
print(f"\nTotal objective (grid + PV cost, over the {T} steps): {total_cost:.2f}\n")

# Degradation summary (combined calendar + cycling ageing, Battery_SOH_Najera)
if hasattr(instance, 'Battery') and hasattr(instance.Battery, 'SOH'):
    P_vals = [pyo.value(instance.Battery.P[t]) for t in instance.t]
    SOC_vals = [pyo.value(instance.Battery.SOC[t]) for t in instance.t]
    Qcal_vals = [pyo.value(instance.Battery.Q_cal[t]) for t in instance.t]
    Qcyc_vals = [pyo.value(instance.Battery.Q_cyc[t]) for t in instance.t]
    SOH_vals = [pyo.value(instance.Battery.SOH[t]) for t in instance.t]

    dt_h = float(pyo.value(instance.Battery.dt_hours))
    T_steps = len(list(instance.t))
    horizon_years = T_steps * dt_h / 8760.0

    print("\n===== Battery SOH degradation summary (calendar + cycling) =====")
    print(f"Model: SOH = 1 - (Q_cal + Q_cyc)/100, Q_cal & Q_cyc per Najera et al. 2023")
    print(f"Horizon: {T_steps} steps (~{horizon_years:.2f} years)")
    print(f"SOH start={SOH_vals[0]:.6f}  SOH end={SOH_vals[-1]:.6f}")
    print(f"Q_cal start={Qcal_vals[0]:.4f}%  Q_cal end={Qcal_vals[-1]:.4f}%")
    print(f"Q_cyc start={Qcyc_vals[0]:.4f}%  Q_cyc end={Qcyc_vals[-1]:.4f}%")
    print("==================================================================\n")

    # Save Power, SOC and ageing breakdown to CSV
    out_dir = os.path.join(os.path.dirname(__file__), 'Cycling_and_Calendar_Aging')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'Battery_2Years_CombinedAging.csv')
    df = pd.DataFrame({
        'Time_hour': list(range(1, T_steps + 1)),
        'Power': P_vals,
        'SOC': SOC_vals,
        'Q_cal_pct': Qcal_vals,
        'Q_cyc_pct': Qcyc_vals,
        'SOH': SOH_vals,
    })
    df.to_csv(out_path, index=False)
    print(f"Saved combined ageing results to: {out_path}")
#instance.EB.P_bal.pprint()
#instance.EB.P.pprint()

#instance.Battery.P.pprint()
#instance.Battery.E0.pprint()
#instance.Battery.Estr.pprint()
# instance.Battery.SOC.pprint()
#instance.Battery.c_E.pprint()

#nstance.Battery.Pfcr.pprint()
#instance.PV.P.pprint()

#instance.PV.Pdim.pprint()

#instance.Battery.P.pprint()
#instance.Pump2.Pe.pprint()
