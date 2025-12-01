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

# Import builder
from BuilderBatteryPV import data_parser, builder

# Import devices
from Devices.MainGrid import Grid
from Devices.EB import EB
# from Devices.Batteries import Battery_MV
from Devices.SolarPV import SolarPV
from Devices.Batteries import Battery_SOH
#from Devices.Pumps import Pump


# Import useful functions

#clean console and variable pane
# clear_clc() #consider removing if you are not working with Spyder
from Utilities import clear_clc

"""
Select the 8760-hour example dataset (base name, without _time/_cost).
"""
data_filename = "ExampleBatteryPV8760H"

"""
Generate JSON from Excel and infer horizon T. dt=1 means 1 hour/step for this case.
"""
T = data_parser(data_filename, dt=1)

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

# Degradation summary and life estimation
if hasattr(instance, 'Battery') and hasattr(instance.Battery, 'SOH'):
    # Print SOH over time (compact) and key parameters
    instance.Battery.SOH.pprint()

    # Extract parameters and compute metrics
    try:
        k = float(pyo.value(instance.Battery.k))
        n = float(pyo.value(instance.Battery.n))
        soh_min = float(pyo.value(instance.Battery.SOH_min))
        dt_h = float(pyo.value(instance.Battery.dt_hours))
    except Exception:
        k, n, soh_min, dt_h = 0.03, 0.5, 0.8, 1.0

    T_steps = len(list(instance.t)) if hasattr(instance, 't') else len(l_t)
    horizon_years = T_steps * dt_h / 8760.0

    # Theoretical EOL (years) when SOH(t) = SOH_min = 1 - k * t^n
    # t_EOL = ((1 - SOH_min) / k)^(1/n) if k>0
    if k > 0 and (1 - soh_min) > 0:
        eol_years = ((1.0 - soh_min) / k) ** (1.0 / n)
    else:
        eol_years = float('inf')

    # Observed start/end SOH and average annual degradation over solved horizon
    soh_vals = [float(pyo.value(instance.Battery.SOH[t])) for t in instance.t]
    soh_start = soh_vals[0]
    soh_end = soh_vals[-1]
    avg_deg_per_year = (soh_start - soh_end) / max(horizon_years, 1e-9)

    # Instantaneous degradation rate at end (derivative of model): dSOH/dt_years = -k*n*t^(n-1)
    t_end_years = horizon_years
    if t_end_years > 0:
        inst_rate_end = -k * n * (t_end_years ** (n - 1.0))
    else:
        inst_rate_end = 0.0

    print("\n===== Battery SOH degradation summary =====")
    print(f"Model: SOH = 1 - k * t^n  (t in years)")
    print(f"k={k:.5f}  n={n:.3f}  SOH_min={soh_min:.3f}  dt_hours={dt_h}")
    print(f"Horizon: {T_steps} steps (~{horizon_years:.2f} years)")
    print(f"SOH start={soh_start:.6f}  SOH end={soh_end:.6f}")
    print(f"Average annual degradation over horizon ≈ {avg_deg_per_year*100:.3f}%/year")
    print(f"Instantaneous degradation rate at end ≈ {inst_rate_end*100:.3f}%/year")
    if eol_years != float('inf'):
        print(f"Estimated end-of-life (SOH={soh_min:.0%}) at ≈ {eol_years:.2f} years from t=0")
        if horizon_years < eol_years:
            print(f"Remaining life from horizon end ≈ {max(eol_years - horizon_years, 0):.2f} years")
    print("==========================================\n")
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
