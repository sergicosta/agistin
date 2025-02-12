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
# Import pyomo2
import pyomo.environ as pyo
import pandas as pd
from pyomo.network import Arc, Port

# Import builder
from BuilderBatteryPumpPVOpt import data_parser, builder

# Import devices
from Devices.Reservoirs import Reservoir_Ex0
from Devices.Sources import Source
from Devices.Pipes import Pipe_Ex0
from Devices.Pumps import Pump
from Devices.EB import EB
from Devices.SolarPV import SolarPV
from Devices.MainGrid import Grid
from Devices.Batteries import Battery_FCR


# Import useful functions

#clean console and variable pane
# clear_clc() #consider removing if you are not working with Spyder
from Utilities import clear_clc

data_filename = "ExampleBatteryPumpPV10H"

# generate system json file
data_parser(data_filename, dt=1) # dt = value of each timestep (if using SI this is seconds)

m = pyo.ConcreteModel()

# time
l_t = list(range(10)) #TODO this should be inferred from the number of rows in the excel time series,
#TODO it would be nice to have a consistency check ensuring that data has been correctly filled in all sheets.
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
#""
#Objective function


cost_new_battery_MW  = 105.3333# Euros/day 100 70
cost_new_battery_MWh = 104.8333# Euros/day   90 40
#cost_P_Add_Charge = 10 # Euros/day
#cost_new_pv = 1.041666 # Euros/day
#ost_new_BESS_ = 200

    #return sum((m.Grid.Pbuy[t]*m.cost_MainGrid[t] - m.Grid.Psell[t]*m.cost_MainGrid[t]/2) for t in l_t ) + m.PV.Pdim*m.cost_PV1[0]
#m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)
def obj_fun(m):
    return ((sum((abs(m.Battery.P_FCRCharge[t]) + abs(m.Battery.P_FCRDisCharge[t]))*0.25*m.Battery.FCR_reBAP[t] + m.Battery.Pdim_FCR*m.Battery.FCR_Remuneration[t]
            - (m.Grid.Pbuy[t]- m.Battery.P_FCRCharge[t])*m.cost_MainGrid[t] + (m.Grid.Psell[t] + m.Battery.P_FCRDisCharge[t])*m.cost_MainGrid[t] for t in l_t))
            - cost_new_battery_MW*m.Battery.Pdim_FCR - cost_new_battery_MWh*m.Battery.Edim_FCR - m.PV.Pdim*m.cost_PV1[0] )
m.goal = pyo.Objective(rule=obj_fun, sense=pyo.maximize)

#"""
instance = m.create_instance()
options = {'mumps_icntl_13': 64000}
solver = pyo.SolverFactory('ipopt') #  glpk'

solver.options['max_iter'] = 10000000 # Increase maximum iterations
solver.options['hessian_approximation'] = 'limited-memory'  # Use Hessian approximation

solver.solve(instance, tee=True)

instance.Grid.P.pprint()
instance.Grid.Psell.pprint()
instance.Grid.Pbuy.pprint()

instance.Battery.Pdim_FCR.pprint()
instance.Battery.Edim_FCR.pprint()
instance.Battery.Pout.pprint()
instance.Battery.P_FCRDisCharge.pprint()
instance.Battery.P_FCRCharge.pprint()
instance.Battery.EnerStorg.pprint()

instance.PV.Pinst.pprint()
instance.PV.Pdim.pprint()
instance.PV.P.pprint()

instance.Reservoir1.W.pprint()
instance.Reservoir0.W.pprint()

instance.Pump1.Pe.pprint()
instance.Pump2.Pe.pprint()

instance.Pump1.Ph.pprint()

PowerFCRDisCharge_values = [pyo.value(instance.Battery.P_FCRDisCharge[t]) for t in instance.t]
PowerFCRCharge_values = [pyo.value(instance.Battery.P_FCRCharge[t]) for t in instance.t]
EnerStorg_values = [pyo.value(instance.Battery.EnerStorg[t]) for t in instance.t]

PowerPV_values = [pyo.value(instance.PV.P[t]) for t in instance.t]
GridP_values = [pyo.value(instance.Grid.P[t]) for t in instance.t]
GridPsell_values = [pyo.value(instance.Grid.Psell[t]) for t in instance.t]
GridPbuy_values = [pyo.value(instance.Grid.Pbuy[t]) for t in instance.t]

Pump1P_Values = [pyo.value(instance.Pump1.Pe[t]) for t in instance.t]
Pump2P_Values = [pyo.value(instance.Pump2.Pe[t]) for t in instance.t]

# Store the values in a dictionary for summary results
results_summary = {
}

# Convert the dictionary to a pandas DataFrame
summary_df = pd.DataFrame(results_summary)

# Store the detailed time-series results in a dictionary
results_detailed = {
    'Time': l_t,
    'PowerGrid': GridP_values,
    'PsellGrid': GridPsell_values,
    'PbuyGrid': GridPbuy_values,

    'EnerStorg': EnerStorg_values,
    'PowerFCRDisCharge': PowerFCRDisCharge_values,
    'PowerFCRCharge': PowerFCRCharge_values,

    'PowerPV': PowerPV_values,

    'PowerPump1': Pump1P_Values,
    'PowerPump2': Pump2P_Values,

}

# Convert the dictionary to a pandas DataFrame
detailed_df = pd.DataFrame(results_detailed)

# Create a Pandas Excel writer using openpyxl as the engine
with pd.ExcelWriter('Battery+Pv+Pump_optimization_results.xlsx', engine='openpyxl') as writer:
    # Write the summary DataFrame to the Excel file
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

    # Write the detailed DataFrame to the Excel file
    detailed_df.to_excel(writer, sheet_name='Detailed', index=False)

print("Results have been written to 'Battery+Pv+Pump_optimization_results.xlsx'")




