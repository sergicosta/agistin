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
from Devices.Batteries import Battery_MV
from Devices.SolarPV import SolarPV
#from Devices.Pumps import Pump


# Import useful functions

#clean console and variable pane
# clear_clc() #consider removing if you are not working with Spyder
from Utilities import clear_clc

data_filename = "ExampleBatteryPV"

# generate system json file
data_parser(data_filename, dt=1) # dt = value of each timestep (if using SI this is seconds)

m = pyo.ConcreteModel()

# time
l_t = list(range(25)) #TODO this should be inferred from the number of rows in the excel time series,
#TODO it would be nice to have a consistency check ensuring that data has been correctly filled in all sheets.
m.t = pyo.Set(initialize=l_t)

builder(m, data_filename)

# Define variables
m.EstrgOut = pyo.Var(m.t, domain=pyo.NonNegativeReals)  # Ensure non-negativity for energy storage output
m.SOC = pyo.Var(m.t, domain=pyo.NonNegativeReals)  # Ensure non-negativity for state of charge
# Define indexed parameters for SOC limits
SOCmin_limit = 0.25  # 25%
SOCmax_limit = 0.75  # 75%

# Initialize SOC limits
m.Battery.SOCmin = pyo.Param(m.t, initialize=SOCmin_limit)
m.Battery.SOCmax = pyo.Param(m.t, initialize=SOCmax_limit)

# Add constraints
# Add constraints
# Add constraints
# def soc_limits_rule(m, t):
#     return (0, m.SOC[t], 1 + 1e-6)  # Ensure SOC stays within limits of 0 to 1 with a small tolerance

# m.soc_limits = pyo.Constraint(m.t, rule=soc_limits_rule)


def estrg_out_non_negative_rule(m, t):
    return m.EstrgOut[t] >= 0 + 1e-6  # Ensure EstrgOut is non-negative with a small tolerance
m.estrg_out_non_negative = pyo.Constraint(m.t, rule=estrg_out_non_negative_rule)

# Modify EstrgOut calculation based on SOC, charged power, and discharged power
def estrg_out_rule(m, t):
    return m.EstrgOut[t] == m.Battery.Pcharged[t] - m.Battery.Pdischarged[t]

m.estrg_out_calculation = pyo.Constraint(m.t, rule=estrg_out_rule)
# Define efficiency parameter for the battery
m.Battery.Efficiency = pyo.Param(initialize=0.95)  # Example value, replace with the actual efficiency

# After reaching full charge or discharge, update SOC for the next time step
def soc_update_rule(m, t):
    if t == m.t.last():
        # Handle the last time step separately to avoid out-of-bounds index
        return pyo.Constraint.Skip
    else:
        return m.SOC[t+1] == m.SOC[t] + (m.Battery.Pcharged[t] - m.Battery.Pdischarged[t]) / m.Battery.Efficiency

m.soc_update = pyo.Constraint(m.t, rule=soc_update_rule)



# Adjusted SOC limits constraint to ensure SOC stays strictly within the range of 0 to 1
# Adjusted SOC limits constraint to ensure SOC stays strictly within the range of 0 to 1
# Adjusted SOC limits constraint to ensure SOC stays strictly within the range of 0 to 1
# Adjusted SOC limits constraint with a stricter lower bound
# Adjusted SOC limits constraint with a stricter lower bound
def soc_limits_rule(m, t):
    return (0, m.SOC[t], 1 - 1e-6)  # Ensure SOC stays within limits of 0 to 1 with a small tolerance

m.soc_limits = pyo.Constraint(m.t, rule=soc_limits_rule)




m.soc_limits = pyo.Constraint(m.t, rule=soc_limits_rule)




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
def obj_fun(m):
	return sum((m.Grid.Pbuy[t]*m.cost_MainGrid[t] - m.Grid.Psell[t]*m.cost_MainGrid[t]/2) for t in l_t ) + m.PV.Pdim*m.cost_PV1[0] 
    #return sum((m.Grid.Pbuy) for t in l_t )
#m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)
#"""
instance = m.create_instance()
solver = pyo.SolverFactory('ipopt')
solver.solve(instance, tee=False)

#instance.Grid.P.pprint()

# instance.Battery.Pdemanded.pprint()
# instance.Battery.Pcharged.pprint()
# instance.Battery.Pdischarged.pprint()
instance.Battery.EstrgOut.pprint()
#instance.Battery.EstrgIni.pprint()

instance.Battery.SOC.pprint()

instance.Battery.SOCmin.pprint()
instance.Battery.SOCmax.pprint()

instance.Battery.Pout.pprint()
# instance.Battery.Edim.pprint()

instance.PV.P.pprint()
instance.PV.Pdim.pprint()
instance.PV.Pinst.pprint()




