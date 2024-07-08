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
#from Devices.SolarPV import SolarPV
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
l_t = list(range(55)) #TODO this should be inferred from the number of rows in the excel time series,
#TODO it would be nice to have a consistency check ensuring that data has been correctly filled in all sheets.
m.t = pyo.Set(initialize=l_t)
builder(m, data_filename)

# """
# # Connections

# m.grideb = Arc(ports=(m.Grid.port_P, m.EB.port_P), directed=True)
# m.pveb = Arc(ports=(m.PV.port_P, m.EB.port_P), directed=True)
# m.batteryebeb = Arc(ports=(m.Battery.port_P, m.EB.port_P), directed=True)

# pyo.TransformationFactory("network.expand_arcs").apply_to(m) # apply arcs to model
# """

#%% RUN THE OPTIMIZATION

m.max_SOC = pyo.Var(within=pyo.NonNegativeReals)
m.W = pyo.Var(within=pyo.NonNegativeReals)


# Objective function
def obj_fun(m):
    return (
        sum(((m.Battery.PowerFCRCharge[t] * 5) + (m.Battery.PowerFCRDisCharge[t]) * -10 + m.Battery.Pfeedin[t] * -6 + m.Battery.Pfeedout[t] * -8) for t in l_t)
         - (m.Battery.Pmax * m.W)
         - (m.Battery.Emax * m.max_SOC)
    )

    # return sum(((m.Battery.PowerFCRCharge[t]*5) + (m.Battery.PowerFCRDisCharge[t])*-10 + m.Battery.Pfeedin[t]*-6 + m.Battery.Pfeedout[t]*-8)  for t in l_t ) #+ m.PV.Pdim*m.cost_PV1[0]   
m.goal = pyo.Objective(rule=obj_fun, sense=pyo.maximize)
    #"""
instance = m.create_instance()
solver = pyo.SolverFactory('ipopt')
solver.solve(instance, tee=False)

#   return sum(((m.Battery.PowerFCRCharge[t]*10 + m.Battery.P_SCcharged[t]*5 ) + (m.Battery.PowerFCRDisCharge[t] + m.Battery.P_SCdischarged[t])) for t in l_t ) #+ m.PV.Pdim*m.cost_PV1[0]  
#    return ((m.Batteries.BCharge[t] + m.Batteries.BDischarge[t]) for t in l_t ) #+ m.PV.Pdim*m.cost_PV1[0]   
#instance.Grid.P.pprint()

#instance.Battery.EENS.pprint()
#instance.Battery.Pcharged.pprint()
#instance.Battery.Pdischarged.pprint()
#instance.Battery.EstrgOut.pprint()
#instance.Battery.EstrgIni.pprint()
#instance.Battery.FeedinMax.pprint()

instance.Battery.SOC.pprint()
instance.Battery.PowerFCRDisCharge.pprint()
instance.Battery.PowerFCRCharge.pprint()
instance.Battery.Pfeedin.pprint()
instance.Battery.Pfeedout.pprint()
#Y(t)=(Pfeedin(t)+ PowerFCRDisCharge(+)) 
#X(t)=(PowerFCRCharge(t) + Pfeedout(t))
instance.Battery.Y.pprint()
instance.Battery.X.pprint()

Y_values = [pyo.value(instance.Battery.Y[t]) for t in instance.t]
X_values = [pyo.value(instance.Battery.X[t]) for t in instance.t]
SOC_values = [pyo.value(instance.Battery.SOC[t]) for t in instance.t]

max_Y = max(Y_values)
max_X = max(X_values)
max_SOC = max(SOC_values)

# Compare max(X) and max(Y) and 
# find out one max value and store in W.

W = max(max_Y, max_X)

print("Maximum Y_Pfeedin:", max_Y)
print("Maximum X_Pfeedout:", max_X)
print("Maximum SOC:", max_SOC)
print("Maximum W:", W)


#print(instance.Battery.MaxPfeedout) # Working

#instance.Battery.Pout.pprint()
#instance.Battery.Edim.pprint()

#instance.PV.P.pprint()
#instance.PV.Pdim.pprint()
#instance.PV.Pinst.pprint()

