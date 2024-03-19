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
from BuilderBattery import data_parser, builder

# Import devices
#from Devices.MainGrid import Grid
#from Devices.EB import EB
#from Devices.Batteries import Battery_MV
#from Devices.SolarPV import SolarPV
#from Devices.Pumps import Pump


# Import useful functions

#clean console and variable pane
# clear_clc() #consider removing if you are not working with Spyder
from Utilities import clear_clc

data_filename = "ExampleBattery"

# generate system json file
data_parser(data_filename, dt=1) # dt = value of each timestep (if using SI this is seconds)

m = pyo.ConcreteModel()

# time
l_t = list(range(5)) #TODO this should be inferred from the number of rows in the excel time series,
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
def obj_fun(m):
	#return sum((m.Grid.Pbuy[t]*m.cost_MainGrid[t] - m.Grid.Psell[t]*m.cost_MainGrid[t]/2) for t in l_t ) + m.PV.Pdim*m.cost_PV1[0]
    return sum((m.Grid.Pbuy) for t in l_t )
#m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)
#"""
instance = m.create_instance()
solver = pyo.SolverFactory('ipopt')
solver.solve(instance, tee=False)


#instance.Reservoir1.W.pprint()
#instance.Reservoir0.W.pprint()
#instance.Grid.Pbuy.pprint()
#instance.Grid.Psell.pprint()
instance.Grid.P.pprint()

#instance.EB.P_bal.pprint()
#instance.EB.P.pprint()

instance.Battery.P.pprint()
#instance.Battery._b.P[_t].pprint()

#nstance.Battery.Pfcr.pprint()
#instance.PV.P.pprint()

#instance.PV.Pdim.pprint()

#instance.Battery.P.pprint()
#instance.Pump2.Pe.pprint()

