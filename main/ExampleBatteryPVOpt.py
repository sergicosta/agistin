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
from pyomo.network import Arc, Port

# Import builder
from BuilderBatteryPVOpt import data_parser, builder

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
l_t = list(range(55)) #TODO this should be inferred from the number of rows in the excel time series,
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
	return sum(( (m.cost_MainGrid[t])*m.Grid.Pbuy[t]) - (m.Battery.Pout[t]*m.rev_FCR[t]) - (m.cost_feed_in[t]*m.Grid.Psell[t]) for t in l_t ) - ( m.Battery.Edim*m.cost_Battery1[0] )#+ m.Battery.Edim*m.cost_Battery1[0])
m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)

# m.Battery.Edim: Additional battery capacity to be installed as result of the optimization

#"""
instance = m.create_instance()
solver = pyo.SolverFactory('ipopt') #  glpk'
solver.solve(instance, tee=False)

instance.Grid.P.pprint()
#instance.Grid.Psell.pprint()
#instance.Grid.Pbuy.pprint()

#instance.Battery.Pdemanded.pprint()
instance.Battery.EstrgOut.pprint()
instance.Battery.Pout.pprint()


instance.Battery.SOC.pprint()
instance.Battery.Edim.pprint()
#instance.Battery.POutdischarged.pprint()
#instance.Battery.POutcharged.pprint()

#instance.Battery.Edim.pprint()

#instance.PV.P.pprint()
#instance.PV.Pdim.pprint()
#instance.Battery.Edim.pprint()
#instance.PV.Pinst.pprint()





