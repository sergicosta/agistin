import pyomo.environ as pyo
from pyomo.network import Arc, Port
import json
from BuilderBatteryPV import data_parser, builder
from Devices.MainGrid import Grid
from Devices.EB import EB
from Devices.SolarPV import SolarPV
from Devices.Nishat_Battery import Battery_MV
from Utilities import clear_clc

data_filename = "ExampleBatteryPV"

# Generate system json file
data_parser(data_filename, dt=1)

m = pyo.ConcreteModel()

# Time
l_t = list(range(55))
m.t = pyo.Set(initialize=l_t)

builder(m, data_filename)

# Add Nishat Battery MV
F_data = [
    50.0, 50.2, 50.2, 50.05, 50.05, 50.1, 50.2, 50.0, 50.0, 49.8, 
    49.8, 49.8, 49.8, 49.8, 49.8, 49.8, 49.8, 49.8, 49.8, 49.8, 
    49.8, 49.95, 50.0, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 
    50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 
    50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 
    50.0, 49.8, 49.8, 49.8, 49.8
]

data = {
    'E0': 2000,  
    'Emax': 24000,  
    'Emin': 0,  
    'SOCmax': 1,  
    'SOCmin': 0,  
    'Pmax': 1000,  
    'Einst': 24000,  
    'Pinst': 1000,  
    'rend_ch': 0.9,  
    'rend_disc': 1.1,  
    'slope_fcr': -5,  
    'dt': 1,  
    'F': F_data,  
}

Battery_MV(m, l_t, data)

# Define variables, constraints, and objective function as needed

# Solve the optimization problem
instance = m.create_instance()
solver = pyo.SolverFactory('ipopt')
solver.solve(instance, tee=False)

# Print results or perform further analysis as needed
instance.Battery.EstrgOut.pprint()
instance.Battery.SOC.pprint()
