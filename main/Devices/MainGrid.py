"""
AGISTIN project 

.\Devices\Reservoirs.py

MainGrid pyomo block contains characteristics of the point of connection.
"""


import pyomo.environ as pyo
from pyomo.network import *


# data: Pmax
# init_data: None

def grid(b, t, data, init_data):
    
    # Parameters
    b.Pmax = pyo.Param(initialize=data['P_max'])
    
    # Variables
    b.P = pyo.Var(t, initialize={k:0.0 for k in range(1,len(t)+1)}, bounds=(-b.Pmax, b.Pmax), domain=pyo.Reals)
    
    # Ports
    b.port_P = Port(initialize={'P': (b.P, Port.Extensive)})
    
    # Constraints