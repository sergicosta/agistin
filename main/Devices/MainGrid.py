"""
AGISTIN project 

.\Devices\MainGrid.py

MainGrid pyomo block contains characteristics of the point of connection.
"""


import pyomo.environ as pyo
from pyomo.network import *


# data: Pmax
# init_data: None

def Grid(b, t, data, init_data=None):
    
    # Parameters
    b.Pmax = pyo.Param(initialize=data['P_max'])
    
    # Variables
    b.P = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, bounds=(-b.Pmax, b.Pmax), domain=pyo.Reals)
    
    # Ports
    b.port_P = Port(initialize={'P': (b.P, Port.Extensive)})
    
    # Constraints