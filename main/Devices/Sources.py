"""
AGISTIN project 

.\Devices\Sources.py

Source pyomo block contains characteristics of a source.
"""


import pyomo.environ as pyo
from pyomo.network import *


# data: Q(t)

def Source(b, t, data):
    
    # Parameters
    b.Q = pyo.Param(t, initialize=data['Q'])
    
    # Variables
    
    
    # Ports
    b.outlet = Port(initialize={'Q': (b.Q, Port.Extensive)})
    
    # Constraints
    