"""
AGISTIN project 

.\Devices\SolarPV.py

SolarPV pyomo block contains characteristics of a solar PV plant.
"""


import pyomo.environ as pyo
from pyomo.network import *


# data: P(t)
# init_data: None

def SolarPV(b, t, data, init_data=None):
    
    # Parameters
    b._P = pyo.Param(t, initialize=data['P'])
    
    # Variables
    b.P = pyo.Var(t, initialize=data['P'], bounds=(None,0), domain=pyo.Reals)
    
    # Ports
    b.port_P = Port(initialize={'P': (b.P, Port.Extensive)})
    
    # Constraints
    def Constraint_P(_b, _t):
        return b._P[_t] == _b.P[_t]
    b.Constraint_P = pyo.Constraint(t, rule=Constraint_P)
    