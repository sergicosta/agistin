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
    b.Pmax = pyo.Param(initialize=data['Pmax'])
    
    # Variables
    b.P = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, bounds=(-b.Pmax, b.Pmax), domain=pyo.Reals)
    b.Psell = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, bounds=(0, b.Pmax), domain=pyo.NonNegativeReals)
    b.Pbuy = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, bounds=(0, b.Pmax), domain=pyo.NonNegativeReals)
    
    # Ports
    b.port_P = Port(initialize={'P': (b.P, Port.Extensive)})
    
    # Constraints
    def Constraint_P(_b, _t):
        return _b.P[_t] == _b.Psell[_t] - _b.Pbuy[_t]
    b.c_P = pyo.Constraint(t, rule=Constraint_P)
    
    def Constraint_P0(_b, _t):
        return 0 == _b.Psell[_t]*_b.Pbuy[_t]
    b.c_P0 = pyo.Constraint(t, rule=Constraint_P0)