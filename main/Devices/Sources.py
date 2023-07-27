"""
AGISTIN project 

.\Devices\Sources.py

Source pyomo block contains characteristics of a source.
"""


import pyomo.environ as pyo
from pyomo.network import *


# data: Q(t)

def Source(b, t, data, init_data=None):
    
    # Parameters
    b.Q = pyo.Param(t, initialize=data['Q'])
    
    # Variables
    b.Qin = pyo.Var(t, initialize=data['Q'], within=pyo.Reals)
    b.Qout = pyo.Var(t, initialize=data['Q'], within=pyo.Reals)
    
    # Ports
    b.port_Qin = Port(initialize={'Q': (b.Qin, Port.Extensive)})
    b.port_Qout = Port(initialize={'Q': (b.Qout, Port.Extensive)})
    
    # Constraints
    def Constraint_Qin(_b, _t):
        return _b.Qin[_t] == -_b.Q[_t]
    b.c_Qin = pyo.Constraint(t, rule=Constraint_Qin)
    def Constraint_Qout(_b, _t):
        return _b.Qout[_t] == _b.Q[_t]
    b.c_Qout = pyo.Constraint(t, rule=Constraint_Qout)
    