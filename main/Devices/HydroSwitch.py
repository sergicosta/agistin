"""
AGISTIN project 

.\Devices\HydroSwitch.py

HydroSwitch pyomo block contains characteristics of a switch for pipe commutation.
"""


import pyomo.environ as pyo
from pyomo.network import *


# data: None
# init_data: None

def HydroSwitch(b, t, data=None, init_data=None):
    
    # Parameters
    
    # Variables
    b.Qin0 = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, domain=pyo.Reals)
    b.Qin1 = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, domain=pyo.Reals)
    b.Qout = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, domain=pyo.Reals)
    
    # Ports
    b.port_Qin0 = Port(initialize={'Q': (b.Qin0, Port.Extensive)})
    b.port_Qin1 = Port(initialize={'Q': (b.Qin1, Port.Extensive)})
    b.port_Qout = Port(initialize={'Q': (b.Qout, Port.Extensive)})
    
    # Constraints
    def Constraint_State(_b, _t):
        return _b.Qin0[_t] * _b.Qin1[_t] == 0
    b.c_State_h = pyo.Constraint(t, rule=Constraint_State)
    def Constraint_Pout(_b, _t):
        return 0 == _b.Qin0[_t] + _b.Qin1[_t] - _b.Qout[_t]
    b.c_Qout = pyo.Constraint(t, rule=Constraint_Pout)
    