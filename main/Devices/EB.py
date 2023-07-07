"""
AGISTIN project 

.\Devices\EB.py

EB pyomo block contains characteristics of a pumping station.
"""


import pyomo.environ as pyo
from pyomo.network import *


# data: None
# init_data: None

def EB(b, t, data, init_data):
    
    # Parameters
    
    # Variables
    b.P_bal = pyo.Var(t, initialize={k:0.0 for k in range(1,len(t)+1)}, domain=pyo.Reals)
    
    # Ports
    b.port_P = Port(initialize={'P': (b.P_bal, Port.Extensive)})
    
    # Constraints
    def balance(_b, _t):
        return _b.P_bal[_t] == 0
    b.bal = pyo.Constraint(t, rule=balance)