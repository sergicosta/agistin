"""
AGISTIN project 

.\Devices\Reservoirs.py

EB pyomo block contains characteristics of a pumping station.
"""


import pyomo.environ as pyo
from pyomo.network import *


# data: None
# init_data: None

def EB(b, t, data, init_data):
    
    # Parameters
    
    # Variables
    b.p_bal = pyo.Var(t, initialize={k:0.0 for k in range(1,len(t)+1)}, domain=pyo.Reals)
    
    # Ports
    b.outlet = Port(initialize={'p': (b.p_bal, Port.Extensive)})
    
    # Constraints
    def balance(_b, _t):
        return _b.p_bal[_t] == 0
    b.bal = pyo.Constraint(t, rule=balance)