# AGISTIN project 
# .\Devices\EB.py
"""
EB pyomo block containing the characteristics of a pumping station.

It is meant to act as a node for power balance.
"""


import pyomo.environ as pyo
from pyomo.network import *


# data: None
# init_data: None

def EB(b, t, data=None, init_data=None):

    """
    Basic EB.
    
    Captures the powers linked to it and sets their sum to 0:
    
    .. math::
        \sum_{i} P_i = 0 
    
    Block parameters
    ----------------
    - t: time set
    - data: *None*
    - init_data: *None*
    
    Pyomo declarations
    ------------------
    
    - **Parameters**: 
        - *None*
    - **Variables**: 
        - P_bal
    - **Ports**: 
        - port_P
    - **Constraints**: 
        - balance
    """

    
    # Parameters
    
    # Variables
    b.P_bal = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, domain=pyo.Reals)
    
    # Ports
    b.port_P = Port(initialize={'P': (b.P_bal, Port.Extensive)})
    
    # Constraints
    def balance(_b, _t):
        return _b.P_bal[_t] == 0
    b.bal = pyo.Constraint(t, rule=balance)