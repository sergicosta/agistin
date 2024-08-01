# AGISTIN project 
# .\Devices\EB.py
"""
EB pyomo block containing the characteristics of a pumping station.

It is meant to act as a node for power balance.
"""


import pyomo.environ as pyo
from pyomo.network import Arc, Port


# data: None
# init_data: None

def EB(b, t, data=None, init_data=None):

    """
    Basic energy balance node.
    
    Captures the powers linked to it and sets their sum to 0:
    
    .. math::
        \sum_{i} P_i = 0 
    
    Block parameters
    ----------------
    - t: pyomo Set() referring to time
    - data: *None*
    - init_data: *None*
    
    Pyomo declarations
    ------------------
    
    - Parameters: 
        - P_bal (t) :math:`\in \mathbb{R}`
    - Variables: 
        - *None*
    - Ports: 
        - port_P @ P_bal (Extensive)
    - Constraints: 
        - *None*
    """

    
    # Parameters
    b.P_bal = pyo.Param(t, initialize=0)
    
    # Variables
    
    # Ports
    b.port_P = Port(initialize={'P': (b.P_bal, Port.Extensive)})
    
    # Constraints