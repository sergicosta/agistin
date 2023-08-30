# AGISTIN project 
# .\Devices\Switch.py
"""
Switch pyomo block contains characteristics of a switch for line commutation.
"""

import pyomo.environ as pyo
from pyomo.network import *


# data: None
# init_data: None

def Switch(b, t, data=None, init_data=None):
   
    """
    Electrical switch.
    
    Commutes the output :math:`P_{out}(t)` between 2 inputs of water flow :math:`P_{in,0}(t)` and :math:`P_{in,1}(t)`
    
    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: ``None``
    :param init_data: ``None``
    
    Pyomo declarations    
        - Parameters:
            - `None`
        - Variables: 
            - Pin0 (t)
            - Pin1 (t)
            - Pout (t)
        - Ports: 
            - port_Pin0 @ Pin0 as ``Extensive``
            - port_Pin1 @ Pin1 as ``Extensive``
            - port_Pout @ Pout as ``Extensive``
        - Constraints: 
            - c_State: :math:`P_{in,0}(t) \cdot P_{in,1}(t) = 0`
            - c_Pout: :math:`P_{out}(t) + P_{in,0}(t) + P_{in,1}(t) = 0`
    """
    
    # Parameters
    
    # Variables
    b.Pin0 = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, domain=pyo.Reals)
    b.Pin1 = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, domain=pyo.Reals)
    b.Pout = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, domain=pyo.Reals)
    
    # Ports
    b.port_Pin0 = Port(initialize={'P': (b.Pin0, Port.Extensive)})
    b.port_Pin1 = Port(initialize={'P': (b.Pin1, Port.Extensive)})
    b.port_Pout = Port(initialize={'P': (b.Pout, Port.Extensive)})
    
    # Constraints
    def Constraint_State(_b, _t):
        return _b.Pin0[_t] * _b.Pin1[_t] == 0
    b.c_State = pyo.Constraint(t, rule=Constraint_State)
    def Constraint_Pout(_b, _t):
        return 0 == _b.Pin0[_t] + _b.Pin1[_t] + _b.Pout[_t]
    b.c_Pout = pyo.Constraint(t, rule=Constraint_Pout)
    