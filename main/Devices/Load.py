# -*- coding: utf-8 -*-
"""
Electrical Load pyomo block.
"""
import pyomo.environ as pyo
from pyomo.network import Arc, Port


# data: P(t)
# init_data: None

def Load(b, t, data, init_data=None):
    
    """
    Basic Load.
    
    Injects a given power :math:`P(t)`, behaving as the electrical load consume.
    
    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: data ``dict``
    :param init_data: ``None``
        
    data
         - 'P': Injected flow :math:`P(t)` as a ``list``
    
    Pyomo declarations    
        - Parameters: 
            - P (t)
        - Variables: 
            - Pin (t) :math:`\in \mathbb{R}`
            - Pout (t) :math:`\in \mathbb{R}`
        - Ports: 
            - port_Pin @ Pin as ``Extensive``
            - port_Pout @ Pout as ``Extensive``
        - Constraints: 
            - c_Pin: :math:`P_{in}(t) = -P(t)`
            - c_Pout: :math:`P_{out}(t) = P(t)`
    """
    
    # Parameters
    b.P = pyo.Param(t, initialize=data['P'])
    
    # Variables
    b.Pin = pyo.Var(t, initialize=data['P'], within=pyo.Reals)
    b.Pout = pyo.Var(t, initialize=data['P'], within=pyo.Reals)
    
    # Ports
    b.port_Pin = Port(initialize={'P': (b.Pin, Port.Extensive)})
    b.port_Pout = Port(initialize={'P': (b.Pout, Port.Extensive)})
    
    # Constraints
    def Constraint_Pin(_b, _t):
        return _b.Pin[_t] == -_b.P[_t]
    b.c_Pin = pyo.Constraint(t, rule=Constraint_Pin)
    def Constraint_Pout(_b, _t):
        return _b.Pout[_t] == _b.P[_t]
    b.c_Pout = pyo.Constraint(t, rule=Constraint_Pout)
    