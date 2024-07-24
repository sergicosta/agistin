# AGISTIN project 
# .\Devices\Sources.py
"""
Source pyomo block contains characteristics of a source.
"""

import pyomo.environ as pyo
from pyomo.network import Arc, Port


# data: Q(t)
# init_data: None

def Source(b, t, data, init_data=None):
    
    """
    Basic Source.
    
    Injects a given flow :math:`Q(t)`, behaving as the hydro analogous of a current source
    
    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: data ``dict``
    :param init_data: ``None``
        
    data
         - 'Q': Injected flow :math:`Q(t)` as a ``list``
    
    Pyomo declarations    
        - Parameters: 
            - Q (t)
        - Variables: 
            - Qin (t) :math:`\in \mathbb{R}`
            - Qout (t) :math:`\in \mathbb{R}`
        - Ports: 
            - port_Qin @ Qin as ``Extensive``
            - port_Qout @ Qout as ``Extensive``
        - Constraints: 
            - c_Qin: :math:`Q_{in}(t) = -Q(t)`
            - c_Qout: :math:`Q_{out}(t) = Q(t)`
    """
    
    # Parameters
    b.Q = pyo.Param(t, initialize=data['Q'])
    
    # Variables
    b.Qin = pyo.Var(t, initialize=[-k for k in data['Q']], within=pyo.Reals)
    b.Qout = pyo.Var(t, initialize=data['Q'], within=pyo.Reals)
    
    for k in range(len(t)):
        b.Qin[k].bounds = (-data['Q'][k],-data['Q'][k])
        b.Qout[k].bounds = (data['Q'][k],data['Q'][k])
    
    # Ports
    b.port_Qin = Port(initialize={'Q': (b.Qin, Port.Extensive)})
    b.port_Qout = Port(initialize={'Q': (b.Qout, Port.Extensive)})
    
    # Constraints


def VarSource(b, t, data, init_data):
    
    #Parameters
    b.Qmax = pyo.Param(initialize=data['Qmax'])

    #Variables
    b.Qin = pyo.Var(t, initialize=init_data['Qin'], within=pyo.NonPositiveReals)
    b.Qout = pyo.Var(t, initialize=init_data['Qout'], within=pyo.Reals, bounds = [0,b.Qmax])
    # b.Q = pyo.Var(t, initialize=init_data['Q'], within=pyo.Reals)

    # Ports
    b.port_Qin = Port(initialize={'Q': (b.Qin, Port.Extensive)})
    b.port_Qout = Port(initialize={'Q': (b.Qout, Port.Extensive)})

    
    # Constraint
    def Constraint_Qin(_b, _t):
        return _b.Qin[_t] == -_b.Qout[_t]
    b.c_Qin = pyo.Constraint(t, rule=Constraint_Qin)
    
    # def Constraint_Qout(_b, _t):
    #     return _b.Qout[_t] == _b.Q[_t]
    # b.c_Qout = pyo.Constraint(t, rule=Constraint_Qout)
    
    # def Constraint_Q (_b,_t):
    #     return _b.Q[_t] <= _b.Qmax
    # b.c_Q = pyo.Constraint(t, rule = Constraint_Q)

    