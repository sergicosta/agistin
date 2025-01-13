# AGISTIN project 
# .\Devices\Sources.py
"""
Source pyomo block contains characteristics of a source.
"""

import pyomo.environ as pyo
from pyomo.network import Arc, Port


def Source(b, t, data, init_data=None):
    
    """
    Basic Source.
    
    Injects a given flow :math:`Q(t)`, behaving as the hydro analogous of a current source.
    
    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: data ``dict``
    :param init_data: ``None``
        
    data
         - 'Q': Injected flow :math:`Q(t)` as a ``list`` or pandas ``Series``
    
    Pyomo declarations    
        - Parameters: 
            - Qin (t) 
            - Qout (t)
        - Variables: 
        - Ports: 
            - port_Qin @ Qin as ``Extensive``
            - port_Qout @ Qout as ``Extensive``
        - Constraints:
    """
    
    # Parameters
    b.Qin = pyo.Param(t, initialize=[-k for k in data['Q']])
    b.Qout = pyo.Param(t, initialize=data['Q'])
    
    # Variables
    
    # Ports
    b.port_Qin = Port(initialize={'Q': (b.Qin, Port.Extensive)})
    b.port_Qout = Port(initialize={'Q': (b.Qout, Port.Extensive)})
    
    # Constraints


def VarSource(b, t, data, init_data):
        
    """
    Undetermined flow Source.
    
    Injects an undetermined flow :math:`Q(t)`, limited by :math:`Q_{max}`.
    
    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: data ``dict``
    :param init_data: ``None``
        
    data
         - 'Qmin': Minimum flow :math:`Q_{min}` as a ``float`` (can be negative)
         - 'Qmax': Maximum flow :math:`Q_{max}` as a ``float`` (can be negative)
    
    init_data
         - 'Q': Injected flow :math:`Q_{out}(t)` as a ``list`` or pandas ``Series``
         
    Pyomo declarations    
        - Parameters: 
            - Qmin 
            - Qmax
        - Variables: 
            - Qin (t) bounded :math:`Q_{in} \in [-Q_{max},-Q_{min}]`
            - Qout (t) bounded :math:`Q_{out} \in [Q_{min},Q_{max}]`
        - Ports: 
            - port_Qin @ Qin as ``Extensive``
            - port_Qout @ Qout as ``Extensive``
        - Constraints:
            - c_Q: :math:`Q_{in}(t)=-Q_{out}(t)`
    """
    
    #Parameters
    b.Qmax = pyo.Param(initialize=data['Qmax'])
    b.Qmin = pyo.Param(initialize=data['Qmin'])

    #Variables
    b.Qin = pyo.Var(t, initialize=[-k for k in init_data['Q']], within=pyo.Reals, bounds = (-b.Qmax, -b.Qmin))
    b.Qout = pyo.Var(t, initialize=init_data['Q'], within=pyo.Reals, bounds = (b.Qmin, b.Qmax))

    # Ports
    b.port_Qin = Port(initialize={'Q': (b.Qin, Port.Extensive)})
    b.port_Qout = Port(initialize={'Q': (b.Qout, Port.Extensive)})

    # Constraints
    def Constraint_Q(_b, _t):
        return _b.Qin[_t] == -_b.Qout[_t]
    b.c_Q = pyo.Constraint(t, rule=Constraint_Q)

    