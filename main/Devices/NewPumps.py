# AGISTIN project 
# .\Devices\NewPumps.py

"""
NewPump pyomo block contains characteristics of a pump to size.
"""


import pyomo.environ as pyo
from pyomo.network import Arc, Port


# data: eff
# init_data: Q(t), H(t), Pe(t)

def NewPump(b, t, data, init_data):
    
    """
    Sizing of a new pump's nominal power :math:`P_{dim}`
    
    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: data ``dict``
    :param init_data: init_data ``dict``
        
    data
         - 'eff': Efficiency at the nominal operating point in p.u. :math:`\eta` as a ``float``
         
    init_data
         - 'Q': Flow :math:`Q(t)` as a ``list`` or pandas ``Series``
         - 'H': Head :math:`H(t)` as a ``list`` or pandas ``Series``
         - 'Pe': Electrical power :math:`P_e(t)` as a ``list`` or pandas ``Series``

    Pyomo declarations    
        - Parameters:
            - eff
        - Variables:
            - Pdim bounded :math:`P_{dim} \ge 0`
            - Qin (t) bounded :math:`Q_{in} \le 0`
            - Qout (t) bounded :math:`Q_{out} \ge 0`
            - H (t) bounded :math:`H \ge 0`
            - Ph (t) bounded :math:`P_h \ge 0`
            - Pe (t) bounded :math:`P_e \ge 0`
        - Ports:
            - port_Qin @ Qin with 'Q' as ``Extensive``
            - port_Qout @ Qout with 'Q' as ``Extensive``
            - port_P @ Pe with 'P' as ``Extensive``
            - port_H @ H with 'H' as ``Equality``
        - Constraints:
            - c_Q: :math:`Q_{in}(t) = - Q_{out}(t)`
            - c_Ph: :math:`P_h(t) = 9810\, H(t)\, Q_{out}(t)`
            - c_Pe: :math:`P_e(t) = P_h(t)/\eta`
            - c_Pdim: :math:`P_{dim}(t) \ge P_e(t)`
    """
    
    # Parameters
    b.eff = pyo.Param(initialize=data['eff'])
    
    # Variables
    b.Pdim = pyo.Var(initialize=0, within=pyo.NonNegativeReals)
    b.Qin  = pyo.Var(t, initialize=init_data['Q'], within=pyo.NonPositiveReals)
    b.Qout = pyo.Var(t, initialize=init_data['Q'], within=pyo.NonNegativeReals)
    b.H    = pyo.Var(t, initialize=init_data['H'], within=pyo.NonNegativeReals) 
    b.Ph   = pyo.Var(t, initialize=init_data['Pe'], within=pyo.NonNegativeReals)
    b.Pe   = pyo.Var(t, initialize=init_data['Pe'], within=pyo.NonNegativeReals)
    
    # Ports
    b.port_Qin = Port(initialize={'Q': (b.Qin, Port.Extensive)})
    b.port_Qout = Port(initialize={'Q': (b.Qout, Port.Extensive)})
    b.port_P = Port(initialize={'P': (b.Pe, Port.Extensive)})
    b.port_H = Port(initialize={'H': (b.H, Port.Equality)})
    
    # Constraints
    def Constraint_Q(_b, _t):
        return _b.Qin[_t] == -_b.Qout[_t]
    b.c_Q = pyo.Constraint(t, rule = Constraint_Q)
    
    def Constraint_Ph(_b, _t):
        return _b.Ph[_t] == 9810*_b.H[_t]*_b.Qout[_t]
    b.c_Ph = pyo.Constraint(t, rule = Constraint_Ph)
    
    def Constraint_Pe(_b, _t):
        return _b.Pe[_t] == _b.Ph[_t]/_b.eff
    b.c_Pe = pyo.Constraint(t, rule = Constraint_Pe)

    def Constraint_Pdim(_b, _t):
        return _b.Pdim >= _b.Pe[_t]
    b.c_Pdim = pyo.Constraint(t, rule = Constraint_Pdim)