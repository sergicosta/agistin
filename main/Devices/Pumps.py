# AGISTIN project 
# .\Devices\Pumps.py

"""
Pump pyomo block contains characteristics of a pump.
"""


import pyomo.environ as pyo
from pyomo.network import *


# data: A, B, n_n, eff, Qmax, Qnom, Pmax
# init_data: Q(t), H(t), n(t), Pe(t)

def Pump(b, t, data, init_data):

    """
    Frequency controllable hydraulic pump.
    
    Applies the hydraulic head equation to a flow, with the ability to change its rotational speed, limited by a maximum power:
    
    .. math:
        H(t) = (n(t)/n_n)^2\cdot A - B \cdot Q(t)^2
        
    with :math:`A` and :math:`B` the characteristic coefficients that define the behaviour of the pump (i.e its curve).
    
    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: data ``dict``
    :param init_data: init_data ``dict``
        
    data
         - 'A': Constant characteristic coefficient of the pump :math:`A`
         - 'B': Quadratic characteristic coefficient of the pump :math:`B`
         - 'n_n': Nominal rotational speed :math:`n_n`
         - 'Qnom': Nominal flow :math:`Q_n`
         - 'eff': Efficiency at the nominal operating point in p.u. :math:`\eta`
         - 'Qmax': Maximum allowed flow :math:`Q_{max}`
         - 'Pmax': Maximum allowed power :math:`P_{max}`
         
    init_data
         - 'Q': Flow :math:`Q(t)` as a ``list``
         - 'H': Head :math:`H(t)` as a ``list``
         - 'n': Rotational speed :math:`n(t)` as a ``list``
         - 'Pe': Electrical power :math:`P_e(t)` as a ``list``
    
    Pyomo declarations    
        - Parameters:
            - A
            - B
            - Qnom
            - n_n
            - eff
        - Variables:
            - Qin (t) bounded :math:`Q_{in} \in [-Q_{max}, 0]`
            - Qout (t) bounded :math:`Q_{out} \in [0, Q_{max}]`
            - H (t) bounded :math:`H \ge 0`
            - n (t) bounded :math:`n \ge 0`
            - Ph (t) bounded :math:`P_h \in [0, P_{max}]`
            - Pe (t) bounded :math:`P_e \in [0, P_{max}]`
        - Ports:
            - port_Qin @ Qin with 'Q' as ``Extensive``
            - port_Qout @ Qout with 'Q' as ``Extensive``
            - port_P @ Pe with 'P' as ``Extensive``
            - port_H @ H with 'H' as ``Equality``
        - Constraints:
            - c_Q: :math:`Q_{in}(t) = - Q_{out}(t)`
            - c_H: :math:`H(t) = (n(t)/n_n)^2\cdot A - B \cdot Q_{out}(t)^2`
            - c_Ph: :math:`P_h(t) = 9810\cdot H(t)\cdot Q_{out}`
            - c_Pe: :math:`P_e(t) = P_h(t)/\eta`
    """
    
    # Parameters
    b.A = pyo.Param(initialize=data['A'])
    b.B = pyo.Param(initialize=data['B'])
    b.Qnom = pyo.Param(initialize=data['Qnom'])
    b.n_n = pyo.Param(initialize=data['n_n'])
    b.eff = pyo.Param(initialize=data['eff'])
    
    # Variables
    b.Qin  = pyo.Var(t, initialize=init_data['Q'], bounds=(-data['Qmax'], 0), within=pyo.NonPositiveReals)
    b.Qout = pyo.Var(t, initialize=init_data['Q'], bounds=(0, data['Qmax']), within=pyo.NonNegativeReals)
    b.H    = pyo.Var(t, initialize=init_data['H'], within=pyo.NonNegativeReals) 
    b.n    = pyo.Var(t, initialize=init_data['n'], within=pyo.NonNegativeReals) 
    b.Ph   = pyo.Var(t, initialize=init_data['Pe'], bounds=(0, data['Pmax']), within=pyo.NonNegativeReals)
    b.Pe   = pyo.Var(t, initialize=init_data['Pe'], bounds=(0, data['Pmax']), within=pyo.NonNegativeReals)
    
    # Ports
    b.port_Qin = Port(initialize={'Q': (b.Qin, Port.Extensive)})
    b.port_Qout = Port(initialize={'Q': (b.Qout, Port.Extensive)})
    b.port_P = Port(initialize={'P': (b.Pe, Port.Extensive)})
    b.port_H = Port(initialize={'H': (b.H, Port.Equality)})
    
    # Constraints
    def Constraint_Q(_b, _t):
        return _b.Qin[_t] == -_b.Qout[_t]
    b.c_Q = pyo.Constraint(t, rule = Constraint_Q)
    
    def Constraint_H(_b, _t):
        return _b.H[_t] == (_b.n[_t]/_b.n_n)**2*_b.A - _b.B*_b.Qout[_t]**2
    b.c_H = pyo.Constraint(t, rule = Constraint_H)
    
    def Constraint_Ph(_b, _t):
        return _b.Ph[_t] == 9810*_b.H[_t]*_b.Qout[_t]
    b.c_Ph = pyo.Constraint(t, rule = Constraint_Ph)
    
    def Constraint_Pe(_b, _t):
        return _b.Pe[_t] == _b.Ph[_t]/_b.eff
    b.c_Pe = pyo.Constraint(t, rule = Constraint_Pe)
