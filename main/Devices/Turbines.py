"""
AGISTIN project 

.\Devices\Turbines.py

Pump pyomo block contains characteristics of a turbine.
"""


import pyomo.environ as pyo
from pyomo.network import *


# data: eff
# init_data: Q(t), H(t), n(t), Pe(t)

def Turbine(b, t, data, init_data):
    
    # Parameters
    b.eff = pyo.Param(initialize=data['eff'])
    
    # Variables
    b.Pdim = pyo.Var(initialize=0, within=pyo.NonNegativeReals)
    b.Qin  = pyo.Var(t, initialize=init_data['Q'], within=pyo.NonPositiveReals)
    b.Qout = pyo.Var(t, initialize=init_data['Q'], within=pyo.NonNegativeReals)
    b.H    = pyo.Var(t, initialize=init_data['H'], within=pyo.NonNegativeReals) 
    b.Ph   = pyo.Var(t, initialize=init_data['Pe'], within=pyo.NonPositiveReals)
    b.Pe   = pyo.Var(t, initialize=init_data['Pe'], within=pyo.NonPositiveReals)
    
    # Ports
    b.port_Qin  = Port(initialize={'Q': (b.Qin, Port.Extensive)})
    b.port_Qout = Port(initialize={'Q': (b.Qout, Port.Extensive)})
    b.port_P    = Port(initialize={'P': (b.Pe, Port.Extensive)})
    b.port_H    = Port(initialize={'H': (b.H, Port.Equality)})
    
    # Constraints
    def Constraint_Q(_b, _t):
        return -_b.Qin[_t] == _b.Qout[_t]
    b.c_Q = pyo.Constraint(t, rule = Constraint_Q)
    
    def Constraint_Ph(_b, _t):
        return _b.Ph[_t] == 9810*_b.H[_t]*_b.Qin[_t]
    b.c_Ph = pyo.Constraint(t, rule = Constraint_Ph)
    
    def Constraint_Pe(_b, _t):
        return _b.Pe[_t] == _b.Ph[_t]*_b.eff
    b.c_Pe = pyo.Constraint(t, rule = Constraint_Pe)

    def Constraint_Pdim(_b, _t):
        return _b.Pdim >= -_b.Pe[_t]
    b.c_Pdim = pyo.Constraint(t, rule = Constraint_Pdim)
