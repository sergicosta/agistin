"""
AGISTIN project 

.\Devices\Pumps.py

Pump pyomo block contains characteristics of a pump.
"""


import pyomo.environ as pyo
from pyomo.network import *


# data: A, B, n_n, eff, Qmax, Qnom, Pmax
# init_data: Q(t), H(t), n(t), Pe(t)

def Pump(b, t, data, init_data):
    
    # Parameters
    b.A = pyo.Param(initialize=data['A'])
    b.B = pyo.Param(initialize=data['B'])
    b.Qnom = pyo.Param(initialize=data['Qnom'])
    b.n_n = pyo.Param(initialize=data['n_n'])
    b.eff = pyo.Param(initialize=data['eff'])
    
    # Variables
    b.Q = pyo.Var(t, initialize=init_data['Q'], bounds=(0, data['Qmax']), within=pyo.NonNegativeReals)
    b.H = pyo.Var(t, initialize=init_data['H'], within=pyo.NonNegativeReals) 
    b.n = pyo.Var(t, initialize=init_data['n'], within=pyo.NonNegativeReals) 
    b.Ph = pyo.Var(t, initialize=init_data['Pe'], bounds=(0, data['Pmax']), within=pyo.NonNegativeReals)
    b.Pe = pyo.Var(t, initialize=init_data['Pe'], bounds=(0, data['Pmax']), within=pyo.NonNegativeReals)
    
    # Ports
    b.port_Q = Port(initialize={'Q': (b.Q, Port.Extensive)})
    b.port_P = Port(initialize={'P': (b.Pe, Port.Extensive)})
    b.port_H = Port(initialize={'H': (b.H, Port.Equality)})
    
    # Constraints
    def Constraint_H(_b, _t):
        return _b.H[_t] == (_b.n[_t]/_b.n_n)**2*_b.A - _b.B*_b.Q[_t]**2
    b.c_H = pyo.Constraint(t, rule = Constraint_H)
    
    def Constraint_Ph(_b, _t):
        return _b.Ph[_t] == 9810*_b.H[_t]*_b.Q[_t]
    b.c_Ph = pyo.Constraint(t, rule = Constraint_Ph)
    
    def Constraint_Pe(_b, _t):
        return _b.Pe[_t] == _b.Ph[_t]/_b.eff
    b.c_Pe = pyo.Constraint(t, rule = Constraint_Pe)
