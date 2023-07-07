"""
AGISTIN project 

./Devices/Pipes.py

Pipe pyomo block contains characteristics of a pipe.
"""


import pyomo.environ as pyo
from pyomo.network import *


# data: H0, K, Qmax
# init_data: Q(t), H(t)

def Pipe(b, t, data, init_data):
    
    # Parameters
    b.H0 = pyo.Param(initialize=data['H0'])
    b.K = pyo.Param(initialize=data['K'])
    
    # Variables
    b.Q = pyo.Var(t, initialize=init_data['Q'], bounds=(-data['Qmax'], data['Qmax']), within=pyo.NonNegativeReals)
    b.H = pyo.Var(t, initialize=init_data['H'], within=pyo.NonNegativeReals) 
    
    # Ports
    b.inlet = Port(initialize={'Q': (b.Q, Port.Extensive)})
    b.outlet = Port(initialize={'Q': (b.Q, Port.Extensive)})
    
    # Constraints
    def Constraint_H(_b, _t):
        return _b.H[_t] == _b.H0 + _b.K*_b.Q[_t]**2
    b.c_H = pyo.Constraint(t, rule = Constraint_H)