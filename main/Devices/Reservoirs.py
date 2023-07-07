"""
AGISTIN project 

.\Devices\Reservoirs.py

Reservoir pyomo block contains characteristics of a reservoir.
"""


import pyomo.environ as pyo
from pyomo.network import *


# data: W0, Wmin, Wmax
# init_data: Q(t), W(t)

def Reservoir(b, t, data, init_data):
    
    # Parameters
    b.W0 = pyo.Param(initialize=data['W0'])
    
    # Variables
    b.Q = pyo.Var(t, initialize=init_data['Q'], within=pyo.NonNegativeReals)
    b.W = pyo.Var(t, initialize=init_data['W'], bounds=(data['Wmin'], data['Wmax']), within=pyo.NonNegativeReals)
    
    # Ports
    b.port_Q = Port(initialize={'Q': (b.Qin, Port.Extensive)})

    # Constraints
    def Constraint_W(_b, _t):
        if _t>0:
            return _b.W[_t] == _b.W[_t-1] + _b.Q[_t] # TODO: - Qloss - gamma
        else:
            return _b.W[_t] == _b.W0 + _b.Q[_t]
    b.c_W = pyo.Constraint(t, rule = Constraint_W)