"""
AGISTIN project 

.\Devices\Pipes.py

Pipe pyomo block contains characteristics of a pipe.
"""


import pyomo.environ as pyo
from pyomo.network import *


# data: K, Qmax
# init_data: H0(t), Q(t), H(t)

def Pipe(b, t, data, init_data):
    
    # Parameters
    b.K = pyo.Param(initialize=data['K'])
    
    # Variables
    b.Q = pyo.Var(t, initialize=init_data['Q'], bounds=(-data['Qmax'], data['Qmax']), within=pyo.NonNegativeReals)
    b.H = pyo.Var(t, initialize=init_data['H'], within=pyo.NonNegativeReals) 
    b.zlow = pyo.Var(t, initialize=init_data['zlow'], within=pyo.NonNegativeReals) 
    b.zhigh = pyo.Var(t, initialize=init_data['zhigh'], within=pyo.NonNegativeReals) 
    b.H0 = pyo.Var(t, initialize=init_data['H0'], within=pyo.NonNegativeReals)
    
    # Ports
    b.port_Q = Port(initialize={'Q': (b.Q, Port.Extensive)})
    b.port_H = Port(initialize={'H': (b.H, Port.Equality)})
    b.port_zlow = Port(initialize={'z': (b.zlow, Port.Equality)})
    b.port_zhigh = Port(initialize={'z': (b.zhigh, Port.Equality)})
    
    # Constraints
    def Constraint_H(_b, _t):
        return _b.H[_t] == _b.H0[_t] + _b.K*_b.Q[_t]**2
    b.c_H = pyo.Constraint(t, rule = Constraint_H)
    
    def Constraint_H0(_b, _t):
        return _b.H0[_t] == _b.zhigh[_t] - _b.zlow[_t]
    b.c_H0 = pyo.Constraint(t, rule = Constraint_H0)



# data: H0, K, Qmax
# init_data: Q(t), H(t)

def Pipe_Ex0(b, t, data, init_data):
    
    # Parameters
    b.H0 = pyo.Param(initialize=init_data['H0'])
    b.K = pyo.Param(initialize=data['K'])
    
    # Variables
    b.Q = pyo.Var(t, initialize=init_data['Q'], bounds=(-data['Qmax'], data['Qmax']), within=pyo.NonNegativeReals)
    b.H = pyo.Var(t, initialize=init_data['H'], within=pyo.NonNegativeReals) 
    
    # Ports
    b.port_Q = Port(initialize={'Q': (b.Q, Port.Extensive)})
    b.port_H = Port(initialize={'H': (b.H, Port.Equality)})
    
    # Constraints
    def Constraint_H(_b, _t):
        return _b.H[_t] == _b.H0 + _b.K*_b.Q[_t]**2
    b.c_H = pyo.Constraint(t, rule = Constraint_H)