"""
AGISTIN project 

.\Devices\Reservoirs.py

Reservoir pyomo block contains characteristics of a reservoir.
"""


import pyomo.environ as pyo
from pyomo.network import *


# data: W0, Wmin, Wmax, zmin, zmax
# init_data: Q(t), W(t)

def Reservoir(b, t, data, init_data):
    
    def z(_b, w):
        return (w-_b.Wmin)/(_b.Wmax-_b.Wmin)*(_b.zmax-_b.zmin) + _b.zmin
    
    # Parameters
    b.W0 = pyo.Param(initialize=data['W0'])
    b.Wmin = pyo.Param(initialize=data['Wmin'])
    b.Wmax = pyo.Param(initialize=data['Wmax'])
    b.zmin = pyo.Param(initialize=data['zmin'])
    b.zmax = pyo.Param(initialize=data['zmax'])
    
    # Variables
    b.Qin = pyo.Var(t, initialize=init_data['Q'], within=pyo.NonNegativeReals)
    b.Qout = pyo.Var(t, initialize=init_data['Q'], within=pyo.NonNegativeReals)
    b.W = pyo.Var(t, initialize=init_data['W'], bounds=(data['Wmin'], data['Wmax']), within=pyo.NonNegativeReals)
    b.z = pyo.Var(t, initialize={k: z(b,init_data['W'][k]) for k in range(len(t))}, bounds=(data['zmin'], data['zmax']), within=pyo.NonNegativeReals) 
    
    # Ports
    b.port_Qin = Port(initialize={'Q': (b.Qin, Port.Extensive)})
    b.port_Qout = Port(initialize={'Q': (b.Qout, Port.Extensive)})
    b.port_z = Port(initialize={'z': (b.z, Port.Equality)})

    # Constraints
    def Constraint_W(_b, _t):
        if _t>0:
            return _b.W[_t] == _b.W[_t-1] + _b.Qin[_t] - _b.Qout[_t] # TODO: - Qloss - gamma
        else:
            return _b.W[_t] == _b.W0 + _b.Qin[_t] - _b.Qout[_t]
    b.c_W = pyo.Constraint(t, rule = Constraint_W)
    
    def Constraint_z(_b, _t):
        return b.z[_t] == z(_b, _b.W[_t])
    b.c_z = pyo.Constraint(t, rule=Constraint_z)
    