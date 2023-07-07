"""
AGISTIN project 

.\Devices\PumpsClass.py

Pump pyomo block contains characteristics of a pump.
"""


import pyomo.environ as pyo
from pyomo.network import *


# data: A, B, Pmax, Qmin, Qmax, Qnom, rpm_nom, nu_num
# init_data: P, H, Q, n

def Pump(b, t, data, init_data):
    
    # Parameters
    b.A = pyo.Param(initialize=data['A'])
    b.B = pyo.Param(initialize=data['B'])
    b.Pmax = pyo.Param(initialize=data['Pmax'])
    b.Qmin = pyo.Param(initialize=data['Qmin'])
    b.Qmax = pyo.Param(initialize=data['Qmax'])
    b.Qnom = pyo.Param(initialize=data['Qnom'])
    b.n_nom = pyo.Param(initialize=data['rpm_nom'])
    b.nu_nom = pyo.Param(initialize=data['nu_nom'])
    
    # Variables
    b.P = pyo.Var(t, initialize=init_data['P'])#, bounds=(0.0, b.Pmax), domain=pyo.NonNegativeReals)
    b.H = pyo.Var(t, initialize=init_data['H'], domain=pyo.NonNegativeReals)
    b.Q = pyo.Var(t, initialize=init_data['Q'], bounds=(b.Qmin, b.Qmax), domain=pyo.NonNegativeReals)
    b.n = pyo.Var(t, initialize=init_data['n'], domain=pyo.NonNegativeReals)
    
    # Ports
    b.outlet = Port(initialize={'q': (b.Q, Port.Extensive)})
    b.inlet = Port(initialize={'p': (b.P, Port.Extensive)})
    
    def Constraint_H_Pumps(_b, _t): 
     	return _b.H[_t] == ((b.n[_t]/b.n_nom)**2)*b.A - b.B*(b.Q[_t])**2
    b.Constraint_H_Pumps = pyo.Constraint(t, rule=Constraint_H_Pumps)