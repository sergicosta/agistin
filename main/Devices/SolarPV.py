"""
AGISTIN project 

.\Devices\SolarPV.py

SolarPV pyomo block contains characteristics of a solar PV plant.
"""


import pyomo.environ as pyo
from pyomo.network import *


# data: Pinst, Pmax, forecast(t)
# init_data: None

def SolarPV(b, t, data, init_data=None):
    
    # Parameters
    b.Pinst = pyo.Param(initialize=data['Pinst'])
    b.forecast = pyo.Param(t, initialize=data['forecast'])
    
    # Variables
    b.P = pyo.Var(t, initialize={k: -data['Pinst']*data['forecast'][k] for k in range(len(t))} , bounds=(-data['Pmax'],0), domain=pyo.Reals)
    b.Pdim = pyo.Var(initialize=0 , bounds=(0, data['Pmax']-data['Pinst']), domain=pyo.NonNegativeReals)
    
    # Ports
    b.port_P = Port(initialize={'P': (b.P, Port.Extensive)})
    
    # Constraints
    def Constraint_P(_b, _t):
        return _b.P[_t] >= -(b.Pinst+b.Pdim)*b.forecast[_t]
    b.Constraint_P = pyo.Constraint(t, rule=Constraint_P)