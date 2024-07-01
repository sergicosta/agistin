# AGISTIN project 
# .\Devices\SolarPV.py

"""
SolarPV pyomo block contains characteristics of a solar PV plant.
"""


import pyomo.environ as pyo
from pyomo.network import Arc, Port


# data: Pinst, Pmax, forecast(t)
# init_data: None

def SolarPV(b, t, data, init_data=None):

    """
    Expandable solar PV plant.
    
    Delivers power subject to a forecast :math:`f(t)` and its installed power :math:`P_{inst}`.
    The plant will be able to expand up to a :math:`P_{max}`.
    
    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: data ``dict``
    :param init_data: ``None``
        
    data
         - 'Pinst': Installed power :math:`P_{inst}`
         - 'Pmax': Maximum allowed power to be installed :math:`P_{max}`
         - 'forecast': Forecast as power in p.u. :math:`f(t)` as a ``list``
         - 'eff': Efficiency of the PV panels :math:`\eta`
    
    Pyomo declarations
        - Parameters: 
            - Pinst
            - forecast
            - eff
        - Variables: 
            - P (t) :math:`\in [-P_{max}, 0]`
            - Pdim :math:`\in [0, P_{max}-P_{inst}]`
        - Ports: 
            - port_P @ P as ``Extensive``
        - Constraints: 
            - c_P: :math:`P(t) \ge -(P_{inst}+P_{dim})\cdot f(t)\cdot \eta`
    """
    
    # Parameters
    b.Pinst = pyo.Param(initialize=data['Pinst'])
    b.forecast = pyo.Param(t, initialize=data['forecast'])
    b.eff = pyo.Param(initialize=data['eff'])
    
    # Variables
    b.P = pyo.Var(t, initialize={k: -data['Pinst']*data['forecast'][k] for k in range(len(t))} , bounds=(-data['Pmax'],0), domain=pyo.Reals)
    b.Pdim = pyo.Var(initialize=0 , bounds=(0, data['Pmax']), domain=pyo.NonNegativeReals)
    
    # Ports
    b.port_P = Port(initialize={'P': (b.P, Port.Extensive)})
    
    # Constraints
    def Constraint_P(_b, _t):
        return _b.P[_t] == -(_b.Pinst+_b.Pdim)*_b.forecast[_t]*_b.eff
    b.Constraint_P = pyo.Constraint(t, rule=Constraint_P)