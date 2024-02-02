# AGISTIN project
# .\Devices\Evaporation.py

'''
Weather pyomo block contains the model of evaporation based in
Penman method and rainfall.
'''

import pyomo.environ as pyo
from pyomo.network import Arc, Port
from pyomo.core import *

def Evaporation(b, t, data, init_data):
    """
    Evaporation and rainfall.
    
    Add and rest  volume to the reservoirs given the evaporation and rainfall.
        
    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: data ``dict``
    :param init_data: init_data ``dict``
    
    data
         - 'Wind': The wind measured :math:`Wind(t)`
         - 'T': The temperature measured :math:`T(t)`
         - 'G': The average radiation measured :math:`G(t)`
         - 'P': The average atmospheric pressure measured :math:`P(t)`
         - 'H': The average humidity measured :math:`H(t)`
         - 'Rain': The accumulated rain in an interval of time :math:`Rain(t)`
         - 'Latent': Latent heat constant of the water :math:`Latent`
         - 'Hsum': The hours of sun in a day :math:`\sum(Hours_{i})`


         
    init_data
         - 'Q': Flow :math:`Q(t)` as a ``list``
    
    Pyomo declarations    
        - Parameters: 
            - Wind
            - T
            - G
            - P
            - H
            - Rain
            - Latent
            - Hsum
            - slope
            - ea
            - es
            - psy
            - Evap
            - Q
        - Variables: 
            - Q (t) bounded :math:`Q(t) \in [-\infty, \infty]`
        - Ports: 
            - port_Qout @ Qout (Extensive)
        - Constraints:
            - c_Q: :math:`Q_{out}(t) = Q(t)`
    

    """
    #Parameters
    
    b.wind = pyo.Param(t,initialize=data['Wind'])
    b.T = pyo.Param(t,initialize=data['Temperature'])
    b.G = pyo.Param(t,initialize=data['Radiation'])
    b.P = pyo.Param(t,initialize=data['Pressure'])
    b.H = pyo.Param(t,initialize=data['Humitat'])
    b.Rain = pyo.Param(t,initialize=data['Rain'])
    b.A = pyo.Param(initialize=data['Area'])
    b.Latent = pyo.Param(initialize=data['Latent'])
    b.hsum = pyo.Param(initialize=data['hsum'])
    
    #Penman Method:
        
    def Calculate_slope(_b, _t):
        return (4098*(0.06108*2.718**(17.27*_b.T[_t]/(_b.T[_t]+273.3))))/(_b.T[_t]+273.3)**2
    b.slope = pyo.Param(t,initialize=Calculate_slope)
    
    def Calculate_ea(_b,_t):
        return(_b.H[_t]/100)*0.6108*2.718**(17.27*_b.T[_t]/(_b.T[_t]+273.3))
    b.ea = pyo.Param(t,initialize=Calculate_ea)
    
    def Calculate_es(_b,_t):
        return 0.6108*2.718**(17.27*_b.T[_t]/(_b.T[_t]+273.3))
    b.es = pyo.Param(t,initialize=Calculate_es)
    
    def Calculate_psycht(_b,_t):
        return 0.000665*_b.P[_t]/10
    b.psy = pyo.Param(t,initialize=Calculate_psycht)
    
    def Calculate_evap(_b,_t):
        return 1/_b.hsum*1/997*(86.4*(_b.slope[_t]/(_b.slope[_t]+_b.psy[_t]))*(_b.G[_t]/_b.Latent)+((_b.psy[_t]/(_b.psy[_t]+_b.slope[_t]))*0.26*(0.5+0.54*_b.wind[_t])*(_b.ea[_t]-_b.es[_t])*10))
    b.evap = pyo.Param(t, initialize=Calculate_evap)
    
    def Calculate_Q(_b,_t):
        return (_b.Rain[_t]-_b.evap[_t])*_b.A/1000
    b.Q = pyo.Param(t,within=pyo.Reals,initialize=Calculate_Q)
    
    #Variables
    b.Qout = pyo.Var(t, initialize=Calculate_Q, within=pyo.Reals)    
     
    #Ports
    b.port_Qout = Port(initialize={'Q': (b.Qout, Port.Extensive)})
    
    #Constraint
    
    def Constraint_Qout(_b, _t):
        return _b.Qout[_t] == _b.Q[_t]
    b.c_Qout = pyo.Constraint(t, rule=Constraint_Qout)
    



    