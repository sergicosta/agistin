# -*- coding: utf-8 -*-
"""
AGISTIN project 

...\Devices\ReservoirsClass.py

Class Reservoirs contains characteristics of a reservoir.
"""

from Devices.ElementsClass import Elements
import pyomo.environ as pyo

class Sources(Elements):
    def __init__(self):
        super().__init__()
        
        self.Q = {} # --> {(id,t):val}
        
        
    def add(self, id_elem, l_t, Q):
        super().add(id_elem)
        self.Q.update({(id_elem,t): Q[t] for t in l_t})
        
        
    def builder(self, model):
        
        model.i_src = pyo.Set(initialize=self.id)
        
        # PARAMS
        model.src_Q = pyo.Var(model.i_src, model.t, initialize=self.Q, within=pyo.NonNegativeReals)
        
        # VARIABLES
        