# -*- coding: utf-8 -*-
"""
AGISTIN project 

./Devices/PipesClass.py

Class Pipes contains characteristics of a pipe.
"""

from Devices.Elements import Elements
import pyomo.environ as pyo

class Pipes(Elements):
    
    def __init__(self):
        super().__init__()
        
        self.K = {}
        self.Qmax = {}
        self.orig = {}
        self.end = {}
        
        # init variables
        self.init_H = {} # --> {(id,t):val}
        self.init_Q = {} # --> {(id,t):val}
        
        
    def add(self, id_elem, K, id_in, id_out, Qmax=1e6):
        super().add(id_elem)
        
        self.K[id_elem] = K
        self.Qmax[id_elem] = Qmax
        self.id_in[id_elem] = id_in
        self.id_out[id_elem] = id_out
        
        
    def builder(self, model):
        model.i_pipe = pyo.Set(initialize=self.id)
        # PARAMS
        model.pipes_K = pyo.Param(model.i_pipe, initialize=self.K, within=pyo.NonNegativeReals)
        model.pipes_Qmax = pyo.Param(model.i_pipe, initialize=self.Qmax, within=pyo.NonNegativeReals)
        # VARIABLES
        model.pipes_Q = pyo.Var(model.i_pipe, model.t, initialize=self.init_Q, within=pyo.NonNegativeReals)
        model.pipes_H = pyo.Var(model.i_pipe, model.t, initialize=self.init_H, within=pyo.NonNegativeReals)