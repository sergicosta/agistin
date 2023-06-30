# -*- coding: utf-8 -*-
"""
AGISTIN project 

...\Devices\ReservoirsClass.py

Class Reservoirs contains characteristics of a reservoir.
"""

from Devices.Elements import Elements
import pyomo.environ as pyo

class Reservoirs(Elements):
    
    def __init__(self):
        super().__init__()
        
        self.W_0 = {}
        self.W_min = {}
        self.W_max = {}
        self.z_min = {}
        self.z_max = {}
        self.id_in = {}
        self.id_out = {}
        self.Q_cons = {} # --> {(id,t):val}
        
        # init variables
        self.init_W = {} # --> {(id,t):val}
        
        
    def add(self, id_elem, id_in, id_out, W_0, W_min, W_max, z_min, z_max, init_W, Q_cons, l_t):
        super().add(id_elem)
        
        self.W_0[id_elem] = W_0
        self.W_min[id_elem] = W_min
        self.W_max[id_elem] = W_max
        self.z_min[id_elem] = z_min
        self.z_max[id_elem] = z_max
        self.id_in[id_elem] = id_in
        self.id_out[id_elem] = id_out
        self.Q_cons.update({(id_elem,t): Q_cons[t] for t in l_t}) # --> {(id,t):val}

        # init variables
        self.init_W.update({(id_elem,t): init_W for t in l_t})
        
        
    def builder(self, model):
        model.i_res = pyo.Set(initialize=self.id)
        # PARAMS
        model.res_W_0 = pyo.Param(model.i_res, initialize=self.W_0, within=pyo.NonNegativeReals)
        model.res_W_min = pyo.Param(model.i_res, initialize=self.W_min, within=pyo.NonNegativeReals)
        model.res_W_max = pyo.Param(model.i_res, initialize=self.W_max, within=pyo.NonNegativeReals)
        model.res_z_min = pyo.Param(model.i_res, initialize=self.z_min, within=pyo.NonNegativeReals)
        model.res_z_max = pyo.Param(model.i_res, initialize=self.z_max, within=pyo.NonNegativeReals)
        model.res_id_in = pyo.Param(model.i_res, initialize=self.id_in, within=pyo.NonNegativeReals)
        model.res_id_out = pyo.Param(model.i_res, initialize=self.id_out, within=pyo.NonNegativeReals)
        model.res_Q_cons = pyo.Param(model.i_res, model.t, initialize=self.Q_cons, within=pyo.NonNegativeReals)
        # VARIABLES
        model.res_W = pyo.Var(model.i_res, model.t, initialize=self.init_W, within=pyo.NonNegativeReals)