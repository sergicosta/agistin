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
        
        
    def builderConstr(self, model):
        model.Constraint_Wmax_Res = pyo.Constraint(model.i_res, model.t, rule=Constraint_Wmax_Res)
        model.Constraint_Wmin_Res = pyo.Constraint(model.i_res, model.t, rule=Constraint_Wmin_Res)
        model.Constraint_W_Res = pyo.Constraint(model.i_res, model.t, rule=Constraint_W_Res)
        model.Constraint_Q_in = pyo.Constraint(model.i_res, model.t, rule=Constraint_Q_in)
    
# CONSTRAINTS

def Constraint_Wmax_Res(m, i_res, t):
    return m.res_W[i_res, t] <= m.res_W_max[i_res]

def Constraint_Wmin_Res(m, i_res, t):
    return m.res_W[i_res, t] >= m.res_W_min[i_res]

def Constraint_W_Res(m, i_res, t): 
	if t>0:
		return m.res_W[i_res, t] == m.res_W[i_res, t-1] + m.res_Q[i_res, t]**2 - m.res_Q_loss[i_res, t]
	else:
		return m.res_W[i_res, t] == m.res_W_0[i_res] + m.res_Q[i_res, t]**2 - m.res_Q_loss[i_res, t]
   	# TODO: add gamma, Dt
    # if t>0:
   	# 	return m.res_W[i_res, t] == m.res_W[i_res, t-1]*(1 -m.gamma_0[t]) - m.Dt*(m.Q_p0[t] + m.q_irr_0[t])
   	# else:
   	# 	return m.W_r0[t] == 142869 - m.Dt*(m.Q_p0[t] + m.q_irr_0[t])

list_q = [model.src_Q]
def aux_Q_in(m, i_res, t):
    aux_q = 0
    for q in list_q:
        aux_q += sum(q[e_id, t] for e_id in m.res_id_in[i_res]) 
    return aux_q
    
def Constraint_Q_in(m, i_res, t):
    return m.res_Q[i_res, t] == aux_Q_in(m, i_res, t)

