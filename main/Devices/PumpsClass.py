# -*- coding: utf-8 -*-
"""
AGISTIN project 

...\Devices\PumpsClass.py

Class Pumps contains characteristics of a pump.
"""

from Devices.ElementsClass import Elements
import pyomo.environ as pyo

class Pumps(Elements):
    
    def __init__(self):
        super().__init__()
        
        self.A = {} # --> {id:val}
        self.B = {}
        self.rpm_nom = {}
        
        # init variables
        self.init_H = {} # --> {(id,t):val}
        self.init_Q = {} # --> {(id,t):val}
        self.init_n = {} # --> {(id,t):val}
        
        
    def add(self, id_elem, A, B, rpm_nom, init_H, init_Q, l_t, init_n=None):
        super().add(id_elem)
        
        self.A[id_elem] = A
        self.B[id_elem] = B
        self.rpm_nom[id_elem] = rpm_nom
        
        # init variables
        self.init_H.update({(id_elem,t): init_Q for t in l_t})
        self.init_Q.update({(id_elem,t): init_H for t in l_t})
        if init_n==None:
            init_n = rpm_nom
        self.init_n.update({(id_elem,t): init_n for t in l_t})
        
        
    def builder(self, model):
        model.i_pumps = pyo.Set(initialize=self.id)
        # PARAMS
        model.pumps_A = pyo.Param(model.i_pumps, initialize=self.A, within=pyo.NonNegativeReals)
        model.pumps_B = pyo.Param(model.i_pumps, initialize=self.B, within=pyo.NonNegativeReals)
        model.pumps_rpm_nom = pyo.Param(model.i_pumps, initialize=self.rpm_nom, within=pyo.NonNegativeReals)
        # VARIABLES
        model.pumps_H = pyo.Var(model.i_pumps, model.t, initialize=self.init_H, within=pyo.NonNegativeReals)
        model.pumps_Q = pyo.Var(model.i_pumps, model.t, initialize=self.init_Q, within=pyo.NonNegativeReals)
        model.pumps_n = pyo.Var(model.i_pumps, model.t, initialize=self.init_n, within=pyo.NonNegativeReals)
        
        
    def builderConstr(self, model):
        model.Constraint_Qmax_Pumps = pyo.Constraint(model.i_pumps, model.t, rule=Constraint_Qmax_Pumps)
        model.Constraint_H_Pumps = pyo.Constraint(model.i_pumps, model.t, rule=Constraint_H_Pumps)
    
# CONSTRAINTS

def Constraint_Qmax_Pumps(m, i_pumps, t):
    return m.pumps_Q[i_pumps, t] <= m.pumps_Qmax[i_pumps, t]

def Constraint_H_Pumps(m, i_pumps, t): 
 	return m.pumps_H[i_pumps, t] == ((m.pumps_n[i_pumps, t]/m.pumps_rpm_nom[i_pumps])**2) * m.pumps_A[i_pumps] - m.pumps_B[i_pumps]*(m.pumps_Q[i_pumps,t])**2
