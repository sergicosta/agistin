# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:09:16 2023

@author: Sergi Costa
"""

from pyomo.environ import *
from pyomo.network import *
from pyomo.dae import *

l_t = list(range(5))

m = ConcreteModel()

m.t = Set(initialize=l_t)


def source_block(b, t, a):
    # b.t = Set(initialize=lt)
    b.p_out = Var(t, initialize=l_t)
    b.p_out2 = Var(t, initialize=l_t)
    # b.p2 = Var(t)
    b.outlet = Port(initialize={'p': (b.p_out, Port.Extensive)})
    
    def Constraint_p2(b, t):
        return b.p_out[t] == a
    b.c_p2 = Constraint(t, rule = Constraint_p2)

def load_block(b, t):
    # b.t = Set(initialize=lt)
    b.p_in = Var(t)
    
    b.inlet = Port(initialize={'p': (b.p_in, Port.Extensive)})
    
    # def Constraint_inlet(b, t):
    #     return b.p_in == sum(b.inlet)
    # b.c_inlet = Constraint(t, rule = Constraint_inlet)

# m.loads = Set()
# m.loads['Load1'] = Block()
# m.loads['Load2'] = Block()

m.b1 = Block()
m.b2 = Block()
m.b3 = Block()

source_block(m.b1, m.t, 1)
source_block(m.b2, m.t, 2)
# load_block(m.b2, m.t)
load_block(m.b3, m.t)

# m.b12 = Arc(source=m.b1.outlet, destination=m.b2.inlet)
# m.b13 = Arc(source=m.b1.outlet, destination=m.b3.inlet)

m.b13 = Arc(source=m.b1.outlet, destination=m.b3.inlet)
m.b23 = Arc(source=m.b2.outlet, destination=m.b3.inlet)

# TransformationFactory('dae.finite_difference').apply_to(m, nfe=1)
TransformationFactory("network.expand_arcs").apply_to(m)

m.pprint()


def obj_fun(m):
	return 0#sum((m.P_g0[t])*m.cost_elec[t] for t in l_t) + ( + m.P_pv_dim0*m.cost_pv_inst[0])
m.goal = Objective(rule=obj_fun, sense=minimize)

instance = m.create_instance()
solver = SolverFactory('ipopt')
solver.solve(instance, tee=True)

instance.b1.p_out.pprint()
instance.b2.p_out.pprint()
instance.b3.p_in.pprint()
#%%

import pyomo.environ as pyo


l_t = list(range(5))

model = pyo.ConcreteModel()

model.t = Set(initialize=l_t)


def Reservoir_block(b, t):
    
    b.W0 = pyo.Param(t, initialize=5)
    b.Wmin = pyo.Param(t, initialize=0)
    b.Wmax = pyo.Param(t, initialize=10)
    
    b.Qin = pyo.Var(t)
    b.W = pyo.Var(t)
    
    b.inlet = Port(initialize={'Q': (b.Qin, Port.Extensive)})


    def Constraint_W(b, t):
        


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


