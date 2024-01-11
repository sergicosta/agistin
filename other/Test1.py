# -*- coding: utf-8 -*-

import pyomo.environ  as pyo
from pyomo.network import *

m = pyo.ConcreteModel()
m.t = pyo.Set(initialize=list(range(1,6)))
m.b1 = pyo.Block()

data = {'A':1, 'B':0.02, 'Pmax':1e5, 'Qmin':2, 'Qmax':800, 'Qnom':500, 'rpm_nom':1425, 'nu_nom':0.8}
init_data = {'P':{k:8e4 for k in list(m.t)}, 
             'H':{k:10 for k in list(m.t)}, 
             'Q':{k:200 for k in list(m.t)}, 
             'n':{k:1400 for k in list(m.t)}}

Pump(m.b1, m.t, data, init_data)

#%%

import pyomo.environ  as pyo
from pyomo.network import *

m = pyo.ConcreteModel()
m.t = pyo.Set(initialize=list(range(1,6)))

def source_block(b, t, a):
    b.p_out = pyo.Var(t, bounds=a, domain=pyo.NonNegativeReals)
    b.outlet = Port(initialize={'p': (b.p_out, Port.Extensive)})

def load_block(b, t):
    b.p_in = pyo.Var(t, domain=pyo.NonNegativeReals)
    b.W0 = pyo.Var(t, domain=pyo.NonNegativeReals)
    b.W = pyo.Var(t, domain=pyo.NonNegativeReals)
    b.inlet = Port(initialize={'p': (b.p_in, Port.Extensive)})
    def linking_rule(_b, _t):
        if _t == 1:
            return _b.W0[_t] == 5
        else:
            return _b.W0[_t] == _b.W[_t-1]
    b.linking = pyo.Constraint(t, rule=linking_rule)
    def update(_b, _t):
        return _b.W[_t] == _b.W0[_t] + _b.p_in[_t]
    b.update = pyo.Constraint(t, rule=update)

m.b1 = pyo.Block()
m.b2 = pyo.Block()
m.b3 = pyo.Block()

source_block(m.b1, m.t, (1,3))
source_block(m.b2, m.t, (5,8))
load_block(m.b3, m.t)

m.b13 = Arc(source=m.b1.outlet, destination=m.b3.inlet)
m.b23 = Arc(source=m.b2.outlet, destination=m.b3.inlet)

pyo.TransformationFactory("network.expand_arcs").apply_to(m)

def obj_fun(m):
	return sum(m.b3.p_in[t] for t in list(m.t))
m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)

instance = m.create_instance()
solver = pyo.SolverFactory('ipopt')
solver.solve(instance, tee=True)

instance.b1.p_out.pprint()
instance.b2.p_out.pprint()
instance.b3.p_in.pprint()
instance.b3.W.pprint()
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

