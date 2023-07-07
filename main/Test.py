import pyomo.environ as pyo
from pyomo.network import *
from Devices.Reservoirs import Reservoir
from Devices.Sources import Source
from Devices.Pipes import Pipe
from Devices.Pumps import Pump

l_t = list(range(5))

m = pyo.ConcreteModel()

m.t = pyo.Set(initialize=l_t)


m.Source1 = pyo.Block()
m.Source2 = pyo.Block()
m.Reg = pyo.Block()
m.Pipe1 = pyo.Block()
m.Reservoir1 = pyo.Block()
m.Pump1 = pyo.Block()

data_s1 = {'Q':[1,1,1,1,1]}
data_reg = {'Q':[2,1,1,1,1]}

data_c1 = {'H0':20, 'K':0.05, 'Qmax':50}
init_c1 = {'Q':[0,0,0,0,0], 'H':[20,20,20,20,20]}

data_r1 = {'W0':5, 'Wmin':0, 'Wmax':20}
init_r1 = {'Q':[0,0,0,0,0], 'W':[5,5,5,5,5]}

data_p1 = {'A':50, 'B':0.1, 'n_n':1450, 'eff':0.9, 'Qmax':20, 'Qnom':5, 'Pmax':9810*20*20}
init_p1 = {'Q':[0,0,0,0,0], 'H':[20,20,20,20,20], 'n':[1450,1450,1450,1450,1450], 'Pe':[9810*20*20,9810*20*20,9810*20*20,9810*20*20,9810*20*20]}

# Source(m.Source1, m.t, data_s1)
# Source(m.Source2, m.t, data_s1)
Source(m.Reg, m.t, data_reg)
Pipe(m.Pipe1, m.t, data_c1, init_c1)
Reservoir(m.Reservoir1, m.t, data_r1, init_r1)
Pump(m.Pump1, m.t, data_p1, init_p1)

# m.s1c1 = Arc(ports=(m.Source1.port_Q, m.Pipe1.port_Q), directed=True)
# m.s2c1 = Arc(ports=(m.Source2.port_Q, m.Pipe1.port_Q), directed=True)
m.regr1 = Arc(ports=(m.Reg.port_Q, m.Reservoir1.port_Qout), directed=True)
m.c1r1 = Arc(ports=(m.Pipe1.port_Q, m.Reservoir1.port_Qin), directed=True)
m.p1c1_Q = Arc(ports=(m.Pump1.port_Q, m.Pipe1.port_Q), directed=True)
m.p1c1_H = Arc(ports=(m.Pump1.port_H, m.Pipe1.port_H), directed=True)

pyo.TransformationFactory("network.expand_arcs").apply_to(m)

m.pprint()


def obj_fun(m):
	return sum(m.Pump1.Pe[t] for t in l_t)#sum((m.P_g0[t])*m.cost_elec[t] for t in l_t) + ( + m.P_pv_dim0*m.cost_pv_inst[0])
m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)

instance = m.create_instance()
solver = pyo.SolverFactory('ipopt')
solver.solve(instance, tee=True)

instance.Reservoir1.Qin.pprint()
instance.Reservoir1.Qout.pprint()
instance.Reservoir1.W.pprint()
instance.Pipe1.Q.pprint()
instance.Pipe1.H.pprint()
instance.Pump1.Q.pprint()
instance.Pump1.H.pprint()
instance.Pump1.n.pprint()
instance.Pump1.Pe.pprint()