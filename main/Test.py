import pyomo.environ as pyo
from pyomo.network import *
from Devices.Reservoirs import Reservoir
from Devices.Sources import Source

l_t = list(range(5))

m = pyo.ConcreteModel()

m.t = Set(initialize=l_t)


m.Source1 = Block()
m.Source2 = Block()
m.Reservoir1 = Block()

data_s1 = {'Q':[1,1,1,1,1]}
data_r1 = {'W0':5, 'Wmin':0, 'Wmax':20}

i_data_r1 = {'Qin':[0,0,0,0,0], 'W':[5,5,5,5,5]}

Source(m.Source1, m.t, data_s1)
Source(m.Source2, m.t, data_s1)
Reservoir(m.Reservoir1, m.t, data_r1, i_data_r1)

m.s1r1 = Arc(source=m.Source1.outlet, destination=m.Reservoir1.inlet)
m.s2r1 = Arc(source=m.Source2.outlet, destination=m.Reservoir1.inlet)

TransformationFactory("network.expand_arcs").apply_to(m)

m.pprint()


def obj_fun(m):
	return 0#sum((m.P_g0[t])*m.cost_elec[t] for t in l_t) + ( + m.P_pv_dim0*m.cost_pv_inst[0])
m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)

instance = m.create_instance()
solver = pyo.SolverFactory('ipopt')
solver.solve(instance, tee=True)

instance.Source1.Q.pprint()
instance.Source2.Q.pprint()
instance.Reservoir1.Qin.pprint()
instance.Reservoir1.W.pprint()