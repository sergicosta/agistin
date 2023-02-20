# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 16:29:21 2023
"""
import pyomo.environ as pyo

rho_g = 1e3*9.81
A = 900
B = 6
Ho = 305
K = 0.105
cte_v = 100
Q_tot = 4

model = pyo.AbstractModel()

l_cost = (1, 5, )
l_t = tuple(range(len(l_cost)))
model.t = pyo.Set(initialize=l_t)

model.cost_elec = pyo.Param(model.t, initialize=l_cost)

model.rho_g = pyo.Param(initialize=rho_g)
model.A     = pyo.Param(initialize=A)
model.B     = pyo.Param(initialize=B)
model.Ho    = pyo.Param(initialize=Ho)
model.K     = pyo.Param(initialize=K)
model.cte_v = pyo.Param(initialize=cte_v)
model.Q_tot = pyo.Param(initialize=Q_tot)

model.P = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(0, 6*3e6)) # , initialize={0:4e6, 1:6e6}
model.Q = pyo.Var(model.t, within=pyo.NonNegativeReals) #, initialize={0:6, 1:3}
model.H = pyo.Var(model.t, within=pyo.NonNegativeReals) #, initialize={0:200, 1:200}
model.a = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(0, 1)) # , initialize={0:0.5, 1:0.5}

def Constraint_P(m, t):
    return m.P[t] == m.rho_g * m.Q[t] * m.H[t]
model.Constraint_P = pyo.Constraint(model.t, rule=Constraint_P)

def Constraint_H(m, t):
    return m.H[t] == m.A - m.B*(m.Q[t])**2
model.Constraint_H = pyo.Constraint(model.t, rule=Constraint_H)

def Constraint_Hr_Hb(m, t):
    return m.Ho + (m.K + m.cte_v/m.a[t])*(m.Q[t])**2 == m.A - m.B*(m.Q[t])**2
model.Constraint_Hr_Hb = pyo.Constraint(model.t, rule=Constraint_Hr_Hb)

def Constraint_Q_total(m, t):
    return sum(m.Q[t] for t in l_t) == m.Q_tot
model.Constraint_Q_total = pyo.Constraint(model.t, rule=Constraint_Q_total)


def obj_fun(m):
    return sum(m.P[t]*m.cost_elec[t] for t in l_t)
model.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)


instance = model.create_instance()
solver = pyo.SolverFactory('baron')
# solver.options['max_iter']= 10000
results = solver.solve(instance)
results.write()

print(f'Power:  {instance.P.get_values()}')
print(f'Flow:   {instance.Q.get_values()}')
print(f'Height: {instance.H.get_values()}')
print(f'Open:   {instance.a.get_values()}')