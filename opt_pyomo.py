# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 16:29:21 2023
"""
import pyomo.environ as pyo

rho_g = 1e3*9.81
Dt = 1.
A = 900
B = 6
Ho = 305
K = 0.105
cte_v = 100
Q_tot = 10.
W_0 = 35.
W_max = 60.
W_min = 30.
P_max = 6*3e6

model = pyo.AbstractModel()

q_out =  (1, 4, 2, 2, 3, 2, )
l_cost = (1, 5, 8, 3, 9, 0, )
n = len(l_cost)
l_t = tuple(range(n))

model.t = pyo.Set(initialize=l_t)
model.cost_elec = pyo.Param(model.t, initialize=l_cost)

model.rho_g = pyo.Param(initialize=rho_g)
model.A     = pyo.Param(initialize=A)
model.B     = pyo.Param(initialize=B)
model.Ho    = pyo.Param(initialize=Ho)
model.K     = pyo.Param(initialize=K)
model.cte_v = pyo.Param(initialize=cte_v)
model.Q_tot = pyo.Param(initialize=Q_tot)
model.W_0   = pyo.Param(initialize=W_0)
model.Dt    = pyo.Param(initialize=Dt)

model.P = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(1e-6, P_max),  initialize={k:4e6 for k in range(n)},)
model.Q = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(None, None),   initialize={k:2   for k in range(n)},)
model.H = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(None, None),   initialize={k:800 for k in range(n)},)
model.a = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(0, 1),         initialize={k:0.5 for k in range(n)},)
model.b = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(1, None),      initialize={k:2.0 for k in range(n)},)
model.W = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(W_min, W_max), initialize={k:W_0 for k in range(n)},)

def Constraint_P(m, t):
    return m.P[t] == m.rho_g * m.Q[t] * m.H[t]
model.Constraint_P = pyo.Constraint(model.t, rule=Constraint_P)

def Constraint_H(m, t):
    return m.H[t] == m.A - m.B*(m.Q[t])**2
model.Constraint_H = pyo.Constraint(model.t, rule=Constraint_H)

def Constraint_Hr_Hb(m, t):
    return m.Ho + (m.K + m.cte_v*m.b[t])*m.Q[t]**2 == m.A - m.B*m.Q[t]**2
model.Constraint_Hr_Hb = pyo.Constraint(model.t, rule=Constraint_Hr_Hb)

# def Constraint_Q_total(m, t):
#     return sum(m.Q[t] for t in l_t) == m.Q_tot
# model.Constraint_Q_total = pyo.Constraint(model.t, rule=Constraint_Q_total)

def Constraint_dummy(m, t):
    return m.a[t]*m.b[t] == 1.
model.Constraint_dummy = pyo.Constraint(model.t, rule=Constraint_dummy)

def Constraint_W_stored(m, t):
    try:
        return m.W[t] == m.W[t - 1] + m.Dt * (m.Q[t] - q_out[t])
    except KeyError:
        return m.W[t] == m.W_0 + m.Dt * (m.Q[t] - q_out[t])
model.Constr_WSt = pyo.Constraint(model.t, rule=Constraint_W_stored)


def obj_fun(m):
    return sum(m.P[t]*m.cost_elec[t] for t in l_t)
model.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)


instance = model.create_instance()
solver = pyo.SolverFactory('mindtpy')
solver.solve(instance, mip_solver='glpk', nlp_solver='ipopt', tee=True)

print(f'Power:  {instance.P.get_values()}')
print(f'Flow:   {instance.Q.get_values()}')
print(f'Height: {instance.H.get_values()}')
print(f'Open:   {instance.a.get_values()}')
print(f'Volume: {instance.W.get_values()}')