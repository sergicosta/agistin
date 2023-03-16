# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 16:29:21 2023

@author: colives
"""
import pyomo.environ as pyo

Dt = 1.
E_0_1, E_0_2 = 350., 12.
E_max_1, E_max_2 = 1e3, 20.
E_min_1, E_min_2 = 1e2, 10.
P_max = 12
rend = 0.8
epsn = 0.03

model = pyo.AbstractModel()

P_out  = (4, 13, 7, 9, 15, 8, )
P_pv   = (1, 1, 4, 8, 5, 2, )
l_cost = (1, 5, 8, 3, 9, 0, )
n = len(l_cost)
l_t = tuple(range(n))

model.t = pyo.Set(initialize=l_t)
model.cost_elec = pyo.Param(model.t, initialize=l_cost)

model.E_0_1 = pyo.Param(initialize=E_0_1)
model.E_0_2 = pyo.Param(initialize=E_0_2)
model.Dt    = pyo.Param(initialize=Dt)
model.rend  = pyo.Param(initialize=rend)
model.epsn  = pyo.Param(initialize=epsn)

model.P_g  = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(1e-6, P_max),  initialize={k:1.0 for k in range(n)},)
model.P_e  = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(1e-6, P_max),  initialize={k:4.0 for k in range(n)},)
model.P_h  = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(1e-6, P_max),  initialize={k:4.0 for k in range(n)},)
model.Ppvu = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(1e-6, 15), initialize={k:1e-3 for k in range(n)})
model.DE   = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(-1e6, 1e6),  initialize={k:0.0 for k in range(n)},)
model.E_1  = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(E_min_1, E_max_1), initialize={k:E_0_1 for k in range(n)},)
model.E_2  = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(E_min_2, E_max_2), initialize={k:E_0_2 for k in range(n)},)

def Constraint_P_pv(m, t):
    return m.Ppvu[t] <= P_pv[t]
model.Constraint_P_pv = pyo.Constraint(model.t, rule=Constraint_P_pv)

def Constraint_P_grid(m, t):
    return m.P_e[t] == m.P_g[t] + m.Ppvu[t]
model.Constraint_P_grid = pyo.Constraint(model.t, rule=Constraint_P_grid)

def Constraint_P_pump(m, t):
    return m.P_h[t] == m.rend * m.P_e[t]
model.Constraint_P_pump = pyo.Constraint(model.t, rule=Constraint_P_pump)

def Constraint_varE(m, t):
    return m.DE[t] == m.Dt * m.P_h[t] * (1. - m.epsn)
model.COnstrint_varE = pyo.Constraint(model.t, rule=Constraint_varE)

def Constraint_E_1_stored(m, t):
    try:
        return m.E_1[t] == m.E_1[t - 1] - m.DE[t]
    except KeyError:
        return m.E_1[t] == m.E_0_1 - m.DE[t]
model.Constr_E_1 = pyo.Constraint(model.t, rule=Constraint_E_1_stored)

def Constraint_E_2_stored(m, t):
    try:
        return m.E_2[t] == m.E_2[t - 1] + m.DE[t] - m.Dt * P_out[t]
    except KeyError:
        return m.E_2[t] == m.E_0_2 + m.DE[t] - m.Dt * P_out[t]
model.Constr_E_2 = pyo.Constraint(model.t, rule=Constraint_E_2_stored)


def obj_fun(m):
    return sum(m.P_g[t]*m.cost_elec[t] for t in l_t)
model.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)


instance = model.create_instance()
solver = pyo.SolverFactory('mindtpy')
solver.solve(instance, mip_solver='glpk', nlp_solver='ipopt', tee=True)

print(f'P grid:   {instance.P_g.get_values()}')
print(f'P pv:     {instance.Ppvu.get_values()}')
print(f'P elec:   {instance.P_e.get_values()}')
print(f'P hydro:  {instance.P_h.get_values()}')
print(f'var Enrg: {instance.DE.get_values()}')
print(f'Energy 1: {instance.E_1.get_values()}')
print(f'Energy 2: {instance.E_2.get_values()}')