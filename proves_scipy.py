import numpy as np

from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint

bounds = ((0, 1.5), (0, 1.5), (0, 1.5), (0, 1.5), (0, 1.5))
cost_h = np.array([1, 2, 4, 5, 3])

def cost(p):

    return np.dot(p, cost_h)
    #return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

p0 = np.array([1.3, 0.7, 0.8, 1.4, 1.2])

cons = [{'type': 'ineq', 'fun': lambda x:  np.sum(x) - 4}]

# linear_constraint = LinearConstraint([[1, 2], [2, 1]], [-np.inf, 1], [1, 1])
# linear_constraint = LinearConstraint(np.array([1,1,1,1,1]).T, np.array([4]), np.array([np.inf]))

res = minimize(cost, p0, method='SLSQP', bounds=bounds, constraints=cons,
               options={'disp': True})

print(res.x)



#%%
import pyomo.environ
import pyomo.core as pyo
from pyomo.opt import SolverFactory
import scipy as sc
import numpy as np
import math

model = pyo.AbstractModel()

model.x = pyo.Var(within=pyo.NonNegativeReals)


def constraint1(m):
    return m.x>=0
model.constr1 = pyo.Constraint(rule=constraint1)

def obj_fun(m):
    return (m.x-1)**2
model.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)

instance = model.create_instance()
solver = SolverFactory('ipopt')
solver.solve(instance)

instance.x.get_values()[None]

#%%
import pyomo.environ
import pyomo.core as pyo
from pyomo.opt import SolverFactory
import numpy as np

rho_g = 1e3*9.81
A = 900
B = 6
Ho = 305
K = 0.105
cte_v = 0.072

Q_total = 10

model = pyo.AbstractModel()

l_t = list(range(5))
model.t = pyo.Set(initialize=l_t)

l_cost = [0, 5, 10, 5, 0]
# model.cost_elec = pyo.Param(initialize=l_cost)

model.P = pyo.Var(model.t, within=pyo.NonNegativeReals)
model.Q = pyo.Var(model.t, within=pyo.NonNegativeReals)
model.H = pyo.Var(model.t, within=pyo.NonNegativeReals)
model.a = pyo.Var(model.t, within=pyo.NonNegativeReals)

def Constraint_P(m, t):
    return m.P[t] == rho_g * m.Q[t] * m.H[t]
model.Constraint_P = pyo.Constraint(model.t, rule=Constraint_P)

def Constraint_H(m, t):
    return m.H[t] == A - B*(m.Q[t])**2
model.Constraint_H = pyo.Constraint(model.t, rule=Constraint_H)

def Constraint_Hr_Hb(m, t):
    return Ho+(K+cte_v/m.a[t])*(m.Q[t])**2 == m.H[t]
model.Constraint_Hr_Hb = pyo.Constraint(model.t, rule=Constraint_Hr_Hb)

def Constraint_Q_max(m, t):
    return m.Q[t]<=9
model.Constraint_Q_max = pyo.Constraint(model.t, rule=Constraint_Q_max)

def Constraint_Q_total(m, t):
    return sum(m.Q[t] for t in l_t) == Q_total
model.Constraint_Q_total = pyo.Constraint(model.t, rule=Constraint_Q_total)

def Constraint_alpha(m, t):
    return m.a[t]<=1
model.Constraint_alpha = pyo.Constraint(model.t, rule=Constraint_alpha)



def obj_fun(m):
    return sum(m.P[t]*l_cost[t] for t in l_t)
model.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)    


instance = model.create_instance()
solver = SolverFactory('ipopt')
solver.options['max_iter']= 10000
results = solver.solve(instance)
results.write()

instance.P.get_values()
instance.Q.get_values()
# instance.H.get_values()
# instance.a.get_values()






