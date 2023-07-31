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


# CONSTANTS
rho_g = 1e3*9.81


# DEFINITIONS
class Pump:
    
    def __init__(self, A, B, Pmax=1e6):
        self.A = A
        self.B = B
        self.Pmax = Pmax
        
    def H(self, Q):
        return self.A + self.B*(Q**2)

class Pipe:
    
    def __init__(self, Ho, K, Kv, pumps=[]):
        self.Ho = Ho
        self.K = K
        self.Kv = Kv
        self.pumps = pumps
        
    def add_pump(A, B, Pmax):
        self.pumps = self.pumps.append((A,B,Pmax))
        
    def H(self, Q, var_a):
        return self.Ho + (self.K + self.Kv/a)*(Q**2)


class Reservoir:
    
    def __init__(self, W_o, W_total, W_min, W_max, syst_in, syst_out):
        self.W_total = W_total
        self.W_min = W_min
        self.W_max = W_max
        self.W = W_o
        self.syst_in = syst_in
        self.syst_out = syst_out
        
        
    def W(self, dt=3600):
        Q_in = 
        Q_out = 
        self.W = self.W + (Q_in - Q_out)*dt
        return self.W


model = pyo.AbstractModel()


# SETS
l_t = list(range(24))
model.t = pyo.Set(initialize=l_t)

list_pumps = [Pump(900,6), Pump(900,6)]
list_syst = [System(305,0.105,100,[list_pumps[0]]),
             System(305,0.105,100,[list_pumps[1]])]

list_reservoirs = [Reservoir(1e10,0.01,0.99, [], list_syst[0]), 
                   Reservoir(100,0.3,0.7, list_syst[0], list_syst[1]), 
                   Reservoir(100,0.3,0.7, list_syst[1], [])]
model.r = pyo.Set(initialize=list_reservoirs)


# PARAMS
list_cost = [-(x-12)**2+144 for x in range(24)]
model.cost = pyo.Param(initialize=list_cost)


A = 900
B = 6
Ho = 305
K = 0.105
cte_v = 100

Q_total = 10



l_cost = [1, 5, 10, 5, 1]
model.cost_elec = pyo.Param(model.t, initialize=l_cost)


vals = [6e6]*len(l_t)
model.P = pyo.Var(model.t, within=pyo.NonNegativeReals, initialize={ k:v for (k,v) in zip(l_t, vals)})

vals = [1]*len(l_t)
model.Q = pyo.Var(model.t, within=pyo.NonNegativeReals, initialize={ k:v for (k,v) in zip(l_t, vals)})

vals = [400]*len(l_t)
model.H = pyo.Var(model.t, within=pyo.NonNegativeReals, initialize={ k:v for (k,v) in zip(l_t, vals)})

vals = [0.5]*len(l_t)
model.a = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(0,1), initialize={ k:v for (k,v) in zip(l_t, vals)})



def Constraint_P(m, t):
    return m.P[t] == rho_g * m.Q[t] * m.H[t]
model.Constraint_P = pyo.Constraint(model.t, rule=Constraint_P)

def Constraint_H(m, t):
    return m.H[t] == A - B*(m.Q[t])**2
model.Constraint_H = pyo.Constraint(model.t, rule=Constraint_H)

def Constraint_Hr_Hb(m, t):
    return Ho+(K+cte_v/m.a[t])*(m.Q[t])**2 == A - B*(m.Q[t])**2
model.Constraint_Hr_Hb = pyo.Constraint(model.t, rule=Constraint_Hr_Hb)

def Constraint_Q_max(m, t):
    return m.Q[t]<=10
model.Constraint_Q_max = pyo.Constraint(model.t, rule=Constraint_Q_max)

def Constraint_Q_total(m, t):
    return sum(m.Q[t] for t in l_t) == Q_total
model.Constraint_Q_total = pyo.Constraint(model.t, rule=Constraint_Q_total)



def obj_fun(m):
    return sum(m.P[t]*m.cost_elec[t] for t in l_t)
model.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)    


instance = model.create_instance()
solver = SolverFactory('mindtpy') #ipopt
# solver.options['max_iter']= 10000
results = solver.solve(instance, tee=True)
instance.pprint()

print('P: ', instance.P.get_values())
print('Q: ', instance.Q.get_values())
print('H: ', instance.H.get_values())
print('a: ', instance.a.get_values())


#%%

import pyomo.environ
import pyomo.core as pyo
from pyomo.opt import SolverFactory
import numpy as np

# PRE PROCESSING
n_basses = 4
n_canonades = 3

l_time = list(range(24))
l_cost = (np.arange(24, )list(range(24))-12)**2

l_W_top = [10, 10, 10, 10] #n_basses
l_W_bot = [3, 3, 3, 3] #n_basses

l_Ho = [266, 93, 23] #n_canonades
l_K = [0.20, 0.20, 0.20] #n_canonades

l_Pmax = [3*3.2e6, 2*250e3, 2*1.25e6] #n_canonades


model = pyo.AbstractModel()

### Model Sets ###
model.t = pyo.Set(initialize=l_time)



