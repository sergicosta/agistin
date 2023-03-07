# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:36:19 2023
"""

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import diags

class problem():

    def __init__(self, cost, Dt):
        self.Dt = Dt
        self.rho_g = 1e3*9.81
        self.A = 900.
        self.B = 6.
        self.Ho = 305.
        self.K = 0.105
        self.cte_v = 100.
        self.W_0 = 35.
        self.W_max = 60.
        self.W_min = 12.
        self.p = np.ones_like(cost)*15e6
        self.Q = np.ones_like(cost)*2.
        self.H = np.ones_like(cost)*800.
        self.a = np.ones_like(cost)
        self.P_max = 6*3e6
        self.Q_max = None
        self.Q_tot = 6.
        self.cost = cost
        self.W = np.ones(len(cost)+1)
        DIAGS = [self.W, -self.W[1:]]
        self.M = diags(DIAGS, [0,-1]).toarray()
        self.x = np.concatenate((self.p, self.Q, self.H, self.a, self.W))


    def optimization(self):
        n = len(self.cost)
        fun = lambda x: self.Dt*np.dot(self.cost, x[0:n])
        cons = [{'type':'eq', 'fun': lambda x: x[2*n:3*n] - self.A + self.B * x[n:2*n]**2},
                {'type':'eq', 'fun': lambda x: self.A - self.B * x[n:2*n]**2 - self.Ho - (self.K + self.cte_v * x[3*n:4*n])*x[n:2*n]**2},
                {'type':'eq', 'fun': lambda x: x[0:n] - self.rho_g * x[n:2*n]*x[2*n:3*n]},
                {'type':'eq', 'fun': lambda x: sum(x[n:2*n]) - self.Q_tot},
                {'type':'eq', 'fun': lambda x: np.dot(self.M, x[4*n:]) - self.Dt*np.concatenate((np.array([self.W_0]), x[n:2*n]))},]
        bnds = tuple([(0, self.P_max)]*n + [(0, self.Q_max)]*n + [(0, None)]*n + [(1, None)]*n + [(self.W_min, self.W_max)]*(n+1))
        res = minimize(fun, self.x, method='SLSQP', constraints=cons, bounds=bnds, tol = 1e-4, options={'maxiter':300, 'disp':True})
        self.x = res.x


cost = np.array([1, 5, 8, 3])
Dt = 1
n = len(cost)
prb = problem(cost, Dt)
prb.optimization()
print(f'power:  {prb.x[:n]} \nflow:   {prb.x[n:2*n]} \nheight: {prb.x[2*n:3*n]} \nvalv:   {1/prb.x[3*n:4*n]}  \nvolm:   {prb.x[4*n:]}')

