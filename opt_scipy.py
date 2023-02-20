# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:36:19 2023
"""

import numpy as np
from scipy.optimize import minimize

class problem():

    def __init__(self, cost):
        self.rho_g = 1e3*9.81
        self.A = 900
        self.B = 6
        self.Ho = 305
        self.K = 0.105
        self.cte_v = 100.
        self.p = np.ones_like(cost)*15e6
        self.Q = np.ones_like(cost)*2
        self.H = np.ones_like(cost)*800
        self.a = np.ones_like(cost)*0.5
        self.b = np.ones_like(cost)
        self.x = np.concatenate((self.p, self.Q, self.H, self.b))
        self.P_max = 6*3e6
        self.Q_max = None
        self.Q_tot = 4
        self.cost = cost


    def optimization(self):
        n = len(self.cost)
        fun = lambda x: np.dot(self.cost, x[0:n])
        cons = [{'type':'eq', 'fun': lambda x: x[2*n:3*n] - self.A + self.B * x[n:2*n]**2},
                {'type':'eq', 'fun': lambda x: self.A - self.B * x[n:2*n]**2 - self.Ho - (self.K + self.cte_v * x[3*n:4*n])*x[n:2*n]**2},
                {'type':'eq', 'fun': lambda x: x[0:n] - self.rho_g * x[n:2*n]*x[2*n:3*n]},
                {'type':'eq', 'fun': lambda x: sum(x[n:2*n]) - self.Q_tot},]
        #bnds = tuple([(0, None)]*3*n + [(1e-6, 1)]*n)
        bnds = tuple([(0, self.P_max)]*n + [(0, self.Q_max)]*n + [(0, None)]*n + [(1, None)]*n)
        res = minimize(fun, self.x, method='SLSQP', constraints=cons, bounds=bnds, tol = 1e-4)
        print(res.message)
        self.x = res.x


cost = np.array([1, 5,])
n = len(cost)
prb = problem(cost)
prb.optimization()
print(f'power:  {prb.x[:n]} \nflow:   {prb.x[n:2*n]} \nheight: {prb.x[2*n:3*n]} \nvalv:   {1/prb.x[3*n:]} \n')
