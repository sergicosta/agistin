# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 09:32:33 2023

"""

class system():
    
    def __init__(self):
        self.rsvrs = dict()
        self.pumps = dict()
        self.pipes = dict()
        self.id_rs, self.id_pm, self.id_pp = 0, 0, 0
    
    def add_rsvr(self, W_0, W_max, W_min):
        self.rsvrs[f'{self.id_rs}'] = reservoir(self.id_rs, W_0, W_max, W_min)
        self.id_rs += 1
    
    def add_pump(self, rho_g, A, B, p_max, Q_max, ins, outs):
        self.pumps[f'{self.id_pm}'] = pump(self, self.id_pm, rho_g, A, B, p_max, Q_max, ins, outs)
        if self.pumps[f'{self.id_pm}'].verification == False:
            del(self.pumps[f'{self.id_pm}'])
        else:
            self.id_pm += 1
    
    def add_pipe(self, K_i, H_0, orig, end, valve=False, C_v=None, alpha=None):
        self.pipes[f'{self.id_pp}'] = pipe(self, self.id_pp, K_i, H_0, orig, end, valve, C_v, alpha)
        if self.pipes[f'{self.id_pp}'].verification == False:
            del(self.pipes[f'{self.id_pp}'])
        else:
            self.id_pp += 1


class reservoir():
    
    def __init__(self, ID, W_0, W_max, W_min):
        self.id = ID
        self.W_0 = W_0
        self.W_max = W_max
        self.W_min = W_min
        self.x = [f'W_r{ID}']
        self.Q_in = list()
        self.Q_out = list()
        self.var = ['model.'+self.x[0]+' = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=('+str(self.W_min)+', '+str(self.W_max)+'), initialize={k:'+str(self.W_0)+' for k in range(n)},)',]
        

class pump():
    
    def __init__(self, system, ID, rho_g, A, B, p_max, Q_max, ins, outs):
        self.system = system
        self.id = ID
        self.rho_g = rho_g
        self.A = A
        self.B = B
        self.p_max = p_max
        self.Q_max = Q_max
        self.verification = True
        self.input = self.inpt(ins)
        self.output = self.outpt(outs)
        self.x = [f'p_b{ID}', f'Q_b{ID}', f'H_b{ID}']
        self.eqs = list()
        self.var = ['model.'+self.x[0]+' = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(1e-6, '+str(self.p_max)+'),  initialize={k:'+str(0.8*self.p_max)+' for k in range(n)},)',
                    'model.'+self.x[1]+' = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(1e-6, '+str(self.Q_max)+'),   initialize={k:'+str(0.8*self.Q_max)+'   for k in range(n)},)',
                    'model.'+self.x[2]+' = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(None, None),   initialize={k:'+str(self.A)+' for k in range(n)},)',]
        
    def inpt(self, ins):
        if str(ins) in list(self.system.rsvrs.keys()):
            return ins
        else:
            print('Input reservoir ID does not match any created. Pump eliminated')
            self.verification *= False
            
    def outpt(self, out):
        if str(out) in list(self.system.rsvrs.keys()):
            if out != self.input:
                return out
            else:
                print('Input and Output reservoir IDs cannot be the same. Pump eliminated')
                self.verification *= False
        else:
            print('Output reservoir ID does not match any created. Pump eliminated')
            self.verification *= False
        

class pipe():
    
    def __init__(self, system, ID, K_i, H_0, orig, end, valve, C_v, alpha):
        self.system = system
        self.id = ID
        self.K_i = K_i
        self.H_0 = H_0
        self.Q_max = 1e6
        self.verification = True
        self.orig = self.inpt(orig)
        self.end = self.outpt(end)
        self.valve = valve
        self.C_v = C_v
        self.alpha = alpha
        self.x = [f'Q_p{ID}', f'H_p{ID}']
        self.var = ['model.'+self.x[0]+' = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(1e-6, '+str(self.Q_max)+'),   initialize={k:'+str(0.8*self.Q_max)+'   for k in range(n)},)',
                    'model.'+self.x[1]+' = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(None, None),   initialize={k:'+str(self.H_0)+' for k in range(n)},)',]
        
    def inpt(self, ins):
        if str(ins) in list(self.system.rsvrs.keys()):
            return ins
        else:
            print('Input reservoir ID does not match any created. Pipe eliminated')
            self.verification *= False
            
    def outpt(self, out):
        if str(out) in list(self.system.rsvrs.keys()):
            if out != self.orig:
                return out
            else:
                print('Input and Output reservoir IDs cannot be the same. Pipe eliminated')
                self.verification *= False
        else:
            print('Output reservoir ID does not match any created. Pipe eliminated')
            self.verification *= False
        
        
        