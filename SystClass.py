# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 09:32:33 2023

Class system represents the whole system, define elements in the following order:
    1. Reservoirs
    2. Pipes (connecting reservoirs)
    3. Pumps (associated to pipes)
    
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
    
    def add_pump(self, rho_g, A, B, p_max, Q_max, in_pipe):
        self.pumps[f'{self.id_pm}'] = pump(self, self.id_pm, rho_g, A, B, p_max, Q_max, in_pipe)
        if self.pumps[f'{self.id_pm}'].verification == False:
            del(self.pumps[f'{self.id_pm}'])
        else:
            self.id_pm += 1
    
    def add_pump_simple(self, p_max, Q_max, efficiency, in_pipe):
        self.pumps[f'{self.id_pm}'] = pump_simple(self, self.id_pm, p_max, Q_max, efficiency, in_pipe)
        if self.pumps[f'{self.id_pm}'].verification == False:
            del(self.pumps[f'{self.id_pm}'])
        else:
            self.id_pm += 1
    
    def add_pipe(self, K_i, H_0, orig, end, valve=False, C_v=None, alpha=None):
        self.pipes[f'{self.id_pp}'] = pipe(self, self.id_pp, K_i, H_0, orig, end, valve, C_v, alpha)
        if self.pipes[f'{self.id_pp}'].verification == False:
            del(self.pipes[f'{self.id_pp}'])
        else:
            self.pipes[f'{self.id_pp}'].vinculo(self, orig, end)
            self.id_pp += 1
            
    def builder(self): # , obj_fun
        file_name = 'optimization.py'
        
        with open(file_name, 'w') as f:
            f.write('# -*- coding: utf-8 -*-\n\n')
            f.write('import pyomo.environ as pyo\n')
            f.write('model = pyo.AbstractModel()\n\n')
            f.write('n = 24\nl_t = list(range(n))\n')
            f.write('model.t = pyo.Set(initialize=l_t)\n')
            f.write('\n')
            
            for element_dict in [self.rsvrs, self.pumps, self.pipes]:
                for ID in element_dict.keys():
                    for v in element_dict[ID].var: # write variables
                        f.write(v)
                        f.write('\n')
                    f.write('\n')
                    
                    element_dict[ID].eq_write()
                    for eq in element_dict[ID].eqs: # write constraints
                        f.write(eq)
                        f.write('\n')
                    f.write('\n')
            
            f.close()
    
class syst_element(): # Superclass
    def __init__(self, ID):
        self.ID = ID
        self.x = []
        self.var = []
        self.eqs = list()
    
    def eq_write(self):
        pass

class reservoir(syst_element):
    
    def __init__(self, ID, W_0, W_max, W_min):
        super().__init__(ID)
        self.W_0 = W_0
        self.W_max = W_max
        self.W_min = W_min
        self.x = [f'W_r{ID}']
        self.Q_in = list()
        self.Q_out = list()
        self.eqs = list()
        self.var = ['model.'+self.x[0]+' = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=('+str(self.W_min)+', '+str(self.W_max)+'), initialize={k:'+str(self.W_0)+' for k in range(n)},)',]
        
    def eq_write(self):
        qin_txt, qout_txt = '',''
        
        if len(self.Q_in)>0:
            qin_txt = '+ ('
            for q in self.Q_in:
                qin_txt += f'm.{q}[t] + '
            qin_txt = qin_txt[0:-3] + ')'
            
        if len(self.Q_out)>0:
            qout_txt = '- ('
            for q in self.Q_out:
                qout_txt += f'm.{q}[t] + '
            qout_txt = qout_txt[0:-3] + ')'
        
        self.eqs.append(f'def Constraint_{self.x[0]}(m, t): \n'
                        f'\treturn m.{self.x[0]}[t] == m.{self.x[0]}[t-1] '
                        + qin_txt + qout_txt)


class pump(syst_element):
    
    def __init__(self, system, ID, rho_g, A, B, p_max, Q_max, rpm_nominal, in_pipe):
        super().__init__(ID)
        self.system = system
        self.rho_g = rho_g
        self.A = A
        self.B = B
        self.p_max = p_max
        self.Q_max = Q_max
        self.rpm_nominal = rpm_nominal
        self.verification = True
        self.conn = self.conns(in_pipe)
        self.x = [f'p_b{ID}', f'Q_b{ID}', f'H_b{ID}', f'rpm_{ID}']
        self.eqs = list()
        self.var = ['model.'+self.x[0]+' = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(1e-6, '+str(self.p_max)+'),  initialize={k:'+str(0.8*self.p_max)+' for k in range(n)},)',
                    'model.'+self.x[1]+' = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(1e-6, '+str(self.Q_max)+'),   initialize={k:'+str(0.8*self.Q_max)+'   for k in range(n)},)',
                    'model.'+self.x[2]+' = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(None, None),   initialize={k:'+str(self.A)+' for k in range(n)},)',]
        
    def conns(self, conn):
        if str(conn) in list(self.system.pipes.keys()):
            self.system.pipes[str(conn)].parallel_pumps.append(self.ID)
            return conn
        else:
            print('Associated pipe ID does not match any created. Pump eliminated')
            self.verification *= False
        
    def eq_write(self):
        self.eqs.append(f'def Constraint_{self.x[0]}(m, t): \n'
                        f'\treturn m.{self.x[0]}[t] == {self.rho_g} * m.{self.x[1]}[t] * m.{self.x[2]}[t]'
                        f'model.Constraint_{self.x[0]} = pyo.Constraint(model.t, rule=Constraint_{self.x[0]})')
        self.eqs.append(f'def Constraint_{self.x[2]}(m, t): \n'
                        f'\treturn m.{self.x[2]}[t] == ((m.{self.x[3]}[t]/{self.rpm_nominal})**2) * {self.A} - {self.B}*(m.{self.x[1]}[t])**2'
                        f'model.Constraint_{self.x[2]} = pyo.Constraint(model.t, rule=Constraint_{self.x[2]})')
    
    
class pump_simple(syst_element):
    
    def __init__(self, system, ID, p_max, Q_max, efficiency, in_pipe):
        super().__init__(ID)
        self.system = system
        self.p_max = p_max
        self.Q_max = Q_max
        self.efficiency = efficiency
        self.verification = True
        self.conn = self.conns(in_pipe)
        self.x = [f'p_b{ID}', f'Q_b{ID}']
        self.eqs = list()
        self.var = ['model.'+self.x[0]+' = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(1e-6, '+str(self.p_max)+'),  initialize={k:'+str(0.8*self.p_max)+' for k in range(n)},)',
                    'model.'+self.x[1]+' = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(1e-6, '+str(self.Q_max)+'),   initialize={k:'+str(0.8*self.Q_max)+'   for k in range(n)},)',
                    ]
        
    def conns(self, conn):
        if str(conn) in list(self.system.pipes.keys()):
            self.system.pipes[str(conn)].parallel_pumps.append(self.ID)
            return conn
        else:
            print('Associated pipe ID does not match any created. Pump eliminated')
            self.verification *= False
        
    def eq_write(self):
        
        self.eqs.append(f'def Constraint_{self.x[0]}(m, t): \n'
                        f'\treturn m.{self.x[0]}[t] == m.{self.x[1]}[t] / {self.efficiency}\n'
                        f'model.Constraint_{self.x[0]} = pyo.Constraint(model.t, rule=Constraint_{self.x[0]})')


class pipe(syst_element):
    
    def __init__(self, system, ID, K_i, H_0, orig, end, valve, C_v, alpha):
        super().__init__(ID)
        self.system = system
        self.K_i = K_i
        self.H_0 = H_0
        self.Q_max = 1e6
        self.verification = True
        self.parallel_pumps = list() # list of ID (int) of pumps in parallel
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
            
    def vinculo(self, sys, orig, end):
        sys.rsvrs[f'{end}'].Q_in.append(self.x[0])
        sys.rsvrs[f'{orig}'].Q_out.append(self.x[0])
        
    def eq_write(self):
        if len(self.parallel_pumps)>0:
            qb_txt = ''
            for b in self.parallel_pumps:
                qb_txt += f'+ m.Q_b{b}[t]'
            self.eqs.append(f'def Constraint_{self.x[0]}(m, t): \n'
                            f'\treturn m.{self.x[0]}[t] == '+qb_txt+'\n'
                            f'model.Constraint_{self.x[0]} = pyo.Constraint(model.t, rule=Constraint_{self.x[0]})')
    
        
if __name__ == "__main__":
    
    sgr_sud = system()
    sgr_sud.add_rsvr(58, 90, 30)
    sgr_sud.add_rsvr(34, 50, 20)
    sgr_sud.add_pipe(5, 630, 0, 1)
    # sgr_sud.add_pump(9.81e3, 800, 12, 25e4, 1450, 2, 0)
    sgr_sud.add_pump_simple(25e4, 2, 0.5, 0)
    sgr_sud.add_pump_simple(25e4, 2, 0.9, 0)
    
    sgr_sud.builder()
    
    # for i in sgr_sud.rsvrs.keys():
    #     print(sgr_sud.rsvrs[i].x)
    #     for k in sgr_sud.rsvrs[i].var:
    #         print(k)
    #     print('\n')
    
    # for i in sgr_sud.pumps.keys():
    #     print(sgr_sud.pumps[i].x)
    #     for k in sgr_sud.pumps[i].var:
    #         print(k)
    #     sgr_sud.pumps[i].eq_write()
    #     for k in sgr_sud.pumps[i].eqs:
    #         print(k)


        
        
        