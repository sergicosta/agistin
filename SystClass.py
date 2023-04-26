# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 09:32:33 2023

Class system represents the whole system, define elements in the following order:
    1. Reservoirs
    2. Pump Stations [EB] (electrical power balance)
    3. Pipes (connecting reservoirs)
    4. Pumps (associated to pipes)
    5. Turbines (associated to pipes)
    6. PV, batteries ... linked to EBs
    
"""

class system():
    
    def __init__(self, rho_g):
        self.rsvrs = dict()
        self.EBs   = dict()
        self.pumps = dict()
        self.pipes = dict()
        self.pvs   = dict()
        self.batts = dict()
        self.trbs  = dict()
        self.id_rs, self.id_eb, self.id_pm, self.id_pp, self.id_pv, self.id_bat, self.id_trb = 0, 0, 0, 0, 0, 0, 0
        self.obj = list()
        self.x_s = list()
        self.rho_g = rho_g
    
    def add_rsvr(self, W_0, W_max, W_min):
        self.rsvrs[f'{self.id_rs}'] = reservoir(self.id_rs, W_0, W_max, W_min)
        self.x_s.extend(self.rsvrs[f'{self.id_rs}'].x)
        self.id_rs += 1
        
    def add_EB(self, p_trafo):
        self.EBs[f'{self.id_eb}'] = EB(self.id_eb, p_trafo)
        self.x_s.extend(self.EBs[f'{self.id_eb}'].x)
        self.id_eb += 1
    
    def add_pipe(self, K_i, H_0, orig, end, valve=False, C_v=None, alpha=None):
        self.pipes[f'{self.id_pp}'] = pipe(self, self.id_pp, K_i, H_0, orig, end, valve, C_v, alpha)
        if self.pipes[f'{self.id_pp}'].verification == False:
            del(self.pipes[f'{self.id_pp}'])
        else:
            self.x_s.extend(self.pipes[f'{self.id_pp}'].x)
            self.pipes[f'{self.id_pp}'].link_to()
            self.id_pp += 1
     
    def add_pump(self, A, B, p_max, Q_max, rpm_nominal, in_pipe, eb_loc):
        self.pumps[f'{self.id_pm}'] = pump(self, self.id_pm, self.rho_g, A, B, p_max, Q_max, rpm_nominal, in_pipe, eb_loc)
        if self.pumps[f'{self.id_pm}'].verification == False:
            del(self.pumps[f'{self.id_pm}'])
        else:
            self.x_s.extend(self.pumps[f'{self.id_pm}'].x)
            self.pumps[f'{self.id_pm}'].link_to()
            self.id_pm += 1
    
    def add_pump_simple(self, p_max, Q_max, efficiency, in_pipe, eb_loc):
        self.pumps[f'{self.id_pm}'] = pump_simple(self, self.id_pm, p_max, Q_max, efficiency, in_pipe, eb_loc)
        if self.pumps[f'{self.id_pm}'].verification == False:
            del(self.pumps[f'{self.id_pm}'])
        else:
            self.x_s.extend(self.pumps[f'{self.id_pm}'].x)
            self.pumps[f'{self.id_pm}'].link_to()
            self.id_pm += 1
            
    def add_new_pump(self, in_pipe, eb_loc):
        self.pumps[f'{self.id_pm}'] = new_pump(self, self.id_pm, self.rho_g, in_pipe, eb_loc)
        if self.pumps[f'{self.id_pm}'].verification == False:
            del(self.pumps[f'{self.id_pm}'])
        else:
            self.x_s.extend(self.pumps[f'{self.id_pm}'].x)
            self.pumps[f'{self.id_pm}'].link_to()
            self.id_pm += 1
       
    def add_PV(self, EB_loc, p_inst, p_max):
        self.pvs[f'{self.id_pv}'] = PV(self, self.id_pv, EB_loc, p_inst, p_max)
        if self.pvs[f'{self.id_pv}'].verification == False:
            del(self.pvs[f'{self.id_pv}'])
        else:
            self.x_s.extend(self.pvs[f'{self.id_pv}'].x)
            self.pvs[f'{self.id_pv}'].link_to()
            self.id_pv += 1
        
    def add_battery(self, EB_loc, p_bat):
        self.batts[f'{self.id_bat}'] = battery(self, self.id_bat, EB_loc, p_bat)
        if self.batts[f'{self.id_bat}'].verification == False:
            del(self.batts[f'{self.id_bat}'])
        else:
            self.x_s.extend(self.batts[f'{self.id_bat}'].x)
            self.batts[f'{self.id_bat}'].link_to()
            self.id_bat += 1
    
    def add_turbine(self, p_max):
        pass
    
    def add_turbine_simple(self, p_max, Q_max, efficiency, in_pipe, eb_loc):
        self.trbs[f'{self.id_trb}'] = turbine_simple(self, self.id_trb, p_max, Q_max, efficiency, in_pipe, eb_loc)
        if self.trbs[f'{self.id_trb}'].verification == False:
            del(self.trbs[f'{self.id_trb}'])
        else:
            self.x_s.extend(self.trbs[f'{self.id_trb}'].x)
            self.trbs[f'{self.id_trb}'].link_to()
            self.id_trb += 1
    
    def write_obj(self):
        p_grid = '('
        for eb in self.EBs.keys():
            p_grid += f'm.{self.EBs[eb].x[0]}[t] + '
        p_grid = p_grid[:-3] + ')'
        self.obj.append('def obj_fun(m):\n'
                        f'\treturn sum({p_grid}*m.cost_elec[t] for t in l_t)\n'
                        'model.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)\n')
        
    def builder(self, solver): # , obj_fun
        """ This function collects all the information from the system and 
        writes a script (optimization.py) that codes the optimization problem 
        using pyomo formulation."""
        
        file_name = 'optimization.py'
        
        with open(file_name, 'w') as f:
            f.write('# -*- coding: utf-8 -*-\n\n')
            f.write('import pyomo.environ as pyo\n')
            f.write('import json\n')
            f.write('# from pyomo.opt import SolverFactory\n')
            f.write('model = pyo.AbstractModel()\n\n')
            f.write('with open("ext_data.json","r") as file:\n\tdata=json.load(file)\n\n')
            f.write('n = len(data["cost_elec"])\nl_t = list(range(n))\n')
            f.write('model.t = pyo.Set(initialize=l_t)\n')
            f.write('model.Dt = pyo.Param(initialize=1.0)\n')
            f.write('model.cost_elec = pyo.Param(model.t, initialize=data["cost_elec"])\n')
            for k in range(self.id_rs):
                f.write(f'model.gamma_{k} = pyo.Param(model.t, initialize=data["gamma_{k}"])\n')
                f.write(f'model.q_irr_{k} = pyo.Param(model.t, initialize=data["q_irr_{k}"])\n')
            for k in range(self.id_pv):
                f.write(f'model.forecast_pv{k} = pyo.Param(model.t, initialize=data["weather_{k}"])\n')
            f.write('\n')
            
            vars_txt = ''
            eqs_txt = ''
            
            for element_dict in [self.rsvrs, self.EBs, self.pumps, self.pipes, self.pvs, self.batts, self.trbs]:
                for ID in element_dict.keys():
                    for v in element_dict[ID].var: # write variables
                        vars_txt = vars_txt + v + '\n'
                    
                    element_dict[ID].eq_write()
                    for eq in element_dict[ID].eqs: # write constraints
                        eqs_txt = eqs_txt + eq + '\n\n'
            
            self.write_obj()
            obj_txt = self.obj[0]

            f.write(vars_txt + '\n')
            f.write(eqs_txt + '\n')
            f.write(obj_txt + '\n\n')
            f.write('instance = model.create_instance()\n')
            f.write(f"solver = pyo.SolverFactory('{solver}')\n")
            f.write("solver.solve(instance, mip_solver='glpk', nlp_solver='ipopt', tee=True)\n")
            f.write("results = {}\n")
            for x in self.x_s:
                f.write(f"results['{x}'] = list(instance.{x}.get_values().values())\n")
            f.write("with open('results.json', 'w') as jfile:\n\tjson.dump(results, jfile)\n")
            f.close()
            
    def run(self):
        import runpy
        import json
        
        runpy.run_path(path_name='optimization.py')
        with open('results.json', 'r') as file:
            res = json.load(file)
        return res


class syst_element(): # Superclass
    def __init__(self, ID, x):
        self.ID = ID
        self.x = x
        self.var = list()
        self.eqs = list()
    
    def eq_write(self):
        pass


class reservoir(syst_element):
    
    def __init__(self, ID, W_0, W_max, W_min):
        super().__init__(ID, [f'W_r{ID}'])
        self.W_0 = W_0
        self.W_max = W_max
        self.W_min = W_min
        self.Q_in = list()
        self.Q_out = list()
        self.var = ['model.'+self.x[0]+' = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=('+str(self.W_min)+', '+str(self.W_max)+'), initialize={k:'+str(self.W_0)+' for k in range(n)},)',]
        
    def eq_write(self):
        qin_txt, qout_txt = '',''
        
        if len(self.Q_in)>0:
            qin_txt = '+ m.Dt*('
            for q in self.Q_in:
                qin_txt += f'm.{q}[t] + '
            qin_txt = qin_txt[0:-3] + ')'
            
        if len(self.Q_out)>=0:
            qout_txt = '- m.Dt*('
            for q in self.Q_out:
                qout_txt += f'm.{q}[t] + '
            qout_txt += f'm.q_irr_{self.ID}[t])'
            # qout_txt = qout_txt[0:-3] + ')'
        
        self.eqs.append(f'def Constraint_{self.x[0]}(m, t): \n'
                        f'\tif t>0:\n'
                        f'\t\treturn m.{self.x[0]}[t] == m.{self.x[0]}[t-1]*(1 -m.gamma_{self.ID}[t]) ' + qin_txt + qout_txt + '\n'
                        f'\telse:\n'
                        f'\t\treturn m.{self.x[0]}[t] == {self.W_0} ' + qin_txt + qout_txt + '\n'
                        f'model.Constraint_{self.x[0]} = pyo.Constraint(model.t, rule=Constraint_{self.x[0]})')


class EB(syst_element):
    
    def __init__(self, ID, p_max):
        super().__init__(ID, [f'P_g{ID}'])
        self.P_in  = list() # PVs and turbines generated power
        self.P_out = list() # pumps demanded power
        self.P_bal = list() # batteries power
        self.p_g_max = p_max
        self.var = ['model.'+self.x[0]+' = pyo.Var(model.t, within=pyo.Reals, bounds=('+str(-self.p_g_max)+', '+str(self.p_g_max)+'),  initialize={k:0.0 for k in range(n)},)',]
        
    def eq_write(self):
        eb_txt = ''
        p_in, p_out, p_bal = '','',''
        if len(self.P_in)>0:
            for p in self.P_in:
                p_in += f'm.{p}[t] + '
            p_in  = p_in[:-3]
            eb_txt = eb_txt + '- ('+p_in+')'
        if len(self.P_out)>0:
            for p in self.P_out:
                p_out += f'm.{p}[t] + '
            p_out = p_out[:-3]
            eb_txt = eb_txt + '+ ('+p_out+')'
        if len(self.P_bal)>0:
            for p in self.P_bal:
                p_bal += f'm.{p}[t] + '
            p_bal = p_bal[:-3]
            eb_txt = eb_txt + '+ ('+p_bal+')'
        self.eqs.append(f'def Constraint_{self.x[0]}(m, t): \n'
                        f'\treturn m.{self.x[0]}[t] == '+eb_txt+'\n'
                        f'model.Constraint_{self.x[0]} = pyo.Constraint(model.t, rule=Constraint_{self.x[0]})')


class pump(syst_element):
    
    def __init__(self, system, ID, rho_g, A, B, p_max, Q_max, rpm_nominal, in_pipe, EB_loc):
        super().__init__(ID, [f'Ph_b{ID}', f'Q_b{ID}', f'H_b{ID}', f'rpm_{ID}', f'Pe_b{ID}'])
        self.system = system
        self.rho_g = rho_g
        self.A = A
        self.B = B
        self.p_max = p_max
        self.Q_max = Q_max
        self.rpm_nominal = rpm_nominal
        self.verification = True
        self.conn = self.conns(in_pipe)
        self.loc = EB_loc
        self.var = ['model.'+self.x[0]+' = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(0.0, '+str(self.p_max)+'),  initialize={k:'+str(0.8*self.p_max)+' for k in range(n)},)',
                    'model.'+self.x[1]+' = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(0.0, '+str(self.Q_max)+'),   initialize={k:'+str(0.8*self.Q_max)+'   for k in range(n)},)',
                    'model.'+self.x[2]+' = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(None, None),   initialize={k:'+str(self.A)+' for k in range(n)},)',
                    'model.'+self.x[3]+' = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(0.0, None),   initialize={k:'+str(self.rpm_nominal)+' for k in range(n)},)',
                    'model.'+self.x[4]+' = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(0.0, '+str(self.p_max)+'),   initialize={k:'+str(0.8*self.p_max)+' for k in range(n)},)'] # TODO: Pel max
        
    def conns(self, conn):
        if str(conn) in list(self.system.pipes.keys()):
            self.system.pipes[str(conn)].parallel_pumps.append(self.ID)
            return conn
        else:
            print('Associated pipe ID does not match any created. Pump eliminated')
            self.verification *= False
    
    def link_to(self):
        self.system.EBs[f'{self.loc}'].P_out.append(self.x[4])
    
    def eq_write(self):
        #  H_b = (rpm/n_n)^2*A - B*Q^2
        self.eqs.append(f'def Constraint_{self.x[2]}(m, t): \n'
                        f'\treturn m.{self.x[2]}[t] == ((m.{self.x[3]}[t]/{self.rpm_nominal})**2) * {self.A} - {self.B}*(m.{self.x[1]}[t])**2\n'
                        f'model.Constraint_{self.x[2]} = pyo.Constraint(model.t, rule=Constraint_{self.x[2]})')
        #  Ph_b = rho*g*Q*H
        self.eqs.append(f'def Constraint_{self.x[0]}(m, t): \n'
                        f'\treturn m.{self.x[0]}[t] == {self.rho_g} * m.{self.x[1]}[t] * m.{self.x[2]}[t]\n'
                        f'model.Constraint_{self.x[0]} = pyo.Constraint(model.t, rule=Constraint_{self.x[0]})')
        #  TODO: Pe_b = Ph_b/rend
        self.eqs.append(f'def Constraint_{self.x[4]}(m, t): \n'
                        f'\treturn m.{self.x[4]}[t] == m.{self.x[0]}[t]\n'
                        f'model.Constraint_{self.x[4]} = pyo.Constraint(model.t, rule=Constraint_{self.x[4]})')
        
    
class pump_simple(syst_element):
    
    def __init__(self, system, ID, p_max, Q_max, efficiency, in_pipe, EB_loc):
        super().__init__(ID, [f'P_b{ID}', f'Q_b{ID}'])
        self.system = system
        self.p_max = p_max
        self.Q_max = Q_max
        self.efficiency = efficiency
        self.verification = True
        self.conn = self.conns(in_pipe)
        self.loc = EB_loc
        self.var = ['model.'+self.x[0]+' = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(0.0, '+str(self.p_max)+'), initialize={k:'+str(0.8*self.p_max)+' for k in range(n)},)',
                    'model.'+self.x[1]+' = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(0.0, '+str(self.Q_max)+'), initialize={k:'+str(0.8*self.Q_max)+' for k in range(n)},)',
                    ]
        
    def conns(self, conn):
        if str(conn) in list(self.system.pipes.keys()):
            self.system.pipes[str(conn)].parallel_pumps.append(self.ID)
            return conn
        else:
            print('Associated pipe ID does not match any created. Pump eliminated')
            self.verification *= False
            
    def link_to(self):
        self.system.EBs[f'{self.loc}'].P_out.append(self.x[0])
        
    def eq_write(self):
        self.eqs.append(f'def Constraint_{self.x[0]}(m, t): \n'
                        f'\treturn m.{self.x[0]}[t] == m.{self.x[1]}[t] / {self.efficiency}\n'
                        f'model.Constraint_{self.x[0]} = pyo.Constraint(model.t, rule=Constraint_{self.x[0]})')


class new_pump(syst_element):
    
    def __init__(self, system, ID, rho_g, in_pipe, EB_loc):
        super().__init__(ID, [f'Ph_b{ID}', f'Q_b{ID}', f'H_b{ID}', f'rpm_b{ID}', f'Pe_b{ID}', f'A_b{ID}', f'B_b{ID}', f'p_max_b{ID}', f'Q_max_b{ID}', f'rpm_nominal_b{ID}'])
        self.system = system
        self.rho_g = rho_g
        self.verification = True
        self.conn = self.conns(in_pipe)
        self.loc = EB_loc
        self.var = ['model.'+self.x[0]+' = pyo.Var(model.t, within=pyo.NonNegativeReals, initialize={k:0.0 for k in range(n)},)',
                    'model.'+self.x[1]+' = pyo.Var(model.t, within=pyo.NonNegativeReals, initialize={k:0.0 for k in range(n)},)',
                    'model.'+self.x[2]+' = pyo.Var(model.t, within=pyo.NonNegativeReals, initialize={k:0.0 for k in range(n)},)',
                    'model.'+self.x[3]+' = pyo.Var(model.t, within=pyo.NonNegativeReals, initialize={k:0.0 for k in range(n)},)',
                    'model.'+self.x[4]+' = pyo.Var(model.t, within=pyo.NonNegativeReals, initialize={k:0.0 for k in range(n)},)',
                    'model.'+self.x[5]+' = pyo.Var(within=pyo.NonNegativeReals, initialize=0.0,)',
                    'model.'+self.x[6]+' = pyo.Var(within=pyo.NonNegativeReals, initialize=0.0,)',
                    'model.'+self.x[7]+' = pyo.Var(within=pyo.NonNegativeReals, initialize=0.0,)',
                    'model.'+self.x[8]+' = pyo.Var(within=pyo.NonNegativeReals, initialize=0.0,)',
                    'model.'+self.x[9]+' = pyo.Var(within=pyo.NonNegativeReals, initialize=0.0,)',
                    ]
    
    def conns(self, conn):
        if str(conn) in list(self.system.pipes.keys()):
            self.system.pipes[str(conn)].parallel_pumps.append(self.ID)
            return conn
        else:
            print('Associated pipe ID does not match any created. Pump eliminated')
            self.verification *= False
    
    def link_to(self):
        self.system.EBs[f'{self.loc}'].P_out.append(self.x[4])
        
    def eq_write(self):
        #  H_b = (rpm/n_n)^2*A - B*Q^2
        self.eqs.append(f'def Constraint_{self.x[2]}(m, t): \n'
                        f'\treturn m.{self.x[2]}[t] == ((m.{self.x[3]}[t]/m.{self.x[9]})**2) * m.{self.x[5]} - m.{self.x[6]}*(m.{self.x[1]}[t])**2\n'
                        f'model.Constraint_{self.x[2]} = pyo.Constraint(model.t, rule=Constraint_{self.x[2]})')
        #  Ph_b = rho*g*Q*H
        self.eqs.append(f'def Constraint_{self.x[0]}(m, t): \n'
                        f'\treturn m.{self.x[0]}[t] == {self.rho_g} * m.{self.x[1]}[t] * m.{self.x[2]}[t]\n'
                        f'model.Constraint_{self.x[0]} = pyo.Constraint(model.t, rule=Constraint_{self.x[0]})')
        #  TODO: Pe_b = Ph_b/rend
        self.eqs.append(f'def Constraint_{self.x[4]}(m, t): \n'
                        f'\treturn m.{self.x[4]}[t] == m.{self.x[0]}[t]\n'
                        f'model.Constraint_{self.x[4]} = pyo.Constraint(model.t, rule=Constraint_{self.x[4]})')
        #  Ph[t] <= P_max
        self.eqs.append(f'def Constraint_{self.x[0]}{self.x[7]}(m, t): \n'
                        f'\treturn m.{self.x[0]}[t] <= m.{self.x[7]}\n'
                        f'model.Constraint_{self.x[0]}{self.x[7]} = pyo.Constraint(model.t, rule=Constraint_{self.x[0]}{self.x[7]})')
        #  Q[t] <= Q_max
        self.eqs.append(f'def Constraint_{self.x[1]}{self.x[8]}(m, t): \n'
                        f'\treturn m.{self.x[1]}[t] <= m.{self.x[8]}\n'
                        f'model.Constraint_{self.x[1]}{self.x[8]} = pyo.Constraint(model.t, rule=Constraint_{self.x[1]}{self.x[8]})')
        #  H[t] <= A
        self.eqs.append(f'def Constraint_{self.x[2]}{self.x[5]}(m, t): \n'
                        f'\treturn m.{self.x[2]}[t] <= m.{self.x[5]}\n'
                        f'model.Constraint_{self.x[2]}{self.x[5]} = pyo.Constraint(model.t, rule=Constraint_{self.x[2]}{self.x[5]})')
        #  rpm[t] <= rpm_n
        self.eqs.append(f'def Constraint_{self.x[3]}{self.x[9]}(m, t): \n'
                        f'\treturn m.{self.x[3]}[t] <= m.{self.x[9]}\n'
                        f'model.Constraint_{self.x[3]}{self.x[9]} = pyo.Constraint(model.t, rule=Constraint_{self.x[3]}{self.x[9]})')
        #  Pe[t] <= P_max
        self.eqs.append(f'def Constraint_{self.x[4]}{self.x[7]}(m, t): \n'
                        f'\treturn m.{self.x[4]}[t] <= m.{self.x[7]}\n'
                        f'model.Constraint_{self.x[4]}{self.x[7]} = pyo.Constraint(model.t, rule=Constraint_{self.x[4]}{self.x[7]})')
        #  Ph_b_max = rho*g*Q_max*A (max version)
        self.eqs.append(f'def Constraint_{self.x[7]}(m, t): \n'
                        f'\treturn m.{self.x[7]} == {self.rho_g} * m.{self.x[5]} * m.{self.x[8]}\n'
                        f'model.Constraint_{self.x[7]} = pyo.Constraint(model.t, rule=Constraint_{self.x[7]})')


class turbine(syst_element):
    
    def __init__(self, system, ID):
        super().__init__(ID, [f'P_trb{ID}', f'Q_trb{ID}'])
        self.system = system


class turbine_simple(syst_element):
    
    def __init__(self, system, ID, p_max, Q_max, efficiency, in_pipe, EB_loc):
        super().__init__(ID, [f'P_trb{ID}', f'Q_trb{ID}'])
        self.system = system    
        self.p_max = p_max
        self.Q_max = Q_max
        self.efficiency = efficiency
        self.verification = True
        self.conn = self.conns(in_pipe)
        self.loc = EB_loc
        self.var = ['model.'+self.x[0]+' = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(0.0, '+str(self.p_max)+'),  initialize={k:'+str(0.8*self.p_max)+' for k in range(n)},)',
                    'model.'+self.x[1]+' = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(0.0, '+str(self.Q_max)+'),   initialize={k:'+str(0.8*self.Q_max)+'   for k in range(n)},)',
                    ]
        
    def conns(self, conn):
        if str(conn) in list(self.system.pipes.keys()):
            self.system.pipes[str(conn)].parallel_trbs.append(self.ID)
            return conn
        else:
            print('Associated pipe ID does not match any created. Pump eliminated')
            self.verification *= False
            
    def link_to(self):
        self.system.EBs[f'{self.loc}'].P_in.append(self.x[0])
        
    def eq_write(self):
        self.eqs.append(f'def Constraint_{self.x[0]}(m, t): \n'
                        f'\treturn m.{self.x[0]}[t] == {self.system.rho_g}*m.{self.x[1]}[t] * 20 * {self.efficiency}\n'
                        f'model.Constraint_{self.x[0]} = pyo.Constraint(model.t, rule=Constraint_{self.x[0]})')
        
        
class pipe(syst_element):
    
    def __init__(self, system, ID, K_i, H_0, orig, end, valve, C_v, alpha):
        super().__init__(ID, [f'Q_p{ID}', f'H_p{ID}'])
        self.system = system
        self.K_i = K_i
        self.H_0 = H_0
        self.Q_max = 1e6
        self.verification = True
        self.parallel_pumps = list() # list of ID (int) of pumps in parallel
        self.parallel_trbs = list() # list of ID (int) of pumps in parallel
        self.orig = self.inpt(orig)
        self.end = self.outpt(end)
        self.valve = valve
        self.C_v = C_v
        self.alpha = alpha # TODO: alpha es variable, no param
        self.var = ['model.'+self.x[0]+' = pyo.Var(model.t, within=pyo.Reals, bounds=('+str(-self.Q_max)+', '+str(self.Q_max)+'),   initialize={k:'+str(0.8*self.Q_max)+'   for k in range(n)},)',
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
            
    def link_to(self):
        self.system.rsvrs[f'{self.end}'].Q_in.append(self.x[0])
        self.system.rsvrs[f'{self.orig}'].Q_out.append(self.x[0])
        
    def eq_write(self):
        #  Hp = Ho + K*Q^2
        self.eqs.append(f'def Constraint_{self.x[1]}(m, t): \n'
                        f'\treturn m.{self.x[1]}[t] == {self.H_0} + {self.K_i}*m.{self.x[0]}[t]\n'
                        f'model.Constraint_{self.x[1]} = pyo.Constraint(model.t, rule=Constraint_{self.x[1]})')

        if len(self.parallel_pumps)>0:
            qb_txt = ''
            for b in self.parallel_pumps:
                qb_txt += f'+ m.Q_b{b}[t] '
                #  Hp = Hb1 = Hb2...
                self.eqs.append(f'def Constraint_{self.x[1]}_H_b{b}(m, t): \n'
                                f'\treturn m.{self.x[1]}[t] == m.H_b{b}[t]\n'
                                f'model.Constraint_{self.x[1]}_H_b{b} = pyo.Constraint(model.t, rule=Constraint_{self.x[1]}_H_b{b})')
            # Qp = sum(Qb)
            self.eqs.append(f'def Constraint_{self.x[0]}(m, t): \n'
                            f'\treturn m.{self.x[0]}[t] == '+qb_txt+'\n'
                            f'model.Constraint_{self.x[0]} = pyo.Constraint(model.t, rule=Constraint_{self.x[0]})')
    
        if len(self.parallel_trbs)>0:
            qt_txt = ''
            for trb in self.parallel_trbs:
                qt_txt += f'- m.Q_trb{trb}[t] '
            # qt_txt = qt_txt[:-3]
            self.eqs.append(f'def Constraint_{self.x[0]}(m, t): \n'
                            f'\treturn m.{self.x[0]}[t] == '+qt_txt+'\n'
                            f'model.Constraint_{self.x[0]} = pyo.Constraint(model.t, rule=Constraint_{self.x[0]})')
    
    
class PV(syst_element):
    
    def __init__(self, system, ID, EB_loc, p_inst, p_max):
        super().__init__(ID, [f'P_pv_g{ID}', f'P_pv_dim{ID}'])
        self.system = system
        self.eb_loc = self.conn(EB_loc)
        self.p_inst = p_inst
        self.p_max  = p_max
        self.verification = True
        self.var = ['model.'+self.x[0]+' = pyo.Var(model.t, within=pyo.NonNegativeReals,  initialize={k: 0.0 for k in range(n)},)',
                    'model.'+self.x[1]+' = pyo.Var(within=pyo.NonNegativeReals, bounds=(0.0, '+str(self.p_max)+'),   initialize='+str(0.5*self.p_max)+',)',]

    def link_to(self):
        self.system.EBs[f'{self.eb_loc}'].P_in.append(self.x[0])
        
    def conn(self, loc):
        if str(loc) in list(self.system.EBs.keys()):
            return loc
        else:
            print('Associated EB ID does not match any created. PV eliminated')
            self.verification *= False

    def eq_write(self):
        self.eqs.append(f'def Constraint_{self.x[0]}(m, t):\n'
                        f'\treturn m.{self.x[0]}[t] <= ({self.p_inst} + m.{self.x[1]})*m.forecast_pv{self.ID}[t]\n'
                        f'model.Constraint_{self.x[0]} = pyo.Constraint(model.t, rule=Constraint_{self.x[0]})')
        self.eqs.append(f'def Constraint_{self.x[1]}(m):\n'
                        f'\treturn m.{self.x[1]} <= {self.p_max}\n'
                        f'model.Constraint_{self.x[1]} = pyo.Constraint(rule=Constraint_{self.x[1]})')
        
        
class battery(syst_element):
    
    def __init__(self, system, ID, EB_loc, P_bat):
        super().__init__(ID, [f'Pdim_bat{ID}', f'Pchg_bat{ID}', f'Pdchg_bat{ID}', f'E_bat{ID}', f'SOC_bat{ID}'])
        self.system = system
        self.eb_loc = self.conn(EB_loc)
        self.P_bat = P_bat
        self.verification = True
        self.var = ['model.'+self.x[0]+' = pyo.Var(within=pyo.NonNegativeReals, bounds=(0.0, '+str(self.P_bat)+'),   initialize={k:'+str(0.5*self.P_bat)+'   for k in range(n)},)',
                    'model.'+self.x[1]+' = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(0.0, '+str(self.P_bat)+'),   initialize={k:'+str(0.5*self.P_bat)+'   for k in range(n)},)',
                    'model.'+self.x[2]+' = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(0.0, '+str(self.P_bat)+'),   initialize={k:'+str(0.5*self.P_bat)+'   for k in range(n)},)',
                    'model.'+self.x[3]+' = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(0.0, '+str(self.P_bat)+'),   initialize={k:'+str(0.5*self.P_bat)+'   for k in range(n)},)',
                    'model.'+self.x[4]+' = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(0.0, '+str(self.P_bat)+'),   initialize={k:'+str(0.5*self.P_bat)+'   for k in range(n)},)',]
    
    def link_to(self):
        self.system.EBs[f'{self.eb_loc}'].P_bal.append(self.x[0])
    
    def conn(self, loc):
        if str(loc) in list(self.system.EBs.keys()):
            return loc
        else:
            print('Associated EB ID does not match any created. Battery eliminated')
            self.verification *= False
            
    def eq_write(self):
        self.eqs.append(f'def Constraint_dummy_bat{self.ID}(m, t):\n'
                        f'\treturn m.{self.x[1]}[t]*m.{self.x[2]}[t] == 0.0\n'
                        f'model.Constraint_dummy_bat{self.ID} = pyo.Constraint(model.t, rule=Constraint_dummy_bat{self.ID})')
        self.eqs.append(f'def Constraint_{self.x[1]}(m, t):\n'
                        f'\treturn m.{self.x[1]}[t] <= self.P_bat + m.{self.x[0]}\n'
                        f'model.Constraint_{self.x[1]} = pyo.Constraint(rule=Constraint_{self.x[1]})')
        self.eqs.append(f'def Constraint_{self.x[2]}(m, t):\n'
                        f'\treturn m.{self.x[2]}[t] <= self.P_bat + m.{self.x[0]}\n'
                        f'model.Constraint_{self.x[2]} = pyo.Constraint(rule=Constraint_{self.x[2]})')