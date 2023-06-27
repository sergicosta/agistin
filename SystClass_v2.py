import pyomo.environ as pyo


l_t = list(range(5))


class Elements():
    def __init__(self):
        self.id = []
        pass
    
    def add(self, id_elem):
        if id_elem in self.id:
            print("*** ERROR: ID "+ str(id_elem) + " already exists in " + str(self.id) + " ***")
            return -1
        self.id.append(id_elem)



class Pumps(Elements):
    
    def __init__(self):
        super().__init__()
        
        self.A = {} # --> {id:val}
        self.B = {}
        self.rpm_nom = {}
        
        # init variables
        self.init_H = {} # --> {(id,t):val}
        self.init_Q = {} # --> {(id,t):val}
        self.init_n = {} # --> {(id,t):val}
        
    def add(self, id_elem, A, B, rpm_nom, init_H, init_Q, init_n=None):
        super().add(id_elem)
        
        self.A[id_elem] = A
        self.B[id_elem] = B
        self.rpm_nom[id_elem] = rpm_nom
        
        # init variables
        self.init_H.update({(id_elem,t): init_Q for t in l_t})
        self.init_Q.update({(id_elem,t): init_H for t in l_t})
        if init_n==None:
            init_n = rpm_nom
        self.init_n.update({(id_elem,t): init_n for t in l_t})




class Reservoirs(Elements):
    
    def __init__(self):
        super().__init__()
        
        self.W_0 = {}
        self.W_min = {}
        self.W_max = {}
        self.z_min = {}
        self.z_max = {}
        self.id_in = {}
        self.id_out = {}
        self.Q_cons = {} # --> {(id,t):val}
        
        # init variables
        self.init_W = {} # --> {(id,t):val}
        
        
    def add(self, id_elem, id_in, id_out, W_0, W_min, W_max, z_min, z_max, init_W, Q_cons):
        super().add(id_elem)
        
        self.W_0[id_elem] = W_0
        self.W_min[id_elem] = W_min
        self.W_max[id_elem] = W_max
        self.z_min[id_elem] = z_min
        self.z_max[id_elem] = z_max
        self.id_in[id_elem] = id_in
        self.id_out[id_elem] = id_out
        self.Q_cons.update({(id_elem,t): Q_cons[t] for t in l_t}) # --> {(id,t):val}

        # init variables
        self.init_W.update({(id_elem,t): init_W for t in l_t})
        
        
          
# TODO
class Pipes(Elements):
    
    def __init__(self):
        super().__init__()
        
        self.K = {}
        self.Qmax = {}
        self.orig = {}
        self.end = {}
        
        # init variables
        self.init_H = {} # --> {(id,t):val}
        self.init_Q = {} # --> {(id,t):val}
        
        
    def add(self, id_elem, K, id_in, id_out, Qmax=1e6):
        super().add(id_elem)
        
        self.K[id_elem] = K
        self.Qmax[id_elem] = Qmax
        self.id_in[id_elem] = id_in
        self.id_out[id_elem] = id_out
    


pumps_set = Pumps()
res_set = Reservoirs()

pumps_set.add('Bomba1', A=106.3,B=(2e-5)*3600**2,rpm_nom=1450, init_Q=0,init_H=0) 
pumps_set.add('Bomba2', A=106.3,B=(2e-5)*3600**2,rpm_nom=1450, init_Q=0,init_H=0) 
pumps_set.add('Bomba1', A=106.3,B=(2e-5)*3600**2,rpm_nom=1450, init_Q=0,init_H=0) 

res_set.add('Bassa1', [], ['Pipe1'], 142869*0.5, 142869, 142869*0.5, 328, 335.50, 142869*0.5, [0,0,0,0,0])
res_set.add('Bassa3', ['Pipe1'], [], 85268*0.5, 85268, 85268*0.5, 414, 423.50, 85268*0.5, [0,0,0,0,0])


model = pyo.AbstractModel()
model.t = pyo.Set(initialize=l_t)

l_Pump = pumps_set.id
l_Res = res_set.id

# PARAMS
#pumps
model.i_pumps = pyo.Set(initialize=l_Pump)
model.pumps_A = pyo.Param(model.i_pumps, initialize=pumps_set.A, within=pyo.NonNegativeReals)
model.pumps_B = pyo.Param(model.i_pumps, initialize=pumps_set.B, within=pyo.NonNegativeReals)
model.pumps_rpm_nom = pyo.Param(model.i_pumps, initialize=pumps_set.rpm_nom, within=pyo.NonNegativeReals)

#reservoirs
model.i_res = pyo.Set(initialize=l_Res)
model.res_W_0 = pyo.Param(model.i_res, initialize=res_set.W_0, within=pyo.NonNegativeReals)
model.res_W_min = pyo.Param(model.i_res, initialize=res_set.W_min, within=pyo.NonNegativeReals)
model.res_W_max = pyo.Param(model.i_res, initialize=res_set.W_max, within=pyo.NonNegativeReals)
model.res_z_min = pyo.Param(model.i_res, initialize=res_set.z_min, within=pyo.NonNegativeReals)
model.res_z_max = pyo.Param(model.i_res, initialize=res_set.z_max, within=pyo.NonNegativeReals)
model.res_id_in = pyo.Param(model.i_res, initialize=res_set.id_in, within=pyo.NonNegativeReals)
model.res_id_out = pyo.Param(model.i_res, initialize=res_set.id_out, within=pyo.NonNegativeReals)
model.res_Q_cons = pyo.Param(model.i_res, model.t, initialize=res_set.Q_cons, within=pyo.NonNegativeReals)

# VARIABLES
#pumps
model.pumps_H = pyo.Var(model.i_pumps, model.t, initialize=pumps_set.init_H, within=pyo.NonNegativeReals)
model.pumps_Q = pyo.Var(model.i_pumps, model.t, initialize=pumps_set.init_Q, within=pyo.NonNegativeReals)
model.pumps_n = pyo.Var(model.i_pumps, model.t, initialize=pumps_set.init_n, within=pyo.NonNegativeReals)

#reservoirs
model.res_W = pyo.Var(model.i_res, model.t, initialize=res_set.init_W, within=pyo.NonNegativeReals)


# CONSTRAINTS
#pumps
def Constraint_Qmax_Pumps(m, i_pumps, t):
    return m.pumps_Q[i_pumps, t] <= m.pumps_Qmax[i_pumps, t]
model.Constraint_Qmax_Pumps = pyo.Constraint(model.t, rule=Constraint_Qmax_Pumps)


def Constraint_H_Pumps(m, i_pumps, t): 
	return m.pumps_H[i_pumps, t] == ((m.pumps_n[i_pumps, t]/m.pumps_rpm_nom[i_pumps])**2) * m.pumps_A[i_pumps] - m.pumps_B[i_pumps]*(m.pumps_Q[i_pumps,t])**2
model.Constraint_H_Pumps = pyo.Constraint(model.t, rule=Constraint_H_Pumps)


# TODO: Pipes
# def Constraint_Hpumps_Hpipes(m, i_pumps, t):
#     return m.pumps_H[i_pumps, t] == m.pipes_H[m.pipes_in[i_pumps], t]
# model.Constraint_Hpumps_Hpipes = pyo.Constraint(model.t, rule=Constraint_Hpumps_Hpipes)


#reservoirs
def Constraint_Wmax_Res(m, i_res, t):
    return m.res_W[i_res, t] <= m.res_W_max[i_res, t]
model.Constraint_Wmax_Res = pyo.Constraint(model.t, rule=Constraint_Wmax_Res)
def Constraint_Wmin_Res(m, i_res, t):
    return m.res_W[i_res, t] >= m.res_W_min[i_res, t]
model.Constraint_Wmin_Res = pyo.Constraint(model.t, rule=Constraint_Wmin_Res)



#TODO
def Constraint_W_Res(m, i_res, t): 
	if t>0:
		return m.res_W[i_res, t] == m.res_W[i_res, t-1] + sum(m.pipes_Q[pipe, t] for pipe in m.res_id_in[i_res]) - sum(m.pipes_Q[pipe, t] for pipe in m.res_id_out[i_res]) - m.res_Q_cons[i_res, t]
	else:
		return m.res_W[i_res, t] == m.res_W_0[i_res] + sum(m.pipes_Q[pipe, t] for pipe in m.res_id_in[i_res]) - sum(m.pipes_Q[pipe, t] for pipe in m.res_id_out[i_res]) - m.res_Q_cons[i_res, t]
   	# TODO: add gamma, Dt
    # if t>0:
   	# 	return m.res_W[i_res, t] == m.res_W[i_res, t-1]*(1 -m.gamma_0[t]) - m.Dt*(m.Q_p0[t] + m.q_irr_0[t])
   	# else:
   	# 	return m.W_r0[t] == 142869 - m.Dt*(m.Q_p0[t] + m.q_irr_0[t])
model.Constraint_W_Res = pyo.Constraint(model.t, rule=Constraint_W_Res)

















