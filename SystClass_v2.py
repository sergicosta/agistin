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
        self.Ho = {} # --> {(id,t):val}
        self.Qo = {} # --> {(id,t):val}
        self.no = {} # --> {(id,t):val}
        
    def add(self, id_elem, A, B, rpm_nom, Ho, Qo):
        super().add(id_elem)
        
        self.A[id_elem] = A
        self.B[id_elem] = B
        self.rpm_nom[id_elem] = rpm_nom
        
        # init variables
        self.Ho.update({(id_elem,t): Qo for t in l_t})
        self.Qo.update({(id_elem,t): Ho for t in l_t})
        self.no.update({(id_elem,t): rpm_nom for t in l_t})









pumps_set = Pumps()

pumps_set.add('Bomba1', A=106.3,B=(2e-5)*3600**2,rpm_nom=1450, Qo=0,Ho=0) 
pumps_set.add('Bomba2', A=106.3,B=(2e-5)*3600**2,rpm_nom=1450, Qo=0,Ho=0) 
pumps_set.add('Bomba1', A=106.3,B=(2e-5)*3600**2,rpm_nom=1450, Qo=0,Ho=0) 


model = pyo.AbstractModel()
model.t = pyo.Set(initialize=l_t)

l_Pump = pumps_set.id

# PARAMS
model.i_pumps = pyo.Set(initialize=l_Pump)
model.pumps_A = pyo.Param(model.i_pumps, initialize=pumps_set.A, within=pyo.NonNegativeReals)
model.pumps_B = pyo.Param(model.i_pumps, initialize=pumps_set.B, within=pyo.NonNegativeReals)
model.pumps_rpm_nom = pyo.Param(model.i_pumps, initialize=pumps_set.rpm_nom, within=pyo.NonNegativeReals)


# VARIABLES
model.pumps_H = pyo.Var(model.i_pumps, model.t, initialize=pumps_set.Ho, within=pyo.NonNegativeReals)
model.pumps_Q = pyo.Var(model.i_pumps, model.t, initialize=pumps_set.Qo, within=pyo.NonNegativeReals)
model.pumps_n = pyo.Var(model.i_pumps, model.t, initialize=pumps_set.no, within=pyo.NonNegativeReals)


# CONSTRAINTS
def Constraint_Qmax_Pumps(m, i_pumps, t):
    return m.pumps_Q[i_pumps, t] <= m.pumps_Qmax[i_pumps, t]
model.Constraint_Qmax_Pumps = pyo.Constraint(model.t, rule=Constraint_Qmax_Pumps)


def Constraint_H_pump(m, i_pumps, t): 
	return m.pumps_H[i_pumps, t] == ((m.pumps_n[i_pumps, t]/m.pumps_rpm_nom[i_pumps])**2) * m.pumps_A[i_pumps] - m.pumps_B[i_pumps]*(m.pumps_Q[i_pumps,t])**2
model.Constraint_H_pump = pyo.Constraint(model.t, rule=Constraint_H_pump)








# def Constraint_H_p0_H_b0(m, t): 
# 	return m.H_p0[t] == m.H_b0[t]
# model.Constraint_H_p0_H_b0 = pyo.Constraint(model.t, rule=Constraint_H_p0_H_b0)

# def 

#     m.pumps_H[i_Pumps, t] == m.pipes_H[m.pumps_in[i_Pumps], t]




















