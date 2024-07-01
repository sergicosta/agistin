import pyomo.environ as pyo

model = pyo.ConcreteModel()

##Parameters


tot_time=10 #Hours in a year
Demand = [0, 2, 3, 0, 0, 3, 3, 0, 0, 3]
Price = [0.5 for t in range(tot_time)]

model.T=pyo.Set(initialize=[t for t in range(0,tot_time)])
model.EtaCharge=pyo.Param(initialize=1.0, mutable=True) #-    
model.PMinCharge=0 #MW
model.PMaxCharge=5 #MW
model.EtaDischarge=pyo.Param(initialize=1.0, mutable=True) #-
model.PMinDischarge=2 #MW
model.PMaxDischarge=4 #MW

model.S0=pyo.Param(initialize=10, mutable=True) #MWh SOC at t=0

##Variables

model.QEl = pyo.Var(model.T, within=pyo.NonNegativeReals, initialize=0)
model.QCharge= pyo.Var(model.T, within=pyo.NonNegativeReals)
model.QDischarge= pyo.Var(model.T, within=pyo.NonNegativeReals)
model.Charge=pyo.Var(model.T, within = pyo.Binary)
model.Discharge=pyo.Var(model.T, within = pyo.Binary)
model.SOC=pyo.Var(model.T, bounds=(0,100)) #Size of the battery limited to 100 MWh

##Constraints

model.C1 = pyo.ConstraintList()
model.C2 = pyo.ConstraintList()
model.C3 = pyo.ConstraintList()
model.C4 = pyo.ConstraintList()
model.C5 = pyo.ConstraintList()
model.C6 = pyo.ConstraintList()
model.C7 = pyo.ConstraintList()

for t in model.T:
    
    model.C1.add(model.QEl[t] * model.EtaCharge == model.QCharge[t])   

    model.C2.add(model.QDischarge[t] * model.EtaDischarge == Demand[t])   
          
    model.C3.add(model.Charge[t] + model.Discharge[t] <= 1)
    
    model.C4.add(model.QCharge[t] >= model.PMinCharge*model.Charge[t])
                 
    model.C5.add(model.QCharge[t] <= model.PMaxCharge*model.Charge[t])
    
    model.C6.add(model.QDischarge[t] >= model.PMinDischarge*model.Discharge[t])
    
    model.C7.add(model.QDischarge[t] <= model.PMaxDischarge*model.Discharge[t])  


def SOC_storage(model,t):

    if t == 0:
        return (model.SOC[t] == model.S0 + model.QCharge[t] - model.QDischarge[t]) 
    else:
        return (model.SOC[t] == model.SOC[t-1] + model.QCharge[t] - model.QDischarge[t])

model.C8 = pyo.Constraint(model.T,rule = SOC_storage)

model.C9 = pyo.Constraint(expr = model.S0 == model.SOC[tot_time-1])

def NoCharge(model,t):
    if t in no_charge_zone :
        return model.Charge[t]==0
    else:
        return pyo.Constraint.Skip

no_charge_zone = [7, 8]
model.C10 = pyo.Constraint(model.T,rule = NoCharge)

# The below C11 is redundant with C10...  It just shows a different approach.
# I think this is cleaner....  Just make a subset from your "prohibited" time periods
# and pass that in...
def NoCharge2(model, t):
    return model.Charge[t] == 0

model.C11 = pyo.Constraint(no_charge_zone, rule=NoCharge2)

##Objective

model.objective = pyo.Objective(expr = sum(model.QEl[t]*Price[t] for t in model.T), sense = pyo.minimize)

opt=pyo.SolverFactory('ipopt')
results = opt.solve(model, tee = True)
#model.write("myfile_lp.lp", io_options={'symbolic_solver_labels':True})
results.solver.status
results.solver.termination_condition
#model.pprint()
model.Charge.display()
model.SOC.display()
#print(pyo.value(model.objective))