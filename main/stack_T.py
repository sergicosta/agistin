import pyomo.environ as pyo

# Define the data
E0 = 2000
slope_fcr = -5
Pinst = 1000
F = [50.0, 50.2, 50.2, 50.05, 50.05, 50.1, 50.2, 50.0, 50.0, 49.8, 49.8, 49.8, 49.8, 49.8, 49.8, 49.8, 49.8, 49.8, 49.8, 49.8, 49.8, 49.95, 50.0, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.2, 50.0, 49.8, 49.8, 49.8, 49.8]

# Initialize Pyomo model
model = pyo.ConcreteModel()


# Constraint rule for estrout
def Constraint_EstrgOut_rule(model, t):
    if t == 0:
        return model.estrout[t] == E0 + (50 - F[t]) * Pinst * slope_fcr
    else:
        return model.estrout[t] == model.estrout[t - 1] + (50 - F[t]) * Pinst * slope_fcr
# Parameters
tot_time = len(F)
Price = [0.5 for _ in range(tot_time)]
model.EtaCharge = pyo.Param(initialize=0.9)  # Charging efficiency
model.EtaDischarge = pyo.Param(initialize=0.85)  # Discharging efficiency
model.PMinCharge = pyo.Param(initialize=0)  # Minimum charging power
model.PMaxCharge = pyo.Param(initialize=100)  # Maximum charging power
model.PMinDischarge = pyo.Param(initialize=0)  # Minimum discharging power
model.PMaxDischarge = pyo.Param(initialize=100)  # Maximum discharging power

# Sets
model.T = pyo.Set(initialize=[t for t in range(tot_time)])

# Variables
model.QEl = pyo.Var(model.T, within=pyo.NonNegativeReals, initialize=0)
model.QCharge = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.QDischarge = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.Charge = pyo.Var(model.T, within=pyo.Binary)
model.Discharge = pyo.Var(model.T, within=pyo.Binary)
model.SOC = pyo.Var(model.T, bounds=(0, 100))
# model.estrout = pyo.Var(model.T, within=pyo.NonNegativeReals)
model.estrout = pyo.Var(model.T, within=pyo.NonNegativeReals, bounds=(0, 24000))


# Constraints
model.C1 = pyo.ConstraintList()
model.C2 = pyo.ConstraintList()
model.C3 = pyo.ConstraintList()
model.C4 = pyo.ConstraintList()
model.C5 = pyo.ConstraintList()
model.C6 = pyo.ConstraintList()
model.C7 = pyo.ConstraintList()
model.C8 = pyo.ConstraintList()
model.C9 = pyo.ConstraintList()
model.C10 = pyo.Constraint(model.T, rule=Constraint_EstrgOut_rule)  # New constraint for estrout

for t in model.T:
    model.C1.add(model.QEl[t] * model.EtaCharge == model.QCharge[t])
    model.C2.add(model.QDischarge[t] * model.EtaDischarge == F[t])
    model.C3.add(model.Charge[t] + model.Discharge[t] <= 1)
    model.C4.add(model.QCharge[t] >= model.PMinCharge * model.Charge[t])
    model.C5.add(model.QCharge[t] <= model.PMaxCharge * model.Charge[t])
    model.C6.add(model.QDischarge[t] >= model.PMinDischarge * model.Discharge[t])
    model.C7.add(model.QDischarge[t] <= model.PMaxDischarge * model.Discharge[t])

    # State of Charge (SOC) constraints
    if t == 0:
        model.C8.add(model.SOC[t] == E0 + model.QCharge[t] - model.QDischarge[t])
    else:
        model.C8.add(model.SOC[t] == model.SOC[t-1] + model.QCharge[t] - model.QDischarge[t])

    # Final SOC constraint
    if t == tot_time - 1:
        model.C9.add(model.SOC[t] == model.SOC[0] + sum(model.QCharge[i] - model.QDischarge[i] for i in model.T))

# Objective
model.objective = pyo.Objective(expr=sum(model.QEl[t] * Price[t] for t in model.T), sense=pyo.minimize)



# Solve the model
opt = pyo.SolverFactory('ipopt')
results = opt.solve(model, tee=True)

# Display SOC and estrout
model.SOC.display()
model.estrout.display()
