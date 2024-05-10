import pyomo.environ as pyo

# Define the model
m = pyo.ConcreteModel()

# Define sets
m.t = pyo.RangeSet(1, 55)

# Define parameters
m.cost_MainGrid = pyo.Param(m.t, initialize=0)
m.Grid.Pbuy = pyo.Param(m.t, initialize=0)
m.Grid.Psell = pyo.Param(m.t, initialize=0)
m.PV.P = pyo.Param(m.t, initialize=0)
m.PV.Pdim = pyo.Param(initialize=0)
m.PV.Pinst = pyo.Param(initialize=0)

# Define variables
m.Battery = pyo.Block()
m.Battery.EstrgOut = pyo.Var(m.t, within=pyo.NonNegativeReals)
m.Battery.SOC = pyo.Var(m.t, within=pyo.NonNegativeReals)
m.Battery.Pout = pyo.Var(m.t, within=pyo.Reals)
m.PV.Pdim = pyo.Var(within=pyo.NonNegativeReals)

# Define constraints
def estrg_out_non_negative_rule(m, t):
    return m.Battery.EstrgOut[t] >= 0

m.estrg_out_non_negative = pyo.Constraint(m.t, rule=estrg_out_non_negative_rule)

def estrg_out_rule(m, t):
    return m.Battery.Pout[t] == m.Battery.EstrgOut[t]

m.estrg_out_calculation = pyo.Constraint(m.t, rule=estrg_out_rule)

def soc_update_rule(m, t):
    return m.Battery.SOC[t] == m.Battery.SOC[t-1] - m.Battery.EstrgOut[t]

m.soc_update = pyo.Constraint(m.t, rule=soc_update_rule)

def soc_limits_rule(m, t):
    return 0 <= m.Battery.SOC[t] <= 100

m.soc_limits = pyo.Constraint(m.t, rule=soc_limits_rule)

# Addressing the modeling errors
# Deleting and re-adding components
del m.estrg_out_non_negative
del m.estrg_out_calculation
del m.soc_update
del m.soc_limits

m.estrg_out_non_negative = pyo.Constraint(m.t, rule=estrg_out_non_negative_rule)
m.estrg_out_calculation = pyo.Constraint(m.t, rule=estrg_out_rule)
m.soc_update = pyo.Constraint(m.t, rule=soc_update_rule)
m.soc_limits = pyo.Constraint(m.t, rule=soc_limits_rule)

# Explicitly deleting the attribute before reassigning
if hasattr(m, 'cost_PV1'):
    del m.cost_PV1

m.cost_PV1 = pyo.Param(initialize=0)  # Reassign with proper initialization

# Addressing the IndexError in the objective function
def obj_fun(m):
    total_cost = sum((m.Grid.Pbuy[t]*m.cost_MainGrid[t] - m.Grid.Psell[t]*m.cost_MainGrid[t]/2) for t in m.t)
    if m.cost_PV1:
        return total_cost + m.PV.Pdim*m.cost_PV1
    else:
        return total_cost

m.obj = pyo.Objective(rule=obj_fun, sense=pyo.minimize)

# Create instance and solve
instance = m.create_instance()
solver = pyo.SolverFactory('ipopt')  # Use ipopt solver
solver.solve(instance, tee=False)

# Print the results
instance.Battery.EstrgOut.pprint()
instance.Battery.SOC.pprint()
instance.Battery.Pout.pprint()
instance.PV.P.pprint()
instance.PV.Pdim.pprint()
instance.PV.Pinst.pprint()
