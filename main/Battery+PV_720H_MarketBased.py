# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:07:16 2024

@author: mvalois
"""

"""
    AGISTIN - ExampleBattery
    
    Optimization usage example of the data parser. It considers only a battery model connected to the main grid, and
    reads the data from an external spreadsheet file.
    
    Authors: Manuel Valois (Uni-Kassel), Nishat Jillani (Uni-Kassel)
"""
# Import pyomo
import pyomo.environ as pyo
from pyomo.network import Arc, Port
import pandas as pd

# Import builder
from BuilderBatteryPV import data_parser, builder

data_filename = "ExampleBatteryPV720H"

# generate system json file
data_parser(data_filename, dt=1) # dt = value of each timestep (if using SI this is seconds)

m = pyo.ConcreteModel()

# time
l_t = list(range(720)) #999 #TODO this should be inferred from the number of rows in the excel time series,
#TODO it would be nice to have a consistency check ensuring that data has been correctly filled in all sheets.
m.t = pyo.Set(initialize=l_t)
builder(m, data_filename)


#%% RUN THE OPTIMIZATION

# DFINITION OF VARIABLES

m.max_EnerStorg = pyo.Var(domain=pyo.NonNegativeReals)
m.W = pyo.Var(domain=pyo.NonNegativeReals)
m.max_EnerStorg = pyo.Var(domain=pyo.NonNegativeReals)
# Add the max_SOC constraint to the model



# Constraint to ensure max_SOC is greater than or equal to all SOC values
def max_EnerStorg_constraint(m, t):
    return m.max_EnerStorg >= m.Battery.EnerStorg[t]

m.max_EnerStorg_constraint = pyo.Constraint(m.t, rule=max_EnerStorg_constraint)

# def calculate_max_values(m):
#     #_values = [pyo.value(m.Battery.Y[t]) for t in m.t]
#     #X_values = [pyo.value(m.Battery.X[t]) for t in m.t]
#     SOC_values = [pyo.value(m.Battery.SOC[t]) for t in m.t]

#     #max_Y = max(Y_values)
#     #ax_X = max(X_values)
#     max_SOC = max(SOC_values)

#     #W = max(max_Y, max_X)
#     m.max_SOC.values = max_SOC
#     #m.W.value = W

cost_new_battery_MW  = 0
cost_new_battery_MWh = 0
cost_new_pv = 0

def obj_fun(m):
    return ((sum((abs(m.Battery.P_FCRCharge[t]) + abs(m.Battery.P_FCRDisCharge[t]))*0.25*m.Battery.FCR_reBAP[t] + m.Battery.Pdim_FCR*m.Battery.FCR_Remuneration[t] for t in l_t))
            - cost_new_battery_MW*m.Battery.Pdim_FCR - cost_new_battery_MWh*m.Battery.Edim_FCR - cost_new_pv*m.PV.Pdim)
m.goal = pyo.Objective(rule=obj_fun, sense=pyo.maximize)

#def obj_fun(m):
#    return ((sum((abs(m.Battery.P_FCRCharge[t]) + abs(m.Battery.P_FCRDisCharge[t]))*0.25*m.Battery.FCR_reBAP[t] + m.Battery.Pdim_FCR*m.Battery.FCR_Remuneration[t] for t in l_t))
#            - cost_new_battery_MW*m.Battery.Pdim_FCR - cost_new_battery_MWh*m.Battery.Edim_FCR - cost_new_pv*m.PV.Pdim)
#m.goal = pyo.Objective(rule=obj_fun, sense=pyo.maximize)

objective_value = (m.goal)

#return  ((sum(((abs(m.Battery.Pout[t]))for t in l_t)))) # - (m.Battery.P_Add_Charge[t])*cost_P_Add_Charge for t in l_t)) ))#  - (cost_new_battery_MW*m.Battery.Pdim_FCR) - (cost_new_battery_MWh*m.Battery.Edim_FCR)) #- (cost_new_pv*m.PV.Pdim) #-(m.Battery.P_FCRDisCharge[t] + m.Battery.P_FCRDisCharge[t] + m.Grid.P[t])

#um(((abs(m.Battery.Pout[t]))*m.Battery.FCR_reBAP[t])

instance = m.create_instance()
options = {'mumps_icntl_13': 64000}  # Example: start with a larger value, sometime without this an optimal solution is not posible
solver = pyo.SolverFactory('ipopt')
#solver.options['tol'] = 1e-6
solver.options['max_iter'] = 100000 # Increase maximum iterations
#solver.options['tol'] = 1e-6
# solver.options['acceptable_tol'] = 1e-5
# solver.options['max_iter'] = 10000  # Increase maximum iterations
# solver.options['mu_init'] = 1e-2  # Initial barrier parameter
#solver.options['bound_push'] = 1e-2  # Push bounds to avoid numerical issues
solver.options['hessian_approximation'] = 'limited-memory'  # Use Hessian approximation

#solver.options['tol'] = 1e-1  # Convergence tolerance
#solver.options['max_iter']= 1000000

result = solver.solve(instance, tee=True) # True False)

instance.Battery.Pdim_FCR.pprint()
instance.Battery.Edim_FCR.pprint()

instance.PV.Pinst.pprint()
instance.PV.Pdim.pprint()

FCR_Neg_values = [pyo.value(instance.Battery.FCR_Neg[t]) for t in instance.t]

EnerStorg_values = [pyo.value(instance.Battery.EnerStorg[t]) for t in instance.t]
PowerFCRDisCharge_values = [pyo.value(instance.Battery.P_FCRDisCharge[t]) for t in instance.t]

PowerFCRCharge_values = [pyo.value(instance.Battery.P_FCRCharge[t]) for t in instance.t]
PowerFCRP_Add_Charge_values = [pyo.value(instance.Battery.P_Add_Charge[t]) for t in instance.t]

PowerPV_values = [pyo.value(instance.PV.P[t]) for t in instance.t]
GridP_values = [pyo.value(instance.Grid.P[t]) for t in instance.t]

Pout_values = [pyo.value(instance.Battery.Pout[t]) for t in instance.t]


print("FCR_Neg_values:")
for t, value in enumerate(FCR_Neg_values):
   print(f"Time {t}: {value}")

print("PowerFCRCharge_values:")
for t, value in enumerate(PowerFCRCharge_values):
    print(f"Time {t}: {value}")

print("PowerFCRP_Add_Charge_values:")
for t, value in enumerate(PowerFCRP_Add_Charge_values):
    print(f"Time {t}: {value}")

print("PowerFCRDisCharge_values:")
for t, value in enumerate(PowerFCRDisCharge_values):
    print(f"Time {t}: {value}")

print("EnerStorg_values:")
for t, value in enumerate(EnerStorg_values):
    print(f"Time {t}: {value}")

print("PowerPV_values:")
for t, value in enumerate(PowerPV_values):
    print(f"Time {t}: {value}")

print("GridP_values:")
for t, value in enumerate(GridP_values):
    print(f"Time {t}: {value}")

print("Pout_values:")
for t, value in enumerate(Pout_values):
   print(f"Time {t}: {value}")

#Pfeedin_values = [pyo.value(instance.Battery.Pfeedin[t]) for t in instance.t]
#Pfeedout_values = [pyo.value(instance.Battery.Pfeedout[t]) for t in instance.t]

#max_Y = max(Y_values)
#max_X = max(X_values)
max_EnerStorg = max(EnerStorg_values)

# Compare max(X) and max(Y) and
# find out one max value and store in W.

#W = max(max_Y, max_X)

#print("Maximum Y_Pfeedin:", max_Y)


#print("Maximum X_Pfeedout:", max_X)
print("Maximum EnerStorg:", max_EnerStorg)
# Print the objective value after solving
print("Objective function value:", pyo.value(m.goal))
#print("Maximum W:", W)


# Store the values in a dictionary for summary results
results_summary = {
    #'Maximum Y_Pfeedin': [max_Y],
    #'Maximum X_Pfeedout': [max_X],
    'Maximum EnerStorg': [max_EnerStorg],
    #'Maximum W': [W]
}

# Convert the dictionary to a pandas DataFrame
summary_df = pd.DataFrame(results_summary)

# Store the detailed time-series results in a dictionary
results_detailed = {
    'Time': l_t,
    'EnerStorg': EnerStorg_values,
    'PowerFCRDisCharge': PowerFCRDisCharge_values,
    'PowerFCRCharge': PowerFCRCharge_values,
    'PowerPV': PowerPV_values,
    'PowerGrid': GridP_values,
    #'Y': Y_values,
    #'X': X_values
}

# Convert the dictionary to a pandas DataFrame
detailed_df = pd.DataFrame(results_detailed)

# Create a Pandas Excel writer using openpyxl as the engine
with pd.ExcelWriter('optimization_results.xlsx', engine='openpyxl') as writer:
    # Write the summary DataFrame to the Excel file
    summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    # Write the detailed DataFrame to the Excel file
    detailed_df.to_excel(writer, sheet_name='Detailed', index=False)

print("Results have been written to 'optimization_results.xlsx'")


