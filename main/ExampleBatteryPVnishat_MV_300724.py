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

data_filename = "ExampleBatteryPV"

# generate system json file
data_parser(data_filename, dt=1) # dt = value of each timestep (if using SI this is seconds)

m = pyo.ConcreteModel()

# time
l_t = list(range(8760)) #999 #TODO this should be inferred from the number of rows in the excel time series,
#TODO it would be nice to have a consistency check ensuring that data has been correctly filled in all sheets.
m.t = pyo.Set(initialize=l_t)
builder(m, data_filename)


#%% RUN THE OPTIMIZATION

# DFINITION OF VARIABLES

m.max_SOC = pyo.Var(domain=pyo.NonNegativeReals)
m.W = pyo.Var(domain=pyo.NonNegativeReals)
m.max_SOC = pyo.Var(domain=pyo.NonNegativeReals)
# Add the max_SOC constraint to the model



# Constraint to ensure max_SOC is greater than or equal to all SOC values
def max_soc_constraint(m, t):
    return m.max_SOC >= m.Battery.SOC[t]

m.max_SOC_constraint = pyo.Constraint(m.t, rule=max_soc_constraint)

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

l_cost = [0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]
#l_cost =[0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6]
m.cost = pyo.Param(m.t, initialize=l_cost) #-m.cost[t]
#cost_new_battery_MW = 1.853881279 #7.5 P_FCR=8827, 6.5 P_FCR=9975, 8.5 P_FCR=0 con Battery.P_FCRCharge[t]*0
#cost_new_battery_MW = 1.853881279
cost_new_battery_MW  = 25000
#cost_new_battery_MW = 28.53881279  # Significantly increased value
#cost_new_battery_MW = 40.53881279  # Significantly increased value
cost_new_battery_MWh = 2.2831050023 # either 405.7 or 505.7, results are the same
cost_new_pv = 0.05

#55 time steps
#m.cost = pyo.Param(m.t, initialize=l_cost) #-m.cost[t]
#cost_new_battery_MW = 5.7 #7.5 P_FCR=8827, 6.5 P_FCR=9975, 8.5 P_FCR=0 con Battery.P_FCRCharge[t]*0
#cost_new_battery_MWh = 505.7 # either 405.7 or 505.7, results are the same
#cost_new_pv = 0.05

def obj_fun(m):
    # calculate_max_values(m)
    # max_SOC = max(m.Battery.SOC[t] for t in m.t)
    return  (sum(((m.Battery.P_FCRCharge[t]*70) + (m.Battery.P_FCRDisCharge[t])*-60 for t in l_t))
            - (cost_new_battery_MW*m.Battery.P_FCR))  #- (cost_new_battery_MWh*m.max_SOC)) #- m.PV.Pdim*cost_new_pv  m.max_SOC
m.goal = pyo.Objective(rule=obj_fun, sense=pyo.maximize)

# objective_value = (m.goal)
# print(f"Objective function value: {objective_value}")

# find :

# obj of the sum .
# comment line 86
# change cost value honi chahya objective ki.


instance = m.create_instance()
solver = pyo.SolverFactory('ipopt')
# solver.options['tol'] = 1e-6
# solver.options['max_iter'] = 10000
# solver.options['tol'] = 1e-6
# solver.options['acceptable_tol'] = 1e-5
# solver.options['max_iter'] = 10000  # Increase maximum iterations
# solver.options['mu_init'] = 1e-2  # Initial barrier parameter
# solver.options['bound_push'] = 1e-2  # Push bounds to avoid numerical issues
# solver.options['hessian_approximation'] = 'limited-memory'  # Use Hessian approximation

#solver.options['tol'] = 1e-1  # Convergence tolerance
#solver.options['max_iter']= 1000000

result = solver.solve(instance, tee=False)

# Check if the model is solved successfully
#if result.solver.status == solver.status.ok and result.solver.termination_condition == TerminationCondition.optimal:
#    print("Optimization was successful.")
#else:
#    print("Optimization failed.")

#objective_value = value(m.goal)
#print(f"Objective function value: {objective_value}")

# - (2.853881279 * m.W)
# +(m.Battery.Pfeedin[t]*-0.3)+(m.Battery.Pfeedout[t]*-0.3)

# - (155 * m.W) ( these use for 55 time steps.)
# Lithium-ion (Li-ion) Batteries:
# - Price per MW: €500,000 - €800,000
# - Price per MWh: €200 - €400



instance.Battery.P_FCR.pprint()
#instance.PV.Pdim.pprint()
# instance.Battery.SOC.pprint()
#instance.Battery.Pout.pprint()
#instance.Battery.P.pprint()
#instance.Grid.P.pprint()
#instance.PV.P.pprint()
#instance.Battery.Pfeedout.pprint()
#instance.Battery.Y.pprint()
#instance.Battery.X.pprint()

#Y_values = [pyo.value(instance.Battery.Y[t]) for t in instance.t]
#X_values = [pyo.value(instance.Battery.X[t]) for t in instance.t]
SOC_values = [pyo.value(instance.Battery.SOC[t]) for t in instance.t]


PowerFCRDisCharge_values = [pyo.value(instance.Battery.P_FCRDisCharge[t]) for t in instance.t]
PowerFCRCharge_values = [pyo.value(instance.Battery.P_FCRCharge[t]) for t in instance.t]
#Pfeedin_values = [pyo.value(instance.Battery.Pfeedin[t]) for t in instance.t]
#Pfeedout_values = [pyo.value(instance.Battery.Pfeedout[t]) for t in instance.t]



#max_Y = max(Y_values)
#max_X = max(X_values)
max_SOC = max(SOC_values)

# Compare max(X) and max(Y) and
# find out one max value and store in W.

#W = max(max_Y, max_X)

#print("Maximum Y_Pfeedin:", max_Y)
#print("Maximum X_Pfeedout:", max_X)
print("Maximum SOC:", max_SOC)
#print("Maximum W:", W)


# Store the values in a dictionary for summary results
results_summary = {
    #'Maximum Y_Pfeedin': [max_Y],
    #'Maximum X_Pfeedout': [max_X],
    'Maximum SOC': [max_SOC],
    #'Maximum W': [W]
}

# Convert the dictionary to a pandas DataFrame
summary_df = pd.DataFrame(results_summary)

# Store the detailed time-series results in a dictionary
results_detailed = {
    'Time': l_t,
    'SOC': SOC_values,
    'PowerFCRDisCharge': PowerFCRDisCharge_values,
    'PowerFCRCharge': PowerFCRCharge_values,
    #'Pfeedin': Pfeedin_values,
    #'Pfeedout': Pfeedout_values,
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


