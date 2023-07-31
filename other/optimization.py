# -*- coding: utf-8 -*-

import pyomo.environ as pyo
import json
# from pyomo.opt import SolverFactory
model = pyo.AbstractModel()

with open("ext_data.json","r") as file:
	data=json.load(file)

n = len(data["cost_elec"])
l_t = list(range(n))
model.t = pyo.Set(initialize=l_t)
model.Dt = pyo.Param(initialize=1.0)
model.cost_elec = pyo.Param(model.t, initialize=data["cost_elec"])
model.cost_pv_inst = pyo.Param(model.t, initialize=data["cost_pv_inst"])
model.gamma_0 = pyo.Param(model.t, initialize=data["gamma_0"])
model.q_irr_0 = pyo.Param(model.t, initialize=data["q_irr_0"])
model.gamma_1 = pyo.Param(model.t, initialize=data["gamma_1"])
model.q_irr_1 = pyo.Param(model.t, initialize=data["q_irr_1"])
model.forecast_pv0 = pyo.Param(model.t, initialize=data["weather_0"])

model.W_r0 = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(14286.900000000001, 142869), initialize={k:142869 for k in range(n)},)
model.z_r0 = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(328, 335.5), initialize={k:(142869-14286.900000000001)/(128582.1)*(7.5)+328 for k in range(n)},)
model.W_r1 = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(8526.800000000001, 85268), initialize={k:42634.0 for k in range(n)},)
model.z_r1 = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(414, 423.5), initialize={k:(42634.0-8526.800000000001)/(76741.2)*(9.5)+414 for k in range(n)},)
model.P_g0 = pyo.Var(model.t, within=pyo.Reals, bounds=(-100000000.0, 100000000.0),  initialize={k:0.0 for k in range(n)},)
model.Ph_b0 = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(0.0, 275900.0),  initialize={k:220720.0 for k in range(n)},)
model.Q_b0 = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(0.0, 0.32222222222222224),   initialize={k:0.2847222222222222   for k in range(n)},)
model.H_b0 = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(None, None), initialize={k:106.3 for k in range(n)},)
model.rpm_0 = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(0.0, 1480), initialize={k:1184.0 for k in range(n)},)
model.Pe_b0 = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(0.0, 275900.0),   initialize={k:220720.0 for k in range(n)},)
model.nu_b0 = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(0.0, 0.85), initialize={k:0.85 for k in range(n)},)
model.Ph_b1 = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(0.0, 275900.0),  initialize={k:220720.0 for k in range(n)},)
model.Q_b1 = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(0.0, 0.32222222222222224),   initialize={k:0.2847222222222222   for k in range(n)},)
model.H_b1 = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(None, None), initialize={k:106.3 for k in range(n)},)
model.rpm_1 = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(0.0, 1480), initialize={k:1184.0 for k in range(n)},)
model.Pe_b1 = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(0.0, 275900.0),   initialize={k:220720.0 for k in range(n)},)
model.nu_b1 = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(0.0, 0.85), initialize={k:0.85 for k in range(n)},)
model.Ph_b2 = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(0.0, 275900.0),  initialize={k:220720.0 for k in range(n)},)
model.Q_b2 = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(0.0, 0.32222222222222224),   initialize={k:0.2847222222222222   for k in range(n)},)
model.H_b2 = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(None, None), initialize={k:200 for k in range(n)},)
model.rpm_2 = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(0.0, 1480), initialize={k:1184.0 for k in range(n)},)
model.Pe_b2 = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(0.0, 275900.0),   initialize={k:220720.0 for k in range(n)},)
model.nu_b2 = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(0.0, 0.85), initialize={k:0.85 for k in range(n)},)
model.Q_p0 = pyo.Var(model.t, within=pyo.Reals, bounds=(-1000000.0, 1000000.0),   initialize={k:800000.0   for k in range(n)},)
model.H_p0 = pyo.Var(model.t, within=pyo.NonNegativeReals, bounds=(None, None),   initialize={k:95 for k in range(n)},)
model.P_pv_g0 = pyo.Var(model.t, within=pyo.NonNegativeReals,  initialize={k: 0.0 for k in range(n)},)
model.P_pv_dim0 = pyo.Var(within=pyo.NonNegativeReals, bounds=(0.0, 0),   initialize=0.0,)

def Constraint_W_r0(m, t): 
	if t>0:
		return m.W_r0[t] == m.W_r0[t-1]*(1 -m.gamma_0[t]) - m.Dt*(m.Q_p0[t] + m.q_irr_0[t])
	else:
		return m.W_r0[t] == 142869 - m.Dt*(m.Q_p0[t] + m.q_irr_0[t])
model.Constraint_W_r0 = pyo.Constraint(model.t, rule=Constraint_W_r0)

def Constraint_z_r0(m, t): 
	return m.z_r0[t] == (m.W_r0[t]-14286.900000000001)/(128582.1)*(7.5)+328
model.Constraint_z_r0 = pyo.Constraint(model.t, rule=Constraint_z_r0)

def Constraint_W_r1(m, t): 
	if t>0:
		return m.W_r1[t] == m.W_r1[t-1]*(1 -m.gamma_1[t]) + m.Dt*(m.Q_p0[t])- m.Dt*(m.q_irr_1[t])
	else:
		return m.W_r1[t] == 42634.0 + m.Dt*(m.Q_p0[t])- m.Dt*(m.q_irr_1[t])
model.Constraint_W_r1 = pyo.Constraint(model.t, rule=Constraint_W_r1)

def Constraint_z_r1(m, t): 
	return m.z_r1[t] == (m.W_r1[t]-8526.800000000001)/(76741.2)*(9.5)+414
model.Constraint_z_r1 = pyo.Constraint(model.t, rule=Constraint_z_r1)

def Constraint_P_g0(m, t): 
	return m.P_g0[t] == - (m.P_pv_g0[t])+ (m.Pe_b0[t] + m.Pe_b1[t] + m.Pe_b2[t])
model.Constraint_P_g0 = pyo.Constraint(model.t, rule=Constraint_P_g0)

def Constraint_H_b0(m, t): 
	return m.H_b0[t] == ((m.rpm_0[t]/1480)**2) * 106.3 - 259.20000000000005*(m.Q_b0[t])**2
model.Constraint_H_b0 = pyo.Constraint(model.t, rule=Constraint_H_b0)

def Constraint_Ph_b0(m, t): 
	return m.Ph_b0[t] == 9810.0 * m.Q_b0[t] * m.H_b0[t]
model.Constraint_Ph_b0 = pyo.Constraint(model.t, rule=Constraint_Ph_b0)

def Constraint_Pe_b0(m, t): 
	return m.Pe_b0[t]*m.nu_b0[t] == m.Ph_b0[t]
model.Constraint_Pe_b0 = pyo.Constraint(model.t, rule=Constraint_Pe_b0)

def Constraint_nu_b0(m, t): 
	return m.nu_b0[t] == 1.0 - (1.0 - 0.85)*((1480/m.rpm_0[t])**0.1)
model.Constraint_nu_b0 = pyo.Constraint(model.t, rule=Constraint_nu_b0)

def Constraint_H_b1(m, t): 
	return m.H_b1[t] == ((m.rpm_1[t]/1480)**2) * 106.3 - 259.20000000000005*(m.Q_b1[t])**2
model.Constraint_H_b1 = pyo.Constraint(model.t, rule=Constraint_H_b1)

def Constraint_Ph_b1(m, t): 
	return m.Ph_b1[t] == 9810.0 * m.Q_b1[t] * m.H_b1[t]
model.Constraint_Ph_b1 = pyo.Constraint(model.t, rule=Constraint_Ph_b1)

def Constraint_Pe_b1(m, t): 
	return m.Pe_b1[t]*m.nu_b1[t] == m.Ph_b1[t]
model.Constraint_Pe_b1 = pyo.Constraint(model.t, rule=Constraint_Pe_b1)

def Constraint_nu_b1(m, t): 
	return m.nu_b1[t] == 1.0 - (1.0 - 0.85)*((1480/m.rpm_1[t])**0.1)
model.Constraint_nu_b1 = pyo.Constraint(model.t, rule=Constraint_nu_b1)

def Constraint_H_b2(m, t): 
	return m.H_b2[t] == ((m.rpm_2[t]/1480)**2) * 200 - 259.20000000000005*(m.Q_b2[t])**2
model.Constraint_H_b2 = pyo.Constraint(model.t, rule=Constraint_H_b2)

def Constraint_Ph_b2(m, t): 
	return m.Ph_b2[t] == 9810.0 * m.Q_b2[t] * m.H_b2[t]
model.Constraint_Ph_b2 = pyo.Constraint(model.t, rule=Constraint_Ph_b2)

def Constraint_Pe_b2(m, t): 
	return m.Pe_b2[t]*m.nu_b2[t] == m.Ph_b2[t]
model.Constraint_Pe_b2 = pyo.Constraint(model.t, rule=Constraint_Pe_b2)

def Constraint_nu_b2(m, t): 
	return m.nu_b2[t] == 1.0 - (1.0 - 0.85)*((1480/m.rpm_2[t])**0.1)
model.Constraint_nu_b2 = pyo.Constraint(model.t, rule=Constraint_nu_b2)

def Constraint_H_p0(m, t): 
	return m.H_p0[t] == m.z_r1[t]-m.z_r0[t] + 0.005*m.Q_p0[t]**2
model.Constraint_H_p0 = pyo.Constraint(model.t, rule=Constraint_H_p0)

def Constraint_H_p0_H_b0(m, t): 
	return m.H_p0[t] == m.H_b0[t]
model.Constraint_H_p0_H_b0 = pyo.Constraint(model.t, rule=Constraint_H_p0_H_b0)

def Constraint_H_p0_H_b1(m, t): 
	return m.H_p0[t] == m.H_b1[t]
model.Constraint_H_p0_H_b1 = pyo.Constraint(model.t, rule=Constraint_H_p0_H_b1)

def Constraint_H_p0_H_b2(m, t): 
	return m.H_p0[t] == m.H_b2[t]
model.Constraint_H_p0_H_b2 = pyo.Constraint(model.t, rule=Constraint_H_p0_H_b2)


def prova(m,t):
    a = m.Q_b0[t]
    a += m.Q_b1[t]
    a += m.Q_b2[t]
    return a

def Constraint_Q_p0(m, t): 
    list_test = [m.Q_b0, m.Q_b1, m.Q_b2]
    # return m.Q_p0[t] == sum(x[t] for x in list_test)
    return m.Q_p0[t] == prova(m,t)
model.Constraint_Q_p0 = pyo.Constraint(model.t, rule=Constraint_Q_p0)

def Constraint_P_pv_g0(m, t):
	return m.P_pv_g0[t] <= (523000.0 + m.P_pv_dim0)*m.forecast_pv0[t]
model.Constraint_P_pv_g0 = pyo.Constraint(model.t, rule=Constraint_P_pv_g0)

def Constraint_P_pv_dim0(m):
	return m.P_pv_dim0 <= 0
model.Constraint_P_pv_dim0 = pyo.Constraint(rule=Constraint_P_pv_dim0)


def obj_fun(m):
	return sum((m.P_g0[t])*m.cost_elec[t] for t in l_t) + ( + m.P_pv_dim0*m.cost_pv_inst[0])
model.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)


instance = model.create_instance()
solver = pyo.SolverFactory('mindtpy')
solver.solve(instance, mip_solver='glpk', nlp_solver='ipopt', tee=True)
results = {}
results['W_r0'] = list(instance.W_r0.get_values().values())
results['z_r0'] = list(instance.z_r0.get_values().values())
results['W_r1'] = list(instance.W_r1.get_values().values())
results['z_r1'] = list(instance.z_r1.get_values().values())
results['P_g0'] = list(instance.P_g0.get_values().values())
results['Q_p0'] = list(instance.Q_p0.get_values().values())
results['H_p0'] = list(instance.H_p0.get_values().values())
results['Ph_b0'] = list(instance.Ph_b0.get_values().values())
results['Q_b0'] = list(instance.Q_b0.get_values().values())
results['H_b0'] = list(instance.H_b0.get_values().values())
results['rpm_0'] = list(instance.rpm_0.get_values().values())
results['Pe_b0'] = list(instance.Pe_b0.get_values().values())
results['nu_b0'] = list(instance.nu_b0.get_values().values())
results['Ph_b1'] = list(instance.Ph_b1.get_values().values())
results['Q_b1'] = list(instance.Q_b1.get_values().values())
results['H_b1'] = list(instance.H_b1.get_values().values())
results['rpm_1'] = list(instance.rpm_1.get_values().values())
results['Pe_b1'] = list(instance.Pe_b1.get_values().values())
results['nu_b1'] = list(instance.nu_b1.get_values().values())
results['Ph_b2'] = list(instance.Ph_b2.get_values().values())
results['Q_b2'] = list(instance.Q_b2.get_values().values())
results['H_b2'] = list(instance.H_b2.get_values().values())
results['rpm_2'] = list(instance.rpm_2.get_values().values())
results['Pe_b2'] = list(instance.Pe_b2.get_values().values())
results['nu_b2'] = list(instance.nu_b2.get_values().values())
results['P_pv_g0'] = list(instance.P_pv_g0.get_values().values())
results['P_pv_dim0'] = list(instance.P_pv_dim0.get_values().values())
with open('results.json', 'w') as jfile:
	json.dump(results, jfile)
print('Done writing results into .json file')
