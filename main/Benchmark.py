"""
    BENCHMARK FILE FOR TESTING
"""

import logging

import os
import time

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import pyomo
import pyomo.environ as pyo
from pyomo.network import Arc, Port
from pyomo.util.infeasible import log_infeasible_constraints
from pyomo.contrib.gdpopt.enumerate import GDP_Enumeration_Solver
from pyomo.environ import value

# Import useful functions
from Utilities import clear_clc, model_to_file
from Utilities import get_results, get_n_variables

df_full = pd.DataFrame(columns=['test','T_OSMSES','O_OSMSES','T_ISGTb','O_ISGTb','T_ISGT1','O_ISGT1',
                                'T_ISGT2','O_ISGT2','T_ISGT3','O_ISGT3'])
test_name = 'B-Ecp'
test_file_name = 'test/df_full_asd.csv'

for i in range(10):
    print(i)
    df_full.loc[-1,'test'] = test_name
    
    %runcell -n FUNCTIONS
    %runcell -n OSMSES
    %runcell -n ISGTb
    %runcell -n ISGT1
    %runcell -n ISGT2
    %runcell -n ISGT3
    
    df_full.reset_index(inplace=True,drop=True)
    df_full.to_csv(test_file_name, index=False)
    

#%% FUNCTIONS

df_benchmark = pd.DataFrame(columns=['Case','Goal','Time','BaseGoal','BaseTime','Nvariables','BaseNvariables'])

def solve(m, solver='couenne'):

    instance = m.create_instance()

    start_time = time.time()

    if 'neos' in solver:
        os.environ['NEOS_EMAIL'] = 'sergi.costa.dilme@upc.edu'
        solver_manager = pyo.SolverManagerFactory('neos')
        
    if solver=='couenne':
        with open("couenne.opt", "w") as file:
            file.write('''time_limit 100000
                        convexification_cuts 1
                        convexification_points 2
                        use_quadratic no
                        feas_tolerance 1e-1
                        ''')
        solver = pyo.SolverFactory('asl:couenne')
        results = solver.solve(instance, tee=True)
        results.write()
        os.remove('couenne.opt') #Delete options

    # elif solver=='bonmin':
    #     solver = pyo.SolverFactory('bonmin')
    #     results = solver.solve(instance, tee=True)

    # B-BB, B-OA, B-QG, B-Hyb, B-Ecp, B-iFP
    elif solver=='bonmin':
        with open("bonmin.opt", "w") as file:
            file.write('''bonmin.algorithm B-Ecp
                       bonmin.ecp_abs_tol 0.0001
                       bonmin.warm_start optimum
                       tol 0.0001
                        ''')
        solver = pyo.SolverFactory('bonmin')
        results = solver.solve(instance, tee=True)

    elif solver=='ipopt':
        solver = pyo.SolverFactory('ipopt')
        solver.options['max_iter'] = 1000000
        results = solver.solve(instance, tee=True)

    elif solver=='neos_couenne':
        results = solver_manager.solve(instance, solver="couenne")
        
    elif solver=='neos_knitro':
        results = solver_manager.solve(instance, solver="knitro")

    exec_time = time.time() - start_time
    
    return instance, results, exec_time

#%% OSMSES 
# ACADEMIC CASE (OSMSES - 2024 CONFERENCE PAPER) https://doi.org/10.1109/OSMSES62085.2024.10668997

from Devices.Reservoirs import Reservoir
from Devices.Sources import Source
from Devices.Pipes import Pipe
from Devices.Pumps import Pump, RealPump, LinealizedPump
from Devices.Turbines import Turbine
from Devices.EB import EB
from Devices.SolarPV import SolarPV
from Devices.MainGrid import Grid
from Devices.Batteries import Battery, NewBattery

# model
m = pyo.ConcreteModel()


# time
T = 5
l_t = list(range(T))
m.t = pyo.Set(initialize=l_t)

# electricity cost
m.l_cost = [0.15,0.20,0.05,0.05,0.20]
m.cost = pyo.Param(m.t, initialize=m.l_cost)
m.cost_pv_P = 0.00126
m.cost_bat_P = 0.00171
m.cost_bat_E = 0.00856
m.cost_turb_P = 0.00143

# ===== Create the system =====
m.Reservoir1 = pyo.Block()
m.Reservoir0 = pyo.Block()
m.Irrigation1 = pyo.Block()
m.Pump1 = pyo.Block()
m.Pump2 = pyo.Block()
m.Turb1 = pyo.Block()
m.Pipe1 = pyo.Block()
m.PV = pyo.Block()
m.Grid = pyo.Block()
m.EB = pyo.Block()
m.Battery = pyo.Block()


m.data_irr = {'Q':[0.1,0.8,0.8,0.8,0.1]} # irrigation
Source(m.Irrigation1, m.t, m.data_irr, {})

m.data_res0 = {'dt':1, 'W0':5, 'Wmin':0, 'Wmax':10, 'zmin':0, 'zmax':0.05}
m.data_res1 = {'dt':1, 'W0':1, 'Wmin':0, 'Wmax':10, 'zmin':0.75, 'zmax':0.80, 'WT_min':0.9, 'WT_max':1.10}
m.init_res = {'Q':[0]*T, 'W':[0.5]*T}

Reservoir(m.Reservoir1, m.t, m.data_res1, m.init_res)
Reservoir(m.Reservoir0, m.t, m.data_res0, m.init_res)

m.data_c1 = {'K':0.01, 'Qmax':5} # canal
m.init_c1 = {'Q':[0]*T, 'H':[1]*T, 'H0':[1]*T, 'zlow':[0.01]*T, 'zhigh':[0.75]*T}
Pipe(m.Pipe1, m.t, m.data_c1, m.init_c1)

m.data_p = {'A':1, 'B':0.1, 'nmax':1, 'eff':0.9, 'Qnom':1, 'Pmax':9810*1*1, 'Qmin':0.5, 'Qmax':2, 'eff_t':0.5} # pumps (both equal)
m.init_p = {'Q':[0]*T, 'H':[1]*T, 'n':[1]*T, 'Pe':[9810*1*1]*T}
RealPump(m.Pump1, m.t, m.data_p, m.init_p)
RealPump(m.Pump2, m.t, m.data_p, m.init_p)

m.data_t = {'eff':0.85, 'Pmax':2*9810}
m.init_t = {'Q':[0]*T, 'H':[1]*T, 'Pe':[-9810]*T}
Turbine(m.Turb1, m.t, m.data_t, m.init_t)

m.data_pv = {'Pinst':9810, 'Pmax':9810*2, 'forecast':[0,0.1,0.6,1.2,0.2], 'eff': 0.98} # PV
SolarPV(m.PV, m.t, m.data_pv)

Grid(m.Grid, m.t, {'Pmax':9810e3}) # grid

EB(m.EB, m.t)

#Battery data
m.data = {'dt':1,'SOCmax':1,
        'SOCmin':0.2,'Pmax':9810*2,
        'Emax':9810*2,'eff_ch':0.9,
        'eff_dc':0.9}

m.init_data = {'E':[0]*T,'P':0}
NewBattery(m.Battery, m.t, m.data, m.init_data)

# Connections
m.p1r0 = Arc(ports=(m.Pump1.port_Qin, m.Reservoir0.port_Q), directed=True)
m.p1c1_Q = Arc(ports=(m.Pump1.port_Qout, m.Pipe1.port_Q), directed=True)
m.p1c1_H = Arc(ports=(m.Pump1.port_H, m.Pipe1.port_H), directed=True)
m.ebp1 = Arc(ports=(m.Pump1.port_P, m.EB.port_P), directed=True)

m.p2r0 = Arc(ports=(m.Pump2.port_Qin, m.Reservoir0.port_Q), directed=True)
m.p2c1_Q = Arc(ports=(m.Pump2.port_Qout, m.Pipe1.port_Q), directed=True)
m.p2c1_H = Arc(ports=(m.Pump2.port_H, m.Pipe1.port_H), directed=True)
m.ebp2 = Arc(ports=(m.Pump2.port_P, m.EB.port_P), directed=True)

m.t1r0 = Arc(ports=(m.Turb1.port_Qout, m.Reservoir0.port_Q), directed=True)
m.t1c1_Q = Arc(ports=(m.Turb1.port_Qin, m.Pipe1.port_Q), directed=True)
m.t1c1_H = Arc(ports=(m.Turb1.port_H, m.Pipe1.port_H), directed=True)
m.t1eb = Arc(ports=(m.Turb1.port_P, m.EB.port_P), directed=True)

m.c1r1 = Arc(ports=(m.Pipe1.port_Q, m.Reservoir1.port_Q), directed=True)
m.c1r1_z = Arc(ports=(m.Reservoir1.port_z, m.Pipe1.port_zhigh), directed=True)
m.c1r0_z = Arc(ports=(m.Reservoir0.port_z, m.Pipe1.port_zlow), directed=True)

m.r1i1 = Arc(ports=(m.Irrigation1.port_Qin, m.Reservoir1.port_Q), directed=True)

m.grideb = Arc(ports=(m.Grid.port_P, m.EB.port_P), directed=True)
m.pveb = Arc(ports=(m.PV.port_P, m.EB.port_P), directed=True)
m.bat_eb = Arc(ports=(m.Battery.port_P, m.EB.port_P), directed=True)

pyo.TransformationFactory("network.expand_arcs").apply_to(m) # apply arcs to model


# Objective function
def obj_fun(m):
	return sum((m.Grid.Pbuy[t]*m.cost[t] - m.Grid.Psell[t]*m.cost[t]/2) for t in l_t ) + m.PV.Pdim*m.cost_pv_P + m.Turb1.Pdim*m.cost_turb_P + m.Battery.Pdim*m.cost_bat_P + m.Battery.Edim*m.cost_bat_E
m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)

instance, results, exec_time = solve(m,'bonmin')

df_benchmark.loc[-1] = ['OSMSES',value(instance.goal),exec_time,-1918.0601869271836,16.785921096801758,get_n_variables(m),289]
df_benchmark.reset_index(inplace=True,drop=True)
df_benchmark.loc[-1] = ['OSMSES_2024Nov',value(instance.goal),exec_time,-1918.0601869271836,5.869357049,get_n_variables(m),269]
df_benchmark.reset_index(inplace=True,drop=True)

df_full.loc[-1,'T_OSMSES'] = exec_time
df_full.loc[-1,'O_OSMSES'] = value(instance.goal)

file = './test/Benchmark/OSMSES'
df_out, df_param, df_size = get_results(file=file, instance=instance, results=results, l_t=l_t, exec_time=exec_time)

del m, instance, T, l_t, df_out, df_param, df_size, results, file, exec_time

#%% ISGTb
# LES PLANES CASE (ISGT - 2024 CONFERENCE PAPER -- Base case) 

from Devices.Reservoirs import Reservoir
from Devices.Sources import Source
from Devices.Pipes import Pipe
from Devices.Pumps import Pump, RealPump, LinealizedPump
from Devices.Turbines import Turbine
from Devices.EB import EB
from Devices.SolarPV import SolarPV
from Devices.MainGrid import Grid
from Devices.Batteries import Battery, NewBattery
from pyomo.contrib.gdpopt.enumerate import GDP_Enumeration_Solver
from pyomo.environ import value

# model
m = pyo.ConcreteModel()

df_meteo_aug = pd.read_csv('data/meteo/LesPlanes_meteo_hour_aug.csv')
df_cons_aug = pd.read_csv('data/irrigation/LesPlanes_irrigation_aug.csv')
df_grid_aug = pd.read_csv('data/costs/PVPC_aug.csv')

df_meteo_jan = pd.read_csv('data/meteo/LesPlanes_meteo_hour_jan.csv')
df_cons_jan = pd.read_csv('data/irrigation/LesPlanes_irrigation_jan.csv')
df_grid_jan = pd.read_csv('data/costs/PVPC_jan.csv')

df_grid_jan['Excedentes_cut'] = df_grid_jan['Excedentes']*(1-0.3*df_grid_jan['Hour'].apply(lambda x: 1 if (x in [8,9,10,11,12,13,14,15,16]) else 0))
df_grid_jan['Excedentes'] = df_grid_jan['Excedentes_cut']
df_grid_aug['Excedentes_cut'] = df_grid_aug['Excedentes']*(1-0.3*df_grid_aug['Hour'].apply(lambda x: 1 if (x in [8,9,10,11,12,13,14,15,16]) else 0))
df_grid_aug['Excedentes'] = df_grid_aug['Excedentes_cut']

# time
T = 24
l_t = list(range(T))
m.tw = pyo.Set(initialize=l_t)
m.ts = pyo.Set(initialize=l_t)

# electricity cost
l_costw = df_grid_jan['PVPC']
l_excw = df_grid_jan['Excedentes']
m.costw = pyo.Param(m.tw, initialize=l_costw)
m.excw = pyo.Param(m.tw, initialize=l_excw)

l_costs = df_grid_aug['PVPC']
l_excs = df_grid_aug['Excedentes']
m.costs = pyo.Param(m.ts, initialize=l_costs)
m.excs = pyo.Param(m.ts, initialize=l_excs)

cost_new_pv = 0.00126*T/1000
cp_bat = 0.00171*T/1000
ce_bat = 0.00856*T/1000
# Costos en €/(kwh·h). Considerant vida de 20 anys, del llibre de Oriol Gomis i Paco Diaz trobem:
#   - Ce = 750 €/kWh (10 anys de vida) -> x2 = 1500 €/kWh -> /20/365/24 = 0.00856 €/(kWh·h)

# ===== Create the system =====
m.ReservoirEbrew = pyo.Block()
m.Reservoir1w = pyo.Block()
m.Irrigation1w = pyo.Block()
m.Pump1w = pyo.Block()
m.Pump2w = pyo.Block()
# m.Turb1w = pyo.Block()
m.Pipe1w = pyo.Block()
m.PVw = pyo.Block()
m.Gridw = pyo.Block()
m.EBgw = pyo.Block()
m.EBpvw = pyo.Block()
m.Batw = pyo.Block()

m.ReservoirEbres = pyo.Block()
m.Reservoir1s = pyo.Block()
m.Irrigation1s = pyo.Block()
m.Pump1s = pyo.Block()
m.Pump2s = pyo.Block()
# m.Turb1s = pyo.Block()
m.Pipe1s = pyo.Block()
m.PVs = pyo.Block()
m.Grids = pyo.Block()
m.EBgs = pyo.Block()
m.EBpvs = pyo.Block()
m.Bats = pyo.Block()


data_irr = {'Q':df_cons_jan['Qirr']/3600*1.4} # irrigation
Source(m.Irrigation1w, m.tw, data_irr, {})
data_irr = {'Q':df_cons_aug['Qirr']/3600*1.4} # irrigation
Source(m.Irrigation1s, m.ts, data_irr, {})

data_Ebre = {'dt':3600, 'W0':2e5, 'Wmin':0, 'Wmax':3e5, 'zmin':29.5, 'zmax':30}
init_Ebre = {'Q':[0]*T, 'W':[2e5]*T}
Reservoir(m.ReservoirEbrew, m.tw, data_Ebre, init_Ebre)
Reservoir(m.ReservoirEbres, m.ts, data_Ebre, init_Ebre)
data_R1 = {'dt':3600, 'W0':10e3, 'Wmin':9e3, 'Wmax':13e3, 'zmin':135+(141-135)*9/13, 'zmax':141, 'WT_min':0.95*10e3, 'WT_max':1.05*10e3}
init_R1 = {'Q':[0]*T, 'W':[10e3]*T}
Reservoir(m.Reservoir1w, m.tw, data_R1, init_R1)
Reservoir(m.Reservoir1s, m.ts, data_R1, init_R1)

data_c1 = {'K':297.38*0.20, 'Qmax':1.2} # canal
init_c1 = {'Q':[0]*T, 'H':[108]*T, 'H0':[108]*T, 'zlow':[30]*T, 'zhigh':[138]*T}
Pipe(m.Pipe1w, m.tw, data_c1, init_c1)
Pipe(m.Pipe1s, m.ts, data_c1, init_c1)

data_p = {'A':121.54, 'B':3864.8, 'n_n':2900, 'nmax':1, 'eff':0.8, 'eff_t':0.5, 'S':0.1*0.1*3.14, 'Qmin':0.6250, 'Qmax':1.8688, 'Qnom':0.0556, 'Pmax':110e3, 'intervals':3} # pumps (both equal)
init_p = {'Q':[0]*T, 'H':[109]*T, 'n':[0.99]*T, 'Pe':[0]*T}
RealPump(m.Pump1w, m.tw, data_p, init_p)
RealPump(m.Pump1s, m.ts, data_p, init_p)
data_p['eff']=0.8
RealPump(m.Pump2w, m.tw, data_p, init_p)
RealPump(m.Pump2s, m.ts, data_p, init_p)

# data_pdouble = {'A':121.54, 'B':3864.8/(2**2), 'n_n':2900, 'eff':0.8, 'eff_t':0.5, 'S':0.1*0.1*3.14, 'Qmin':0.6250, 'Qmax':1.8688*2, 'Qnom':0.0556, 'Pmax':2*110e3} # pumps (both equal)
# init_pdouble = {'Q':[0]*T, 'H':[108]*T, 'n':[1]*T, 'Pe':[110e3*0.9]*T}
# RealPump(m.Pump1w, m.tw, data_pdouble, init_pdouble)
# RealPump(m.Pump1s, m.ts, data_pdouble, init_pdouble)

# data_t = {'eff':0.5, 'Pmax':110e3}
# init_t = {'Q':[0]*T, 'H':[109]*T, 'Pe':[0]*T}
# Turbine(m.Turb1w, m.tw, data_t, init_t)
# Turbine(m.Turb1s, m.ts, data_t, init_t)

data_pv = {'Pinst':215.28e3, 'Pmax':215.28e3, 'forecast':df_meteo_jan['Irr']/1000, 'eff':0.98} # PV
SolarPV(m.PVw, m.tw, data_pv)
data_pv = {'Pinst':215.28e3, 'Pmax':215.28e3, 'forecast':df_meteo_aug['Irr']/1000, 'eff':0.98} # PV
SolarPV(m.PVs, m.ts, data_pv)

data_bat = {'dt':3600, 'E0':0.05, 'Emax':200e3, 'Pmax':200e3, 'SOCmin':0.2, 'SOCmax':1.0, 'eff_ch':0.8, 'eff_dc':0.8,'Einst':0.1, 'Pinst':0}
init_bat = {'E':[0]*T, 'P':[0]*T}
NewBattery(m.Batw, m.tw, data_bat, init_bat)
NewBattery(m.Bats, m.ts, data_bat, init_bat)

Grid(m.Gridw, m.tw, {'Pmax':100e6}) # grid
Grid(m.Grids, m.ts, {'Pmax':100e6}) # grid

EB(m.EBgw, m.tw)
EB(m.EBgs, m.ts)
EB(m.EBpvw, m.tw)
EB(m.EBpvs, m.ts)


# def ConstraintPumpTurbw(m, t):
#     return m.Turb1w.Qin[t] * (m.Pump1w.PumpOn[t] + m.Pump2w.PumpOn[t]) == 0
# m.Turb1_c_PumpTurbw = pyo.Constraint(m.tw, rule=ConstraintPumpTurbw)
# def ConstraintPumpTurbs(m, t):
#     return m.Turb1s.Qin[t] * (m.Pump1s.PumpOn[t] + m.Pump2s.PumpOn[t]) == 0
# m.Turb1_c_PumpTurbs = pyo.Constraint(m.ts, rule=ConstraintPumpTurbs)

# def ConstraintPump2w(m, t):
#     return m.Pump2w.Qin[t] * m.Pump1w.PumpOn[t] == 0
# m.Turb1_c_Pump2w = pyo.Constraint(m.tw, rule=ConstraintPump2w)
# def ConstraintPump2s(m, t):
#     return m.Pump2s.Qin[t] * m.Pump1s.PumpOn[t] == 0
# m.Turb1_c_Pump2s = pyo.Constraint(m.ts, rule=ConstraintPump2s)


def ConstraintBatEws(m):
    return m.Batw.Edim == m.Bats.Edim
m.c_BatEws = pyo.Constraint(rule=ConstraintBatEws)
def ConstraintBatPws(m):
    return m.Batw.Pdim == m.Bats.Pdim
m.c_BatPws = pyo.Constraint(rule=ConstraintBatPws)


# m.Batw.Edim.fix(0)
# m.Bats.Edim.fix(0)
# m.Batw.Pdim.fix(0)
# m.Bats.Pdim.fix(0)

# m.Turb1s.Pdim.fix(110e3)
# m.Turb1w.Pdim.fix(110e3)


# Connections
m.p1r0w = Arc(ports=(m.Pump1w.port_Qin, m.ReservoirEbrew.port_Q), directed=True)
m.p1c1_Qw = Arc(ports=(m.Pump1w.port_Qout, m.Pipe1w.port_Q), directed=True)
m.p1c1_Hw = Arc(ports=(m.Pump1w.port_H, m.Pipe1w.port_H), directed=True)
m.p1ebw = Arc(ports=(m.Pump1w.port_P, m.EBgw.port_P), directed=True)

m.p2r0w = Arc(ports=(m.Pump2w.port_Qin, m.ReservoirEbrew.port_Q), directed=True)
m.p2c1_Qw = Arc(ports=(m.Pump2w.port_Qout, m.Pipe1w.port_Q), directed=True)
m.p2c1_Hw = Arc(ports=(m.Pump2w.port_H, m.Pipe1w.port_H), directed=True)
m.p2ebw = Arc(ports=(m.Pump2w.port_P, m.EBpvw.port_P), directed=True) # pv node

# m.t1r0w = Arc(ports=(m.Turb1w.port_Qout, m.ReservoirEbrew.port_Q), directed=True)
# m.t1c1_Qw = Arc(ports=(m.Turb1w.port_Qin, m.Pipe1w.port_Q), directed=True)
# m.t1c1_Hw = Arc(ports=(m.Turb1w.port_H, m.Pipe1w.port_H), directed=True)
# m.t1ebw = Arc(ports=(m.Turb1w.port_P, m.EBgw.port_P), directed=True)

m.c1r1_Qw = Arc(ports=(m.Pipe1w.port_Q, m.Reservoir1w.port_Q), directed=True)
m.c1r1_zw = Arc(ports=(m.Reservoir1w.port_z, m.Pipe1w.port_zhigh), directed=True)
m.c1r0_zw = Arc(ports=(m.ReservoirEbrew.port_z, m.Pipe1w.port_zlow), directed=True)

m.r1i1w = Arc(ports=(m.Irrigation1w.port_Qin, m.Reservoir1w.port_Q), directed=True)

m.gridebw = Arc(ports=(m.Gridw.port_P, m.EBgw.port_P), directed=True)
m.pvebw = Arc(ports=(m.PVw.port_P, m.EBpvw.port_P), directed=True) # pv node
m.batebw = Arc(ports=(m.Batw.port_P, m.EBpvw.port_P), directed=True) # pv node

# Connections
m.p1r0s = Arc(ports=(m.Pump1s.port_Qin, m.ReservoirEbres.port_Q), directed=True)
m.p1c1_Qs = Arc(ports=(m.Pump1s.port_Qout, m.Pipe1s.port_Q), directed=True)
m.p1c1_Hs = Arc(ports=(m.Pump1s.port_H, m.Pipe1s.port_H), directed=True)
m.p1ebs = Arc(ports=(m.Pump1s.port_P, m.EBgs.port_P), directed=True)

m.p2r0s = Arc(ports=(m.Pump2s.port_Qin, m.ReservoirEbres.port_Q), directed=True)
m.p2c1_Qs = Arc(ports=(m.Pump2s.port_Qout, m.Pipe1s.port_Q), directed=True)
m.p2c1_Hs = Arc(ports=(m.Pump2s.port_H, m.Pipe1s.port_H), directed=True)
m.p2ebs = Arc(ports=(m.Pump2s.port_P, m.EBpvs.port_P), directed=True) # pv node

# m.t1r0s = Arc(ports=(m.Turb1s.port_Qout, m.ReservoirEbres.port_Q), directed=True)
# m.t1c1_Qs = Arc(ports=(m.Turb1s.port_Qin, m.Pipe1s.port_Q), directed=True)
# m.t1c1_Hs = Arc(ports=(m.Turb1s.port_H, m.Pipe1s.port_H), directed=True)
# m.t1ebs = Arc(ports=(m.Turb1s.port_P, m.EBgs.port_P), directed=True)

m.c1r1_Qs = Arc(ports=(m.Pipe1s.port_Q, m.Reservoir1s.port_Q), directed=True)
m.c1r1_zs = Arc(ports=(m.Reservoir1s.port_z, m.Pipe1s.port_zhigh), directed=True)
m.c1r0_zs = Arc(ports=(m.ReservoirEbres.port_z, m.Pipe1s.port_zlow), directed=True)

m.r1i1s = Arc(ports=(m.Irrigation1s.port_Qin, m.Reservoir1s.port_Q), directed=True)

m.gridebs = Arc(ports=(m.Grids.port_P, m.EBgs.port_P), directed=True)
m.pvebs = Arc(ports=(m.PVs.port_P, m.EBpvs.port_P), directed=True) # pv node
m.batebs = Arc(ports=(m.Bats.port_P, m.EBpvs.port_P), directed=True) # pv node

pyo.TransformationFactory("network.expand_arcs").apply_to(m) # apply arcs to model


# Objective function
def obj_fun(m):
# 	return sum((m.Grid.Pbuy[t]*m.cost[t]/1e6 - m.Grid.Psell[t]*m.exc[t]/1e6) + 0*1/1e6*((m.PV.Pinst+m.PV.Pdim)*m.PV.forecast[t]*m.PV.eff + m.PV.P[t]) for t in l_t ) #+ (m.Bat.Pdim*cp_bat + m.Bat.Edim*ce_bat)/365/20#+ m.PV.Pdim*cost_new_pv
	return sum(( m.Gridw.Pbuy[t]*m.costw[t]/1e6 - m.Gridw.Psell[t]*m.excw[t]/1e6 + 
             m.Grids.Pbuy[t]*m.costs[t]/1e6 - m.Grids.Psell[t]*m.excs[t]/1e6)/2  for t in l_t ) + (m.Batw.Pdim*cp_bat + m.Batw.Edim*ce_bat) #+ m.PV.Pdim*cost_new_pv
m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)


instance, results, exec_time = solve(m,'bonmin')

df_benchmark.loc[-1] = ['ISGTbase',value(instance.goal),exec_time,24.963233294640837,508.1471881866455,get_n_variables(m),2166]
df_benchmark.reset_index(inplace=True,drop=True)


df_full.loc[-1,'T_ISGTb'] = exec_time
df_full.loc[-1,'O_ISGTb'] = value(instance.goal)

file = './test/Benchmark/ISGTbase'
df_out, df_param, df_size = get_results(file=file, instance=instance, results=results, l_t=l_t, exec_time=exec_time)

del df_cons_aug, df_cons_jan, df_grid_aug, df_grid_jan, df_meteo_aug, df_meteo_jan
del ce_bat, cost_new_pv, cp_bat, data_bat, data_c1, data_Ebre, data_irr, data_p, data_pv, data_R1
del init_bat, init_c1, init_Ebre, init_p, init_R1
del l_costs, l_costw, l_excs, l_excw
del m, instance, T, l_t, df_out, df_param, df_size, results, file, exec_time

#%% ISGT1
# LES PLANES CASE (ISGT - 2024 CONFERENCE PAPER -- Case 1) 

from Devices.Reservoirs import Reservoir
from Devices.Sources import Source
from Devices.Pipes import Pipe
from Devices.Pumps import Pump, RealPump, LinealizedPump
from Devices.Turbines import Turbine
from Devices.EB import EB
from Devices.SolarPV import SolarPV
from Devices.MainGrid import Grid
from Devices.Batteries import Battery, NewBattery
from pyomo.contrib.gdpopt.enumerate import GDP_Enumeration_Solver
from pyomo.environ import value

# model
m = pyo.ConcreteModel()

df_meteo_aug = pd.read_csv('data/meteo/LesPlanes_meteo_hour_aug.csv')
df_cons_aug = pd.read_csv('data/irrigation/LesPlanes_irrigation_aug.csv')
df_grid_aug = pd.read_csv('data/costs/PVPC_aug.csv')

df_meteo_jan = pd.read_csv('data/meteo/LesPlanes_meteo_hour_jan.csv')
df_cons_jan = pd.read_csv('data/irrigation/LesPlanes_irrigation_jan.csv')
df_grid_jan = pd.read_csv('data/costs/PVPC_jan.csv')

df_grid_jan['Excedentes_cut'] = df_grid_jan['Excedentes']*(1-0.3*df_grid_jan['Hour'].apply(lambda x: 1 if (x in [8,9,10,11,12,13,14,15,16]) else 0))
df_grid_jan['Excedentes'] = df_grid_jan['Excedentes_cut']
df_grid_aug['Excedentes_cut'] = df_grid_aug['Excedentes']*(1-0.3*df_grid_aug['Hour'].apply(lambda x: 1 if (x in [8,9,10,11,12,13,14,15,16]) else 0))
df_grid_aug['Excedentes'] = df_grid_aug['Excedentes_cut']

# time
T = 24
l_t = list(range(T))
m.tw = pyo.Set(initialize=l_t)
m.ts = pyo.Set(initialize=l_t)

# electricity cost
l_costw = df_grid_jan['PVPC']
l_excw = df_grid_jan['Excedentes']
m.costw = pyo.Param(m.tw, initialize=l_costw)
m.excw = pyo.Param(m.tw, initialize=l_excw)

l_costs = df_grid_aug['PVPC']
l_excs = df_grid_aug['Excedentes']
m.costs = pyo.Param(m.ts, initialize=l_costs)
m.excs = pyo.Param(m.ts, initialize=l_excs)

cost_new_pv = 0.00126*T/1000
cp_bat = 0.00171*T/1000
ce_bat = 0.00856*T/1000
# Costos en €/(kwh·h). Considerant vida de 20 anys, del llibre de Oriol Gomis i Paco Diaz trobem:
#   - Ce = 750 €/kWh (10 anys de vida) -> x2 = 1500 €/kWh -> /20/365/24 = 0.00856 €/(kWh·h)

# ===== Create the system =====
m.ReservoirEbrew = pyo.Block()
m.Reservoir1w = pyo.Block()
m.Irrigation1w = pyo.Block()
m.Pump1w = pyo.Block()
m.Pump2w = pyo.Block()
m.Turb1w = pyo.Block()
m.Pipe1w = pyo.Block()
m.PVw = pyo.Block()
m.Gridw = pyo.Block()
m.EBgw = pyo.Block()
m.EBpvw = pyo.Block()
m.Batw = pyo.Block()

m.ReservoirEbres = pyo.Block()
m.Reservoir1s = pyo.Block()
m.Irrigation1s = pyo.Block()
m.Pump1s = pyo.Block()
m.Pump2s = pyo.Block()
m.Turb1s = pyo.Block()
m.Pipe1s = pyo.Block()
m.PVs = pyo.Block()
m.Grids = pyo.Block()
m.EBgs = pyo.Block()
m.EBpvs = pyo.Block()
m.Bats = pyo.Block()


data_irr = {'Q':df_cons_jan['Qirr']/3600*1.4} # irrigation
Source(m.Irrigation1w, m.tw, data_irr, {})
data_irr = {'Q':df_cons_aug['Qirr']/3600*1.4} # irrigation
Source(m.Irrigation1s, m.ts, data_irr, {})

data_Ebre = {'dt':3600, 'W0':2e5, 'Wmin':0, 'Wmax':3e5, 'zmin':29.5, 'zmax':30}
init_Ebre = {'Q':[0]*T, 'W':[2e5]*T}
Reservoir(m.ReservoirEbrew, m.tw, data_Ebre, init_Ebre)
Reservoir(m.ReservoirEbres, m.ts, data_Ebre, init_Ebre)
data_R1 = {'dt':3600, 'W0':10e3, 'Wmin':9e3, 'Wmax':13e3, 'zmin':135+(141-135)*9/13, 'zmax':141, 'WT_min':0.95*10e3, 'WT_max':1.05*10e3}
init_R1 = {'Q':[0]*T, 'W':[10e3]*T}
Reservoir(m.Reservoir1w, m.tw, data_R1, init_R1)
Reservoir(m.Reservoir1s, m.ts, data_R1, init_R1)

data_c1 = {'K':297.38*0.20, 'Qmax':1.2} # canal
init_c1 = {'Q':[0]*T, 'H':[108]*T, 'H0':[108]*T, 'zlow':[30]*T, 'zhigh':[138]*T}
Pipe(m.Pipe1w, m.tw, data_c1, init_c1)
Pipe(m.Pipe1s, m.ts, data_c1, init_c1)

data_p = {'A':121.54, 'B':3864.8, 'n_n':2900, 'nmax':1, 'eff':0.8, 'eff_t':0.5, 'S':0.1*0.1*3.14, 'Qmin':0.6250, 'Qmax':1.8688, 'Qnom':0.0556, 'Pmax':110e3, 'intervals':3} # pumps (both equal)
init_p = {'Q':[0]*T, 'H':[109]*T, 'n':[0.99]*T, 'Pe':[0]*T}
RealPump(m.Pump1w, m.tw, data_p, init_p)
RealPump(m.Pump1s, m.ts, data_p, init_p)
data_p['eff']=0.8
RealPump(m.Pump2w, m.tw, data_p, init_p)
RealPump(m.Pump2s, m.ts, data_p, init_p)

# data_pdouble = {'A':121.54, 'B':3864.8/(2**2), 'n_n':2900, 'eff':0.8, 'eff_t':0.5, 'S':0.1*0.1*3.14, 'Qmin':0.6250, 'Qmax':1.8688*2, 'Qnom':0.0556, 'Pmax':2*110e3} # pumps (both equal)
# init_pdouble = {'Q':[0]*T, 'H':[108]*T, 'n':[1]*T, 'Pe':[110e3*0.9]*T}
# RealPump(m.Pump1w, m.tw, data_pdouble, init_pdouble)
# RealPump(m.Pump1s, m.ts, data_pdouble, init_pdouble)

data_t = {'eff':0.5, 'Pmax':110e3}
init_t = {'Q':[0]*T, 'H':[109]*T, 'Pe':[0]*T}
Turbine(m.Turb1w, m.tw, data_t, init_t)
Turbine(m.Turb1s, m.ts, data_t, init_t)

data_pv = {'Pinst':215.28e3, 'Pmax':215.28e3, 'forecast':df_meteo_jan['Irr']/1000, 'eff':0.98} # PV
SolarPV(m.PVw, m.tw, data_pv)
data_pv = {'Pinst':215.28e3, 'Pmax':215.28e3, 'forecast':df_meteo_aug['Irr']/1000, 'eff':0.98} # PV
SolarPV(m.PVs, m.ts, data_pv)

data_bat = {'dt':3600, 'E0':0.05, 'Emax':200e3, 'Pmax':200e3, 'SOCmin':0.2, 'SOCmax':1.0, 'eff_ch':0.8, 'eff_dc':0.8,'Einst':0.1, 'Pinst':0}
init_bat = {'E':[0]*T, 'P':[0]*T}
NewBattery(m.Batw, m.tw, data_bat, init_bat)
NewBattery(m.Bats, m.ts, data_bat, init_bat)

Grid(m.Gridw, m.tw, {'Pmax':100e6}) # grid
Grid(m.Grids, m.ts, {'Pmax':100e6}) # grid

EB(m.EBgw, m.tw)
EB(m.EBgs, m.ts)
EB(m.EBpvw, m.tw)
EB(m.EBpvs, m.ts)


# def ConstraintPumpTurbw(m, t):
#     return m.Turb1w.Qin[t] * (m.Pump1w.PumpOn[t] + m.Pump2w.PumpOn[t]) == 0
# m.Turb1_c_PumpTurbw = pyo.Constraint(m.tw, rule=ConstraintPumpTurbw)
# def ConstraintPumpTurbs(m, t):
#     return m.Turb1s.Qin[t] * (m.Pump1s.PumpOn[t] + m.Pump2s.PumpOn[t]) == 0
# m.Turb1_c_PumpTurbs = pyo.Constraint(m.ts, rule=ConstraintPumpTurbs)

# def ConstraintPump2w(m, t):
#     return m.Pump2w.Qin[t] * m.Pump1w.PumpOn[t] == 0
# m.Turb1_c_Pump2w = pyo.Constraint(m.tw, rule=ConstraintPump2w)
# def ConstraintPump2s(m, t):
#     return m.Pump2s.Qin[t] * m.Pump1s.PumpOn[t] == 0
# m.Turb1_c_Pump2s = pyo.Constraint(m.ts, rule=ConstraintPump2s)


def ConstraintBatEws(m):
    return m.Batw.Edim == m.Bats.Edim
m.c_BatEws = pyo.Constraint(rule=ConstraintBatEws)
def ConstraintBatPws(m):
    return m.Batw.Pdim == m.Bats.Pdim
m.c_BatPws = pyo.Constraint(rule=ConstraintBatPws)


# m.Batw.Edim.fix(0)
# m.Bats.Edim.fix(0)
# m.Batw.Pdim.fix(0)
# m.Bats.Pdim.fix(0)

m.Turb1s.Pdim.fix(110e3)
m.Turb1w.Pdim.fix(110e3)


# Connections
m.p1r0w = Arc(ports=(m.Pump1w.port_Qin, m.ReservoirEbrew.port_Q), directed=True)
m.p1c1_Qw = Arc(ports=(m.Pump1w.port_Qout, m.Pipe1w.port_Q), directed=True)
m.p1c1_Hw = Arc(ports=(m.Pump1w.port_H, m.Pipe1w.port_H), directed=True)
m.p1ebw = Arc(ports=(m.Pump1w.port_P, m.EBgw.port_P), directed=True)

m.p2r0w = Arc(ports=(m.Pump2w.port_Qin, m.ReservoirEbrew.port_Q), directed=True)
m.p2c1_Qw = Arc(ports=(m.Pump2w.port_Qout, m.Pipe1w.port_Q), directed=True)
m.p2c1_Hw = Arc(ports=(m.Pump2w.port_H, m.Pipe1w.port_H), directed=True)
m.p2ebw = Arc(ports=(m.Pump2w.port_P, m.EBpvw.port_P), directed=True) # pv node

m.t1r0w = Arc(ports=(m.Turb1w.port_Qout, m.ReservoirEbrew.port_Q), directed=True)
m.t1c1_Qw = Arc(ports=(m.Turb1w.port_Qin, m.Pipe1w.port_Q), directed=True)
m.t1c1_Hw = Arc(ports=(m.Turb1w.port_H, m.Pipe1w.port_H), directed=True)
m.t1ebw = Arc(ports=(m.Turb1w.port_P, m.EBgw.port_P), directed=True)

m.c1r1_Qw = Arc(ports=(m.Pipe1w.port_Q, m.Reservoir1w.port_Q), directed=True)
m.c1r1_zw = Arc(ports=(m.Reservoir1w.port_z, m.Pipe1w.port_zhigh), directed=True)
m.c1r0_zw = Arc(ports=(m.ReservoirEbrew.port_z, m.Pipe1w.port_zlow), directed=True)

m.r1i1w = Arc(ports=(m.Irrigation1w.port_Qin, m.Reservoir1w.port_Q), directed=True)

m.gridebw = Arc(ports=(m.Gridw.port_P, m.EBgw.port_P), directed=True)
m.pvebw = Arc(ports=(m.PVw.port_P, m.EBpvw.port_P), directed=True) # pv node
m.batebw = Arc(ports=(m.Batw.port_P, m.EBpvw.port_P), directed=True) # pv node

# Connections
m.p1r0s = Arc(ports=(m.Pump1s.port_Qin, m.ReservoirEbres.port_Q), directed=True)
m.p1c1_Qs = Arc(ports=(m.Pump1s.port_Qout, m.Pipe1s.port_Q), directed=True)
m.p1c1_Hs = Arc(ports=(m.Pump1s.port_H, m.Pipe1s.port_H), directed=True)
m.p1ebs = Arc(ports=(m.Pump1s.port_P, m.EBgs.port_P), directed=True)

m.p2r0s = Arc(ports=(m.Pump2s.port_Qin, m.ReservoirEbres.port_Q), directed=True)
m.p2c1_Qs = Arc(ports=(m.Pump2s.port_Qout, m.Pipe1s.port_Q), directed=True)
m.p2c1_Hs = Arc(ports=(m.Pump2s.port_H, m.Pipe1s.port_H), directed=True)
m.p2ebs = Arc(ports=(m.Pump2s.port_P, m.EBpvs.port_P), directed=True) # pv node

m.t1r0s = Arc(ports=(m.Turb1s.port_Qout, m.ReservoirEbres.port_Q), directed=True)
m.t1c1_Qs = Arc(ports=(m.Turb1s.port_Qin, m.Pipe1s.port_Q), directed=True)
m.t1c1_Hs = Arc(ports=(m.Turb1s.port_H, m.Pipe1s.port_H), directed=True)
m.t1ebs = Arc(ports=(m.Turb1s.port_P, m.EBgs.port_P), directed=True)

m.c1r1_Qs = Arc(ports=(m.Pipe1s.port_Q, m.Reservoir1s.port_Q), directed=True)
m.c1r1_zs = Arc(ports=(m.Reservoir1s.port_z, m.Pipe1s.port_zhigh), directed=True)
m.c1r0_zs = Arc(ports=(m.ReservoirEbres.port_z, m.Pipe1s.port_zlow), directed=True)

m.r1i1s = Arc(ports=(m.Irrigation1s.port_Qin, m.Reservoir1s.port_Q), directed=True)

m.gridebs = Arc(ports=(m.Grids.port_P, m.EBgs.port_P), directed=True)
m.pvebs = Arc(ports=(m.PVs.port_P, m.EBpvs.port_P), directed=True) # pv node
m.batebs = Arc(ports=(m.Bats.port_P, m.EBpvs.port_P), directed=True) # pv node

pyo.TransformationFactory("network.expand_arcs").apply_to(m) # apply arcs to model


# Objective function
def obj_fun(m):
# 	return sum((m.Grid.Pbuy[t]*m.cost[t]/1e6 - m.Grid.Psell[t]*m.exc[t]/1e6) + 0*1/1e6*((m.PV.Pinst+m.PV.Pdim)*m.PV.forecast[t]*m.PV.eff + m.PV.P[t]) for t in l_t ) #+ (m.Bat.Pdim*cp_bat + m.Bat.Edim*ce_bat)/365/20#+ m.PV.Pdim*cost_new_pv
	return sum(( m.Gridw.Pbuy[t]*m.costw[t]/1e6 - m.Gridw.Psell[t]*m.excw[t]/1e6 + 
             m.Grids.Pbuy[t]*m.costs[t]/1e6 - m.Grids.Psell[t]*m.excs[t]/1e6)/2  for t in l_t ) + (m.Batw.Pdim*cp_bat + m.Batw.Edim*ce_bat) #+ m.PV.Pdim*cost_new_pv
m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)


instance, results, exec_time = solve(m,'bonmin')

df_benchmark.loc[-1] = ['ISGTcase1',value(instance.goal),exec_time,12.904817912002599,679.13290238380398,get_n_variables(m),2456]
df_benchmark.reset_index(inplace=True,drop=True)


df_full.loc[-1,'T_ISGT1'] = exec_time
df_full.loc[-1,'O_ISGT1'] = value(instance.goal)

file = './test/Benchmark/ISGTcase1'
df_out, df_param, df_size = get_results(file=file, instance=instance, results=results, l_t=l_t, exec_time=exec_time)

del df_cons_aug, df_cons_jan, df_grid_aug, df_grid_jan, df_meteo_aug, df_meteo_jan
del ce_bat, cost_new_pv, cp_bat, data_bat, data_c1, data_Ebre, data_irr, data_p, data_pv, data_R1, data_t
del init_bat, init_c1, init_Ebre, init_p, init_R1, init_t
del l_costs, l_costw, l_excs, l_excw
del m, instance, T, l_t, df_out, df_param, df_size, results, file, exec_time

#%% ISGT2
# LES PLANES CASE (ISGT - 2024 CONFERENCE PAPER -- Case 2) 

from Devices.Reservoirs import Reservoir
from Devices.Sources import Source
from Devices.Pipes import Pipe
from Devices.Pumps import Pump, RealPump, LinealizedPump
from Devices.Turbines import Turbine
from Devices.EB import EB
from Devices.SolarPV import SolarPV
from Devices.MainGrid import Grid
from Devices.Batteries import Battery, NewBattery
from pyomo.contrib.gdpopt.enumerate import GDP_Enumeration_Solver
from pyomo.environ import value

# model
m = pyo.ConcreteModel()

df_meteo_aug = pd.read_csv('data/meteo/LesPlanes_meteo_hour_aug.csv')
df_cons_aug = pd.read_csv('data/irrigation/LesPlanes_irrigation_aug.csv')
df_grid_aug = pd.read_csv('data/costs/PVPC_aug.csv')

df_meteo_jan = pd.read_csv('data/meteo/LesPlanes_meteo_hour_jan.csv')
df_cons_jan = pd.read_csv('data/irrigation/LesPlanes_irrigation_jan.csv')
df_grid_jan = pd.read_csv('data/costs/PVPC_jan.csv')

df_grid_jan['Excedentes_cut'] = df_grid_jan['Excedentes']*(1-0.3*df_grid_jan['Hour'].apply(lambda x: 1 if (x in [8,9,10,11,12,13,14,15,16]) else 0))
df_grid_jan['Excedentes'] = df_grid_jan['Excedentes_cut']
df_grid_aug['Excedentes_cut'] = df_grid_aug['Excedentes']*(1-0.3*df_grid_aug['Hour'].apply(lambda x: 1 if (x in [8,9,10,11,12,13,14,15,16]) else 0))
df_grid_aug['Excedentes'] = df_grid_aug['Excedentes_cut']

# time
T = 24
l_t = list(range(T))
m.tw = pyo.Set(initialize=l_t)
m.ts = pyo.Set(initialize=l_t)

# electricity cost
l_costw = df_grid_jan['PVPC']
l_excw = df_grid_jan['Excedentes']
m.costw = pyo.Param(m.tw, initialize=l_costw)
m.excw = pyo.Param(m.tw, initialize=l_excw)

l_costs = df_grid_aug['PVPC']
l_excs = df_grid_aug['Excedentes']
m.costs = pyo.Param(m.ts, initialize=l_costs)
m.excs = pyo.Param(m.ts, initialize=l_excs)

cost_new_pv = 0.00126*T/1000
cp_bat = 0.00171*T/1000
ce_bat = 0.00856*T/1000
# Costos en €/(kwh·h). Considerant vida de 20 anys, del llibre de Oriol Gomis i Paco Diaz trobem:
#   - Ce = 750 €/kWh (10 anys de vida) -> x2 = 1500 €/kWh -> /20/365/24 = 0.00856 €/(kWh·h)

# ===== Create the system =====
m.ReservoirEbrew = pyo.Block()
m.Reservoir1w = pyo.Block()
m.Irrigation1w = pyo.Block()
m.Pump1w = pyo.Block()
m.Pump2w = pyo.Block()
# m.Turb1w = pyo.Block()
m.Pipe1w = pyo.Block()
m.PVw = pyo.Block()
m.Gridw = pyo.Block()
m.EBgw = pyo.Block()
# m.EBpvw = pyo.Block()
m.Batw = pyo.Block()

m.ReservoirEbres = pyo.Block()
m.Reservoir1s = pyo.Block()
m.Irrigation1s = pyo.Block()
m.Pump1s = pyo.Block()
m.Pump2s = pyo.Block()
# m.Turb1s = pyo.Block()
m.Pipe1s = pyo.Block()
m.PVs = pyo.Block()
m.Grids = pyo.Block()
m.EBgs = pyo.Block()
# m.EBpvs = pyo.Block()
m.Bats = pyo.Block()


data_irr = {'Q':df_cons_jan['Qirr']/3600*1.4} # irrigation
Source(m.Irrigation1w, m.tw, data_irr, {})
data_irr = {'Q':df_cons_aug['Qirr']/3600*1.4} # irrigation
Source(m.Irrigation1s, m.ts, data_irr, {})

data_Ebre = {'dt':3600, 'W0':2e5, 'Wmin':0, 'Wmax':3e5, 'zmin':29.5, 'zmax':30}
init_Ebre = {'Q':[0]*T, 'W':[2e5]*T}
Reservoir(m.ReservoirEbrew, m.tw, data_Ebre, init_Ebre)
Reservoir(m.ReservoirEbres, m.ts, data_Ebre, init_Ebre)
data_R1 = {'dt':3600, 'W0':10e3, 'Wmin':9e3, 'Wmax':13e3, 'zmin':135+(141-135)*9/13, 'zmax':141, 'WT_min':0.95*10e3, 'WT_max':1.05*10e3}
init_R1 = {'Q':[0]*T, 'W':[10e3]*T}
Reservoir(m.Reservoir1w, m.tw, data_R1, init_R1)
Reservoir(m.Reservoir1s, m.ts, data_R1, init_R1)

data_c1 = {'K':297.38*0.20, 'Qmax':1.2} # canal
init_c1 = {'Q':[0]*T, 'H':[108]*T, 'H0':[108]*T, 'zlow':[30]*T, 'zhigh':[138]*T}
Pipe(m.Pipe1w, m.tw, data_c1, init_c1)
Pipe(m.Pipe1s, m.ts, data_c1, init_c1)

data_p = {'A':121.54, 'B':3864.8, 'n_n':2900, 'nmax':1, 'eff':0.8, 'eff_t':0.5, 'S':0.1*0.1*3.14, 'Qmin':0.6250, 'Qmax':1.8688, 'Qnom':0.0556, 'Pmax':110e3, 'intervals':3} # pumps (both equal)
init_p = {'Q':[0]*T, 'H':[109]*T, 'n':[0.99]*T, 'Pe':[0]*T}
RealPump(m.Pump1w, m.tw, data_p, init_p)
RealPump(m.Pump1s, m.ts, data_p, init_p)
data_p['eff']=0.8
RealPump(m.Pump2w, m.tw, data_p, init_p)
RealPump(m.Pump2s, m.ts, data_p, init_p)

# data_pdouble = {'A':121.54, 'B':3864.8/(2**2), 'n_n':2900, 'eff':0.8, 'eff_t':0.5, 'S':0.1*0.1*3.14, 'Qmin':0.6250, 'Qmax':1.8688*2, 'Qnom':0.0556, 'Pmax':2*110e3} # pumps (both equal)
# init_pdouble = {'Q':[0]*T, 'H':[108]*T, 'n':[1]*T, 'Pe':[110e3*0.9]*T}
# RealPump(m.Pump1w, m.tw, data_pdouble, init_pdouble)
# RealPump(m.Pump1s, m.ts, data_pdouble, init_pdouble)

# data_t = {'eff':0.5, 'Pmax':110e3}
# init_t = {'Q':[0]*T, 'H':[109]*T, 'Pe':[0]*T}
# Turbine(m.Turb1w, m.tw, data_t, init_t)
# Turbine(m.Turb1s, m.ts, data_t, init_t)

data_pv = {'Pinst':215.28e3, 'Pmax':215.28e3, 'forecast':df_meteo_jan['Irr']/1000, 'eff':0.98} # PV
SolarPV(m.PVw, m.tw, data_pv)
data_pv = {'Pinst':215.28e3, 'Pmax':215.28e3, 'forecast':df_meteo_aug['Irr']/1000, 'eff':0.98} # PV
SolarPV(m.PVs, m.ts, data_pv)

data_bat = {'dt':3600, 'E0':0.05, 'Emax':200e3, 'Pmax':200e3, 'SOCmin':0.2, 'SOCmax':1.0, 'eff_ch':0.8, 'eff_dc':0.8,'Einst':0.1, 'Pinst':0}
init_bat = {'E':[0]*T, 'P':[0]*T}
NewBattery(m.Batw, m.tw, data_bat, init_bat)
NewBattery(m.Bats, m.ts, data_bat, init_bat)

Grid(m.Gridw, m.tw, {'Pmax':100e6}) # grid
Grid(m.Grids, m.ts, {'Pmax':100e6}) # grid

EB(m.EBgw, m.tw)
EB(m.EBgs, m.ts)
# EB(m.EBpvw, m.tw)
# EB(m.EBpvs, m.ts)


# def ConstraintPumpTurbw(m, t):
#     return m.Turb1w.Qin[t] * (m.Pump1w.PumpOn[t] + m.Pump2w.PumpOn[t]) == 0
# m.Turb1_c_PumpTurbw = pyo.Constraint(m.tw, rule=ConstraintPumpTurbw)
# def ConstraintPumpTurbs(m, t):
#     return m.Turb1s.Qin[t] * (m.Pump1s.PumpOn[t] + m.Pump2s.PumpOn[t]) == 0
# m.Turb1_c_PumpTurbs = pyo.Constraint(m.ts, rule=ConstraintPumpTurbs)

# def ConstraintPump2w(m, t):
#     return m.Pump2w.Qin[t] * m.Pump1w.PumpOn[t] == 0
# m.Turb1_c_Pump2w = pyo.Constraint(m.tw, rule=ConstraintPump2w)
# def ConstraintPump2s(m, t):
#     return m.Pump2s.Qin[t] * m.Pump1s.PumpOn[t] == 0
# m.Turb1_c_Pump2s = pyo.Constraint(m.ts, rule=ConstraintPump2s)


def ConstraintBatEws(m):
    return m.Batw.Edim == m.Bats.Edim
m.c_BatEws = pyo.Constraint(rule=ConstraintBatEws)
def ConstraintBatPws(m):
    return m.Batw.Pdim == m.Bats.Pdim
m.c_BatPws = pyo.Constraint(rule=ConstraintBatPws)


# m.Batw.Edim.fix(0)
# m.Bats.Edim.fix(0)
# m.Batw.Pdim.fix(0)
# m.Bats.Pdim.fix(0)

# m.Turb1s.Pdim.fix(110e3)
# m.Turb1w.Pdim.fix(110e3)


# Connections
m.p1r0w = Arc(ports=(m.Pump1w.port_Qin, m.ReservoirEbrew.port_Q), directed=True)
m.p1c1_Qw = Arc(ports=(m.Pump1w.port_Qout, m.Pipe1w.port_Q), directed=True)
m.p1c1_Hw = Arc(ports=(m.Pump1w.port_H, m.Pipe1w.port_H), directed=True)
m.p1ebw = Arc(ports=(m.Pump1w.port_P, m.EBgw.port_P), directed=True)

m.p2r0w = Arc(ports=(m.Pump2w.port_Qin, m.ReservoirEbrew.port_Q), directed=True)
m.p2c1_Qw = Arc(ports=(m.Pump2w.port_Qout, m.Pipe1w.port_Q), directed=True)
m.p2c1_Hw = Arc(ports=(m.Pump2w.port_H, m.Pipe1w.port_H), directed=True)
m.p2ebw = Arc(ports=(m.Pump2w.port_P, m.EBgw.port_P), directed=True) # pv node

# m.t1r0w = Arc(ports=(m.Turb1w.port_Qout, m.ReservoirEbrew.port_Q), directed=True)
# m.t1c1_Qw = Arc(ports=(m.Turb1w.port_Qin, m.Pipe1w.port_Q), directed=True)
# m.t1c1_Hw = Arc(ports=(m.Turb1w.port_H, m.Pipe1w.port_H), directed=True)
# m.t1ebw = Arc(ports=(m.Turb1w.port_P, m.EBgw.port_P), directed=True)

m.c1r1_Qw = Arc(ports=(m.Pipe1w.port_Q, m.Reservoir1w.port_Q), directed=True)
m.c1r1_zw = Arc(ports=(m.Reservoir1w.port_z, m.Pipe1w.port_zhigh), directed=True)
m.c1r0_zw = Arc(ports=(m.ReservoirEbrew.port_z, m.Pipe1w.port_zlow), directed=True)

m.r1i1w = Arc(ports=(m.Irrigation1w.port_Qin, m.Reservoir1w.port_Q), directed=True)

m.gridebw = Arc(ports=(m.Gridw.port_P, m.EBgw.port_P), directed=True)
m.pvebw = Arc(ports=(m.PVw.port_P, m.EBgw.port_P), directed=True) # pv node
m.batebw = Arc(ports=(m.Batw.port_P, m.EBgw.port_P), directed=True) # pv node

# Connections
m.p1r0s = Arc(ports=(m.Pump1s.port_Qin, m.ReservoirEbres.port_Q), directed=True)
m.p1c1_Qs = Arc(ports=(m.Pump1s.port_Qout, m.Pipe1s.port_Q), directed=True)
m.p1c1_Hs = Arc(ports=(m.Pump1s.port_H, m.Pipe1s.port_H), directed=True)
m.p1ebs = Arc(ports=(m.Pump1s.port_P, m.EBgs.port_P), directed=True)

m.p2r0s = Arc(ports=(m.Pump2s.port_Qin, m.ReservoirEbres.port_Q), directed=True)
m.p2c1_Qs = Arc(ports=(m.Pump2s.port_Qout, m.Pipe1s.port_Q), directed=True)
m.p2c1_Hs = Arc(ports=(m.Pump2s.port_H, m.Pipe1s.port_H), directed=True)
m.p2ebs = Arc(ports=(m.Pump2s.port_P, m.EBgs.port_P), directed=True) # pv node

# m.t1r0s = Arc(ports=(m.Turb1s.port_Qout, m.ReservoirEbres.port_Q), directed=True)
# m.t1c1_Qs = Arc(ports=(m.Turb1s.port_Qin, m.Pipe1s.port_Q), directed=True)
# m.t1c1_Hs = Arc(ports=(m.Turb1s.port_H, m.Pipe1s.port_H), directed=True)
# m.t1ebs = Arc(ports=(m.Turb1s.port_P, m.EBgs.port_P), directed=True)

m.c1r1_Qs = Arc(ports=(m.Pipe1s.port_Q, m.Reservoir1s.port_Q), directed=True)
m.c1r1_zs = Arc(ports=(m.Reservoir1s.port_z, m.Pipe1s.port_zhigh), directed=True)
m.c1r0_zs = Arc(ports=(m.ReservoirEbres.port_z, m.Pipe1s.port_zlow), directed=True)

m.r1i1s = Arc(ports=(m.Irrigation1s.port_Qin, m.Reservoir1s.port_Q), directed=True)

m.gridebs = Arc(ports=(m.Grids.port_P, m.EBgs.port_P), directed=True)
m.pvebs = Arc(ports=(m.PVs.port_P, m.EBgs.port_P), directed=True) # pv node
m.batebs = Arc(ports=(m.Bats.port_P, m.EBgs.port_P), directed=True) # pv node

pyo.TransformationFactory("network.expand_arcs").apply_to(m) # apply arcs to model


# Objective function
def obj_fun(m):
# 	return sum((m.Grid.Pbuy[t]*m.cost[t]/1e6 - m.Grid.Psell[t]*m.exc[t]/1e6) + 0*1/1e6*((m.PV.Pinst+m.PV.Pdim)*m.PV.forecast[t]*m.PV.eff + m.PV.P[t]) for t in l_t ) #+ (m.Bat.Pdim*cp_bat + m.Bat.Edim*ce_bat)/365/20#+ m.PV.Pdim*cost_new_pv
	return sum(( m.Gridw.Pbuy[t]*m.costw[t]/1e6 - m.Gridw.Psell[t]*m.excw[t]/1e6 + 
             m.Grids.Pbuy[t]*m.costs[t]/1e6 - m.Grids.Psell[t]*m.excs[t]/1e6)/2  for t in l_t ) + (m.Batw.Pdim*cp_bat + m.Batw.Edim*ce_bat) #+ m.PV.Pdim*cost_new_pv
m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)


instance, results, exec_time = solve(m,'bonmin')

df_benchmark.loc[-1] = ['ISGTcase2',value(instance.goal),exec_time,-29.510384042216199,3950.4513001441901,get_n_variables(m),2070]
df_benchmark.reset_index(inplace=True,drop=True)


df_full.loc[-1,'T_ISGT2'] = exec_time
df_full.loc[-1,'O_ISGT2'] = value(instance.goal)

file = './test/Benchmark/ISGTcase2'
df_out, df_param, df_size = get_results(file=file, instance=instance, results=results, l_t=l_t, exec_time=exec_time)

del df_cons_aug, df_cons_jan, df_grid_aug, df_grid_jan, df_meteo_aug, df_meteo_jan
del ce_bat, cost_new_pv, cp_bat, data_bat, data_c1, data_Ebre, data_irr, data_p, data_pv, data_R1
del init_bat, init_c1, init_Ebre, init_p, init_R1
del l_costs, l_costw, l_excs, l_excw
del m, instance, T, l_t, df_out, df_param, df_size, results, file, exec_time


#%% ISGT3
# LES PLANES CASE (ISGT - 2024 CONFERENCE PAPER -- Case 3) 

from Devices.Reservoirs import Reservoir
from Devices.Sources import Source
from Devices.Pipes import Pipe
from Devices.Pumps import Pump, RealPump, LinealizedPump
from Devices.Turbines import Turbine
from Devices.EB import EB
from Devices.SolarPV import SolarPV
from Devices.MainGrid import Grid
from Devices.Batteries import Battery, NewBattery
from pyomo.contrib.gdpopt.enumerate import GDP_Enumeration_Solver
from pyomo.environ import value

# model
m = pyo.ConcreteModel()

df_meteo_aug = pd.read_csv('data/meteo/LesPlanes_meteo_hour_aug.csv')
df_cons_aug = pd.read_csv('data/irrigation/LesPlanes_irrigation_aug.csv')
df_grid_aug = pd.read_csv('data/costs/PVPC_aug.csv')

df_meteo_jan = pd.read_csv('data/meteo/LesPlanes_meteo_hour_jan.csv')
df_cons_jan = pd.read_csv('data/irrigation/LesPlanes_irrigation_jan.csv')
df_grid_jan = pd.read_csv('data/costs/PVPC_jan.csv')

df_grid_jan['Excedentes_cut'] = df_grid_jan['Excedentes']*(1-0.3*df_grid_jan['Hour'].apply(lambda x: 1 if (x in [8,9,10,11,12,13,14,15,16]) else 0))
df_grid_jan['Excedentes'] = df_grid_jan['Excedentes_cut']
df_grid_aug['Excedentes_cut'] = df_grid_aug['Excedentes']*(1-0.3*df_grid_aug['Hour'].apply(lambda x: 1 if (x in [8,9,10,11,12,13,14,15,16]) else 0))
df_grid_aug['Excedentes'] = df_grid_aug['Excedentes_cut']

# time
T = 24
l_t = list(range(T))
m.tw = pyo.Set(initialize=l_t)
m.ts = pyo.Set(initialize=l_t)

# electricity cost
l_costw = df_grid_jan['PVPC']
l_excw = df_grid_jan['Excedentes']
m.costw = pyo.Param(m.tw, initialize=l_costw)
m.excw = pyo.Param(m.tw, initialize=l_excw)

l_costs = df_grid_aug['PVPC']
l_excs = df_grid_aug['Excedentes']
m.costs = pyo.Param(m.ts, initialize=l_costs)
m.excs = pyo.Param(m.ts, initialize=l_excs)

cost_new_pv = 0.00126*T/1000
cp_bat = 0.00171*T/1000
ce_bat = 0.00856*T/1000
# Costos en €/(kwh·h). Considerant vida de 20 anys, del llibre de Oriol Gomis i Paco Diaz trobem:
#   - Ce = 750 €/kWh (10 anys de vida) -> x2 = 1500 €/kWh -> /20/365/24 = 0.00856 €/(kWh·h)

# ===== Create the system =====
m.ReservoirEbrew = pyo.Block()
m.Reservoir1w = pyo.Block()
m.Irrigation1w = pyo.Block()
m.Pump1w = pyo.Block()
m.Pump2w = pyo.Block()
m.Turb1w = pyo.Block()
m.Pipe1w = pyo.Block()
m.PVw = pyo.Block()
m.Gridw = pyo.Block()
m.EBgw = pyo.Block()
# m.EBpvw = pyo.Block()
m.Batw = pyo.Block()

m.ReservoirEbres = pyo.Block()
m.Reservoir1s = pyo.Block()
m.Irrigation1s = pyo.Block()
m.Pump1s = pyo.Block()
m.Pump2s = pyo.Block()
m.Turb1s = pyo.Block()
m.Pipe1s = pyo.Block()
m.PVs = pyo.Block()
m.Grids = pyo.Block()
m.EBgs = pyo.Block()
# m.EBpvs = pyo.Block()
m.Bats = pyo.Block()


data_irr = {'Q':df_cons_jan['Qirr']/3600*1.4} # irrigation
Source(m.Irrigation1w, m.tw, data_irr, {})
data_irr = {'Q':df_cons_aug['Qirr']/3600*1.4} # irrigation
Source(m.Irrigation1s, m.ts, data_irr, {})

data_Ebre = {'dt':3600, 'W0':2e5, 'Wmin':0, 'Wmax':3e5, 'zmin':29.5, 'zmax':30}
init_Ebre = {'Q':[0]*T, 'W':[2e5]*T}
Reservoir(m.ReservoirEbrew, m.tw, data_Ebre, init_Ebre)
Reservoir(m.ReservoirEbres, m.ts, data_Ebre, init_Ebre)
data_R1 = {'dt':3600, 'W0':10e3, 'Wmin':9e3, 'Wmax':13e3, 'zmin':135+(141-135)*9/13, 'zmax':141, 'WT_min':0.95*10e3, 'WT_max':1.05*10e3}
init_R1 = {'Q':[0]*T, 'W':[10e3]*T}
Reservoir(m.Reservoir1w, m.tw, data_R1, init_R1)
Reservoir(m.Reservoir1s, m.ts, data_R1, init_R1)

data_c1 = {'K':297.38*0.20, 'Qmax':1.2} # canal
init_c1 = {'Q':[0]*T, 'H':[108]*T, 'H0':[108]*T, 'zlow':[30]*T, 'zhigh':[138]*T}
Pipe(m.Pipe1w, m.tw, data_c1, init_c1)
Pipe(m.Pipe1s, m.ts, data_c1, init_c1)

data_p = {'A':121.54, 'B':3864.8, 'n_n':2900, 'nmax':1, 'eff':0.8, 'eff_t':0.5, 'S':0.1*0.1*3.14, 'Qmin':0.6250, 'Qmax':1.8688, 'Qnom':0.0556, 'Pmax':110e3} # pumps (both equal)
init_p = {'Q':[0]*T, 'H':[109]*T, 'n':[0.99]*T, 'Pe':[0]*T}
RealPump(m.Pump1w, m.tw, data_p, init_p)
RealPump(m.Pump1s, m.ts, data_p, init_p)
data_p['eff']=0.8
RealPump(m.Pump2w, m.tw, data_p, init_p)
RealPump(m.Pump2s, m.ts, data_p, init_p)

# data_pdouble = {'A':121.54, 'B':3864.8/(2**2), 'n_n':2900, 'eff':0.8, 'eff_t':0.5, 'S':0.1*0.1*3.14, 'Qmin':0.6250, 'Qmax':1.8688*2, 'Qnom':0.0556, 'Pmax':2*110e3} # pumps (both equal)
# init_pdouble = {'Q':[0]*T, 'H':[108]*T, 'n':[1]*T, 'Pe':[110e3*0.9]*T}
# RealPump(m.Pump1w, m.tw, data_pdouble, init_pdouble)
# RealPump(m.Pump1s, m.ts, data_pdouble, init_pdouble)

data_t = {'eff':0.5, 'Pmax':110e3}
init_t = {'Q':[0]*T, 'H':[109]*T, 'Pe':[0]*T}
Turbine(m.Turb1w, m.tw, data_t, init_t)
Turbine(m.Turb1s, m.ts, data_t, init_t)

data_pv = {'Pinst':215.28e3, 'Pmax':215.28e3, 'forecast':df_meteo_jan['Irr']/1000, 'eff':0.98} # PV
SolarPV(m.PVw, m.tw, data_pv)
data_pv = {'Pinst':215.28e3, 'Pmax':215.28e3, 'forecast':df_meteo_aug['Irr']/1000, 'eff':0.98} # PV
SolarPV(m.PVs, m.ts, data_pv)

data_bat = {'dt':3600, 'E0':0.05, 'Emax':200e3, 'Pmax':200e3, 'SOCmin':0.2, 'SOCmax':1.0, 'eff_ch':0.8, 'eff_dc':0.8,'Einst':0.1, 'Pinst':0}
init_bat = {'E':[0]*T, 'P':[0]*T}
NewBattery(m.Batw, m.tw, data_bat, init_bat)
NewBattery(m.Bats, m.ts, data_bat, init_bat)

Grid(m.Gridw, m.tw, {'Pmax':100e6}) # grid
Grid(m.Grids, m.ts, {'Pmax':100e6}) # grid

EB(m.EBgw, m.tw)
EB(m.EBgs, m.ts)
# EB(m.EBpvw, m.tw)
# EB(m.EBpvs, m.ts)


# def ConstraintPumpTurbw(m, t):
#     return m.Turb1w.Qin[t] * (m.Pump1w.PumpOn[t] + m.Pump2w.PumpOn[t]) == 0
# m.Turb1_c_PumpTurbw = pyo.Constraint(m.tw, rule=ConstraintPumpTurbw)
# def ConstraintPumpTurbs(m, t):
#     return m.Turb1s.Qin[t] * (m.Pump1s.PumpOn[t] + m.Pump2s.PumpOn[t]) == 0
# m.Turb1_c_PumpTurbs = pyo.Constraint(m.ts, rule=ConstraintPumpTurbs)

# def ConstraintPump2w(m, t):
#     return m.Pump2w.Qin[t] * m.Pump1w.PumpOn[t] == 0
# m.Turb1_c_Pump2w = pyo.Constraint(m.tw, rule=ConstraintPump2w)
# def ConstraintPump2s(m, t):
#     return m.Pump2s.Qin[t] * m.Pump1s.PumpOn[t] == 0
# m.Turb1_c_Pump2s = pyo.Constraint(m.ts, rule=ConstraintPump2s)


def ConstraintBatEws(m):
    return m.Batw.Edim == m.Bats.Edim
m.c_BatEws = pyo.Constraint(rule=ConstraintBatEws)
def ConstraintBatPws(m):
    return m.Batw.Pdim == m.Bats.Pdim
m.c_BatPws = pyo.Constraint(rule=ConstraintBatPws)


# m.Batw.Edim.fix(0)
# m.Bats.Edim.fix(0)
# m.Batw.Pdim.fix(0)
# m.Bats.Pdim.fix(0)

m.Turb1s.Pdim.fix(110e3)
m.Turb1w.Pdim.fix(110e3)


# Connections
m.p1r0w = Arc(ports=(m.Pump1w.port_Qin, m.ReservoirEbrew.port_Q), directed=True)
m.p1c1_Qw = Arc(ports=(m.Pump1w.port_Qout, m.Pipe1w.port_Q), directed=True)
m.p1c1_Hw = Arc(ports=(m.Pump1w.port_H, m.Pipe1w.port_H), directed=True)
m.p1ebw = Arc(ports=(m.Pump1w.port_P, m.EBgw.port_P), directed=True)

m.p2r0w = Arc(ports=(m.Pump2w.port_Qin, m.ReservoirEbrew.port_Q), directed=True)
m.p2c1_Qw = Arc(ports=(m.Pump2w.port_Qout, m.Pipe1w.port_Q), directed=True)
m.p2c1_Hw = Arc(ports=(m.Pump2w.port_H, m.Pipe1w.port_H), directed=True)
m.p2ebw = Arc(ports=(m.Pump2w.port_P, m.EBgw.port_P), directed=True) # pv node

m.t1r0w = Arc(ports=(m.Turb1w.port_Qout, m.ReservoirEbrew.port_Q), directed=True)
m.t1c1_Qw = Arc(ports=(m.Turb1w.port_Qin, m.Pipe1w.port_Q), directed=True)
m.t1c1_Hw = Arc(ports=(m.Turb1w.port_H, m.Pipe1w.port_H), directed=True)
m.t1ebw = Arc(ports=(m.Turb1w.port_P, m.EBgw.port_P), directed=True)

m.c1r1_Qw = Arc(ports=(m.Pipe1w.port_Q, m.Reservoir1w.port_Q), directed=True)
m.c1r1_zw = Arc(ports=(m.Reservoir1w.port_z, m.Pipe1w.port_zhigh), directed=True)
m.c1r0_zw = Arc(ports=(m.ReservoirEbrew.port_z, m.Pipe1w.port_zlow), directed=True)

m.r1i1w = Arc(ports=(m.Irrigation1w.port_Qin, m.Reservoir1w.port_Q), directed=True)

m.gridebw = Arc(ports=(m.Gridw.port_P, m.EBgw.port_P), directed=True)
m.pvebw = Arc(ports=(m.PVw.port_P, m.EBgw.port_P), directed=True) # pv node
m.batebw = Arc(ports=(m.Batw.port_P, m.EBgw.port_P), directed=True) # pv node

# Connections
m.p1r0s = Arc(ports=(m.Pump1s.port_Qin, m.ReservoirEbres.port_Q), directed=True)
m.p1c1_Qs = Arc(ports=(m.Pump1s.port_Qout, m.Pipe1s.port_Q), directed=True)
m.p1c1_Hs = Arc(ports=(m.Pump1s.port_H, m.Pipe1s.port_H), directed=True)
m.p1ebs = Arc(ports=(m.Pump1s.port_P, m.EBgs.port_P), directed=True)

m.p2r0s = Arc(ports=(m.Pump2s.port_Qin, m.ReservoirEbres.port_Q), directed=True)
m.p2c1_Qs = Arc(ports=(m.Pump2s.port_Qout, m.Pipe1s.port_Q), directed=True)
m.p2c1_Hs = Arc(ports=(m.Pump2s.port_H, m.Pipe1s.port_H), directed=True)
m.p2ebs = Arc(ports=(m.Pump2s.port_P, m.EBgs.port_P), directed=True) # pv node

m.t1r0s = Arc(ports=(m.Turb1s.port_Qout, m.ReservoirEbres.port_Q), directed=True)
m.t1c1_Qs = Arc(ports=(m.Turb1s.port_Qin, m.Pipe1s.port_Q), directed=True)
m.t1c1_Hs = Arc(ports=(m.Turb1s.port_H, m.Pipe1s.port_H), directed=True)
m.t1ebs = Arc(ports=(m.Turb1s.port_P, m.EBgs.port_P), directed=True)

m.c1r1_Qs = Arc(ports=(m.Pipe1s.port_Q, m.Reservoir1s.port_Q), directed=True)
m.c1r1_zs = Arc(ports=(m.Reservoir1s.port_z, m.Pipe1s.port_zhigh), directed=True)
m.c1r0_zs = Arc(ports=(m.ReservoirEbres.port_z, m.Pipe1s.port_zlow), directed=True)

m.r1i1s = Arc(ports=(m.Irrigation1s.port_Qin, m.Reservoir1s.port_Q), directed=True)

m.gridebs = Arc(ports=(m.Grids.port_P, m.EBgs.port_P), directed=True)
m.pvebs = Arc(ports=(m.PVs.port_P, m.EBgs.port_P), directed=True) # pv node
m.batebs = Arc(ports=(m.Bats.port_P, m.EBgs.port_P), directed=True) # pv node

pyo.TransformationFactory("network.expand_arcs").apply_to(m) # apply arcs to model


# Objective function
def obj_fun(m):
# 	return sum((m.Grid.Pbuy[t]*m.cost[t]/1e6 - m.Grid.Psell[t]*m.exc[t]/1e6) + 0*1/1e6*((m.PV.Pinst+m.PV.Pdim)*m.PV.forecast[t]*m.PV.eff + m.PV.P[t]) for t in l_t ) #+ (m.Bat.Pdim*cp_bat + m.Bat.Edim*ce_bat)/365/20#+ m.PV.Pdim*cost_new_pv
	return sum(( m.Gridw.Pbuy[t]*m.costw[t]/1e6 - m.Gridw.Psell[t]*m.excw[t]/1e6 + 
             m.Grids.Pbuy[t]*m.costs[t]/1e6 - m.Grids.Psell[t]*m.excs[t]/1e6)/2  for t in l_t ) + (m.Batw.Pdim*cp_bat + m.Batw.Edim*ce_bat) #+ m.PV.Pdim*cost_new_pv
m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)


instance, results, exec_time = solve(m,'bonmin')

df_benchmark.loc[-1] = ['ISGTcase3',value(instance.goal),exec_time,-30.22893512732645,400.43591022491455,get_n_variables(m),2550]
df_benchmark.reset_index(inplace=True,drop=True)


df_full.loc[-1,'T_ISGT3'] = exec_time
df_full.loc[-1,'O_ISGT3'] = value(instance.goal)

file = './test/Benchmark/ISGTcase3'
df_out, df_param, df_size = get_results(file=file, instance=instance, results=results, l_t=l_t, exec_time=exec_time)

del df_cons_aug, df_cons_jan, df_grid_aug, df_grid_jan, df_meteo_aug, df_meteo_jan
del ce_bat, cost_new_pv, cp_bat, data_bat, data_c1, data_Ebre, data_irr, data_p, data_pv, data_R1, data_t
del init_bat, init_c1, init_Ebre, init_p, init_R1, init_t
del l_costs, l_costw, l_excs, l_excw
del m, instance, T, l_t, df_out, df_param, df_size, results, file, exec_time
