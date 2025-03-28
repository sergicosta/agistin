"""

Sensitivity analysis file

"""

import logging

import os
import time

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
# Import pyomo
import pyomo.environ as pyo
from pyomo.network import Arc, Port
from pyomo.util.infeasible import log_infeasible_constraints
from pyomo.contrib.gdpopt.enumerate import GDP_Enumeration_Solver
from pyomo.environ import value

# Import useful functions
from Utilities import clear_clc, model_to_file
from Utilities import get_results, get_n_variables



#%% ISGT3

def ISGT3(df_grid_jan,df_grid_aug):
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
    # df_grid_aug = pd.read_csv('data/costs/PVPC_aug.csv')
    
    df_meteo_jan = pd.read_csv('data/meteo/LesPlanes_meteo_hour_jan.csv')
    df_cons_jan = pd.read_csv('data/irrigation/LesPlanes_irrigation_jan.csv')
    # df_grid_jan = pd.read_csv('data/costs/PVPC_jan.csv')
    
    df_grid_jan['Excedentes_cut'] = df_grid_jan['Excedentes']*(1-0.3*df_grid_jan['Hour'].apply(lambda x: 1 if (x in [8,9,10,11,12,13,14,15,16]) else 0))
    df_grid_jan['Excedentes'] = df_grid_jan['Excedentes_cut']
    df_grid_aug['Excedentes_cut'] = df_grid_aug['Excedentes']*(1-0.3*df_grid_aug['Hour'].apply(lambda x: 1 if (x in [8,9,10,11,12,13,14,15,16]) else 0))
    df_grid_aug['Excedentes'] = df_grid_aug['Excedentes_cut']
    
    # df_grid_aug = df_grid_aug * (1 + increment/100)
    # df_grid_jan = df_grid_jan * (1 + increment/100)
    
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
    
    data_p = {'A':121.54, 'B':3864.8, 'n_n':2900, 'nmax':1, 'eff':0.8, 'eff_t':0.5, 'S':0.1*0.1*3.14, 'Qmin':0.6250, 'Qmax':1.8688, 'Qnom':0.0556, 'Pmax':110e3,'intervals':3} # pumps (both equal)
    init_p = {'Q':[0]*T, 'H':[109]*T, 'n':[0.99]*T, 'Pe':[0]*T}
    LinealizedPump(m.Pump1w, m.tw, data_p, init_p)
    LinealizedPump(m.Pump1s, m.ts, data_p, init_p)
    data_p['eff']=0.8
    LinealizedPump(m.Pump2w, m.tw, data_p, init_p)
    LinealizedPump(m.Pump2s, m.ts, data_p, init_p)
    
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
    
    
    
    # df_full.loc[-1,'T_ISGT3'] = exec_time
    # df_full.loc[-1,'O_ISGT3'] = value(instance.goal)
    
    # file = './test/Benchmark/ISGTcase3'
    # df_out, df_param, df_size = get_results(file=file, instance=instance, results=results, l_t=l_t, exec_time=exec_time)
    
    # del df_cons_aug, df_cons_jan, df_grid_aug, df_grid_jan, df_meteo_aug, df_meteo_jan
    # del ce_bat, cost_new_pv, cp_bat, data_bat, data_c1, data_Ebre, data_irr, data_p, data_pv, data_R1, data_t
    # del init_bat, init_c1, init_Ebre, init_p, init_R1, init_t
    # del l_costs, l_costw, l_excs, l_excw
    # del m, instance, T, l_t, df_out, df_param, df_size, results, file, exec_time

    return instance,results,exec_time,m

def solve(m, solver='couenne'):

    instance = m.create_instance()

    start_time = time.time()
        
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


#%% Sensitivity analysis



# v_increment = [0,-8,8, 13,-13]      #increment cost vector in %

df_grid_aug = pd.read_csv('data/costs/PVPC_aug.csv')
df_grid_jan = pd.read_csv('data/costs/PVPC_jan.csv')

df_grid_aug2023 = df_grid_aug
df_grid_jan2023 = df_grid_jan

df_grid_aug2023['PVPC'] = df_grid_aug['PVPC']*1
df_grid_jan2023['PVPC'] = df_grid_jan['PVPC']*1


df_grid = {'2024':[df_grid_aug,df_grid_jan],
           '2023':[df_grid_aug2023,df_grid_jan2023]}


df_sensitivity = pd.DataFrame(columns=['Case','Sensitivity','Goal','BaseGoal','Time','Base Time'])

j=0
for key, (df_grid_aug, df_grid_jan) in df_grid.items():
    
    instance, results, exec_time,m = ISGT3(df_grid_aug, df_grid_jan)

    if j == 0:
        value_goal0 = value(instance.goal) 
        exec_time0 = exec_time  
    else:
        df_sensitivity.loc[j] = ['ISGT 3', key, value(instance.goal), value_goal0, exec_time, exec_time0]
    
    j += 1  
    del instance,results,exec_time,m

# Sensitivity analysis (Tornado Diagram)

labels = np.char.array(["X1% +/-","X5% +/-","X6% +/-"])
low_values = []
high_values = []

for i in range(len(df_sensitivity)): 
    if df_sensitivity['Increment'][i+1] < 0:
        low_values.append(df_sensitivity['Goal'][i+1])
    else:
        high_values.append(df_sensitivity['Goal'][i+1])
        
midpoint = df_sensitivity['BaseGoal'][1]
high_values = np.array(high_values)
low_values = np.array(low_values)

var_effect = np.abs(high_values - low_values)/midpoint
                
data = pd.DataFrame({'Labels':labels,'Low values':low_values,'High values':high_values,'Variable effect':var_effect})


data = data.sort_values('Variable effect',ascending=True,inplace=False,ignore_index=False,key=None)



def tornado_chart(labels, midpoint, low_values, high_values, title="Tornado Diagram"):
    """
    Parameters
    ----------
    labels : np.array()
        List of label titles used to identify the variables, y-axis of bar
        chart. The length of labels is used to iterate through to generate 
        the bar charts.
    midpoint : float
        Center value for bar charts to extend from. In sensitivity analysis
        this is often the 'neutral' or 'default' model output.
    low_values : np.array()
        An np.array of the model output resulting from the low variable 
        selection. Same length and order as labels. 
    high_values : np.array()
        An np.array of the model output resulting from the high variable
        selection. Same length and order as labels.
    """
    
    color_low = '#e1ceff'  # Azul claro para low_values
    color_high = '#ff6262' # Rojo para high_values
    
    ys = range(len(labels))  
    
    fig, ax = plt.subplots(figsize=(9, len(labels) * 0.5 + 4))
    
    for y, low_value, high_value in zip(ys, low_values, high_values):
    
        low_width = midpoint - low_value
        high_width = high_value - midpoint
    
        ax.broken_barh(
            [
                (low_value, low_width),
                (midpoint, high_width)
            ],
            (y - 0.4, 0.5),  
            facecolors=[color_low, color_high],
            edgecolors=['black', 'black'],
            linewidth=0.5
        )
        
        offset_x = 0.5
        offset_y = 0.2

        ax.text(low_value - offset_x, y + offset_y, f"{low_value:.2f}", va='center', ha='right', fontsize=10, fontweight='bold')
        ax.text(high_value + offset_x, y + offset_y, f"{high_value:.2f}", va='center', ha='left', fontsize=10, fontweight='bold')
    
    ax.axvline(midpoint, color='black', linewidth=1)

    ax.spines[['right', 'left', 'top']].set_visible(False)
    ax.set_yticks([])
    
    ax.text(midpoint, len(labels)-0.4, title, color='black', fontsize=15, va='center', ha='center', fontweight='bold')

    ax.set_xlabel('+y, -y')
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)

    x_min = min(low_values) - 5 
    x_max = max(high_values) + 5  
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.5, len(labels) - 0.5)
    
    ax.tick_params(left=False)

    low_patch = mpatches.Patch(color=color_low, label='- y')
    high_patch = mpatches.Patch(color=color_high, label='+ y')
    ax.legend(handles=[low_patch, high_patch], loc='upper right', fontsize=10, frameon=False)

    plt.show()

    return

tornado_chart(labels, midpoint, low_values, high_values)
