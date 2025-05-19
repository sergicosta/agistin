"""
    AGISTIN - EXAMPLE 2
    
    Optimization usage example of two reservoirs, considering irrigation consumption,
    and two pumps. The pumping station gathers the pumps' power consumption, a solar
    PV plant and a connection point to the public grid.
    The PV plant can be upgraded to double its size (from 50e3 to 100e3) with a cost of 10 per power unit.
    The price of selling power to the grid is half of the cost of buying it.
    
    Authors: Sergi Costa Dilmé (CITCEA-UPC), Juan Carlos Olives-Camps (CITCEA-UPC), Paula Muñoz Peña (CITCEA-UPC)
"""

"""
RUGOSITAT:
    - PE-100: Ka = 0.050 mm
    - PVC-O: Ka = 0.050 mm
"""
from pyomo.util.infeasible import log_infeasible_constraints
import logging

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import pyomo
import pyomo.environ as pyo
from pyomo.network import Arc, Port

# Import devices
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

# Import useful functions
from Utilities import clear_clc, model_to_file

#clean console and variable pane
# clear_clc() #consider removing if you are not working with Spyder

# model
m = pyo.ConcreteModel()

df_meteo_aug = pd.read_csv('data/meteo/LesPlanes_meteo_hour_aug.csv')
df_cons_aug = pd.read_excel('data/irrigation/EB5_irrigation_s.xlsx')
df_grid_aug = pd.read_csv('data/costs/PVPC_aug.csv')
df_B1cons_aug= pd.read_csv('data/irrigation/B1B2B3_irrigation_s.csv')

df_meteo_jan = pd.read_csv('data/meteo/LesPlanes_meteo_hour_jan.csv')
df_cons_jan = pd.read_excel('data/irrigation/EB5_irrigation_w.xlsx')
df_grid_jan = pd.read_csv('data/costs/PVPC_jan.csv')
df_B1cons_jan= pd.read_csv('data/irrigation/B1B2B3_irrigation_w.csv')

df_grid_jan['Excedentes_cut'] = df_grid_jan['Excedentes']*(1*df_grid_jan['Hour'].apply(lambda x: 1 if (x in [8,9,10,11,12,13,14,15,16]) else 0))
# df_grid_jan['Excedentes'] = df_grid_jan['Excedentes_cut']
df_grid_aug['Excedentes_cut'] = df_grid_aug['Excedentes']*(1*df_grid_aug['Hour'].apply(lambda x: 1 if (x in [8,9,10,11,12,13,14,15,16]) else 0))
# df_grid_aug['Excedentes'] = df_grid_aug['Excedentes_cut']

# time
T = 24
l_t = list(range(T))
m.tw = pyo.Set(initialize=l_t)
m.ts = pyo.Set(initialize=l_t)

# electricity cost
l_costw = df_grid_jan['PVPC'].head(T)
l_excw = df_grid_jan['Excedentes'].head(T)
m.costw = pyo.Param(m.tw, initialize=l_costw)
m.excw = pyo.Param(m.tw, initialize=l_excw)

l_costs = df_grid_aug['PVPC'].head(T)
l_excs = df_grid_aug['Excedentes'].head(T)
m.costs = pyo.Param(m.ts, initialize=l_costs)
m.excs = pyo.Param(m.ts, initialize=l_excs)

cost_new_pv = 0.00126*T/1000
cp_bat = 0.00171*T/1000
ce_bat = 0.00856*T/1000
# Costos en €/(kwh·h). Considerant vida de 20 anys, del llibre de Oriol Gomis i Paco Diaz trobem:
#   - Ce = 750 €/kWh (10 anys de vida) -> x2 = 1500 €/kWh -> /20/365/24 = 0.00856 €/(kWh·h)

# ===== Create the system =====
m.B0w = pyo.Block()
m.B1w = pyo.Block()
m.B4w = pyo.Block()
m.Irrigation1w = pyo.Block()
m.Irrigation4w = pyo.Block()
m.Pump4w = pyo.Block() #Pumps B1 --> B4
m.Pump5w = pyo.Block() #Pumps B1 --> B4
# m.Pump6w = pyo.Block() #Pumps B1 --> B4
m.Pump7w = pyo.Block() #Pumps B0 --> B1
m.Pump8w = pyo.Block() #Pumps B0 --> B1
# m.Pump9w = pyo.Block() 
m.Turb2w = pyo.Block()
m.Pipe0w = pyo.Block()
m.Pipe2w = pyo.Block()
m.PVw = pyo.Block()
m.Gridw = pyo.Block()
m.EBgw = pyo.Block()
m.EBpvw = pyo.Block()
# m.Batw = pyo.Block()

m.B0s = pyo.Block()
m.B1s = pyo.Block()
m.B4s = pyo.Block()
m.Irrigation1s = pyo.Block()
m.Irrigation4s = pyo.Block()
m.Pump4s = pyo.Block()  #Pumps B1 to B4
m.Pump5s = pyo.Block()  #Pumps B1 to B4
# m.Pump6s = pyo.Block() #Pumps B1 to B4
m.Pump7s = pyo.Block()  #Pumps B0 to B1
m.Pump8s = pyo.Block()  #Pumps B0 to B1
# m.Pump9s = pyo.Block()  #Pumps B0 to B1
m.Turb2s = pyo.Block()
m.Pipe0s = pyo.Block()
m.Pipe2s = pyo.Block()
m.PVs = pyo.Block()
m.Grids = pyo.Block()
m.EBgs = pyo.Block()
m.EBpvs = pyo.Block()
# m.Bats = pyo.Block()


data_irrB1 = {'Q':df_B1cons_jan['Qirr'].head(T)/3600} # irrigation
Source(m.Irrigation1w, m.tw, data_irrB1, {})
data_irrB1 = {'Q':df_B1cons_aug['Qirr'].head(T)/3600} # irrigation
Source(m.Irrigation1s, m.ts, data_irrB1, {})

data_irrB4 = {'Q':2*df_cons_jan['Qirr'].head(T)/3600} # irrigation
Source(m.Irrigation4w, m.tw, data_irrB4, {})
data_irrB4 = {'Q':2*df_cons_aug['Qirr'].head(T)/3600} # irrigation
Source(m.Irrigation4s, m.ts, data_irrB4, {})

data_B0 = {'dt':3600, 'W0':120e6, 'Wmin':0, 'Wmax':2*120e6, 'zmin':129, 'zmax':131}
init_B0 = {'Q':[0]*T, 'W':[120e6]*T}
Reservoir(m.B0w, m.tw, data_B0, init_B0)
Reservoir(m.B0s, m.ts, data_B0, init_B0)
data_B1 = {'dt':3600, 'W0':120e3, 'Wmin':0.80*120e3, 'Wmax':142869, 'zmin':331.1+(339-331.1)*0.8*120e3/142869, 'zmax':339, 'WT_min':0.9999*120e3, 'WT_max':1.2*120e3}
init_B1 = {'Q':-df_B1cons_aug['Qirr'].head(T)/3600, 'W':[120e3]*T}
Reservoir(m.B1w, m.tw, data_B1, init_B1)
Reservoir(m.B1s, m.ts, data_B1, init_B1)
data_B4 = {'dt':3600, 'W0':230e3, 'Wmin':0.80*230e3, 'Wmax':270e3, 'zmin':420.19+(431.06-420.19)*0.8*230e3/269485, 'zmax':431.06, 'WT_min':0.9999*230e3, 'WT_max':1.2*230e3}
init_B4 = {'Q':-2*df_cons_aug['Qirr'].head(T)/3600, 'W':[230e3]*T}
Reservoir(m.B4w, m.tw, data_B4, init_B4)
Reservoir(m.B4s, m.ts, data_B4, init_B4)


data_c0 = {'K':27.90, 'Qmax':100, 'H_approx':206} # canal B0 to B1
init_c0 = {'Q':[0]*T, 'H':[206]*T, 'H0':[206]*T, 'zlow':[131]*T, 'zhigh':[337]*T}
Pipe(m.Pipe0w, m.tw, data_c0, init_c0)
Pipe(m.Pipe0s, m.ts, data_c0, init_c0)
data_c2 = {'K':27.81, 'Qmax':100, 'H_approx':88} # canal B1 to B4
init_c2 = {'Q':[0]*T, 'H':[88]*T, 'H0':[88]*T, 'zlow':[339]*T, 'zhigh':[427]*T}
Pipe(m.Pipe2w, m.tw, data_c2, init_c2)
Pipe(m.Pipe2s, m.ts, data_c2, init_c2)

#Pumps B1 to B4
data_p1 = {'A':148, 'B':103.68,'nmax':1, 'eff':0.922, 'Qmin':0.3, 'Qmax':1.1, 'Qnom':0.9125, 'Pmax':1250e3, 'intervals':3} # pumps (both equal)
init_p1 = {'Q':[0]*T, 'H':[88]*T, 'n':[0.99]*T, 'Pe':[0]*T}
LinealizedPump(m.Pump4w, m.tw, data_p1, init_p1)
LinealizedPump(m.Pump4s, m.ts, data_p1, init_p1)
data_p1['eff']=0.922
LinealizedPump(m.Pump5w, m.tw, data_p1, init_p1)
LinealizedPump(m.Pump5s, m.ts, data_p1, init_p1)
# data_p1['eff']=0.9
# LinealizedPump(m.Pump6w, m.tw, data_p1, init_p1) #pump de reserva
# LinealizedPump(m.Pump6s, m.ts, data_p1, init_p1) 

#Pumps B0 to B1
data_p0 = {'A':300, 'B':62.208,'nmax':1, 'eff':0.922, 'Qmin':0.3, 'Qmax':1.1, 'Qnom':1.06, 'Pmax':3200e3, 'intervals':3} # pumps (both equal)
init_p0 = {'Q':[0]*T, 'H':[206]*T, 'n':[0.99]*T, 'Pe':[0]*T}
LinealizedPump(m.Pump7w, m.tw, data_p0, init_p0)
LinealizedPump(m.Pump7s, m.ts, data_p0, init_p0)
data_p0['eff']=0.922
LinealizedPump(m.Pump8w, m.tw, data_p0, init_p0)
LinealizedPump(m.Pump8s, m.ts, data_p0, init_p0)
# data_p0['eff']=0.9
# RealPump(m.Pump9w, m.tw, data_p0, init_p0)
# RealPump(m.Pump9s, m.ts, data_p0, init_p0)

data_t2 = {'eff':0.7, 'Pmax':1250e3,'H_approx':88} #B1 to B4
init_t2 = {'Q':[0]*T, 'H':[88]*T, 'Pe':[0]*T}
Turbine(m.Turb2w, m.tw, data_t2, init_t2)
Turbine(m.Turb2s, m.ts, data_t2, init_t2)

data_pv = {'Pinst':527.5e3, 'Pmax':527.5e3, 'forecast':df_meteo_jan['Irr'].head(T)/1000, 'eff':0.98} # PV
SolarPV(m.PVw, m.tw, data_pv)
data_pv = {'Pinst':527.5e3, 'Pmax':527.5e3, 'forecast':df_meteo_aug['Irr'].head(T)/1000, 'eff':0.98} # PV
SolarPV(m.PVs, m.ts, data_pv)

# data_bat = {'dt':3600, 'E0':0.05, 'Emax':200e3, 'Pmax':200e3, 'SOCmin':0.2, 'SOCmax':1.0, 'eff_ch':0.8, 'eff_dc':0.8,'Einst':0.1, 'Pinst':0}
# init_bat = {'E':[0]*T, 'P':[0]*T}
# NewBattery(m.Batw, m.tw, data_bat, init_bat)
# NewBattery(m.Bats, m.ts, data_bat, init_bat)

Grid(m.Gridw, m.tw, {'Pmax':6e6}) # grid
Grid(m.Grids, m.ts, {'Pmax':6e6}) # grid

EB(m.EBgw, m.tw)
EB(m.EBgs, m.ts)
# EB(m.EBpvw, m.tw) #Node electric: Pump 5
# EB(m.EBpvs, m.ts)

# def ConstraintBatEws(m):
#     return m.Batw.Edim == m.Bats.Edim
# m.c_BatEws = pyo.Constraint(rule=ConstraintBatEws)

# def ConstraintBatPws(m):
#     return m.Batw.Pdim == m.Bats.Pdim
# m.c_BatPws = pyo.Constraint(rule=ConstraintBatPws)

# m.Batw.Edim.fix(0)
# m.Bats.Edim.fix(0)
# m.Batw.Pdim.fix(0)
# m.Bats.Pdim.fix(0)

m.Turb2s.Pdim.fix(1250e3)
m.Turb2w.Pdim.fix(1250e3)

# m.PVw.Pdim.fix(0)

# Connections
m.p4r1w = Arc(ports=(m.Pump4w.port_Qin, m.B1w.port_Q), directed=True)
m.p4c2_Qw = Arc(ports=(m.Pump4w.port_Qout, m.Pipe2w.port_Q), directed=True)
m.p4c2_Hw = Arc(ports=(m.Pump4w.port_H, m.Pipe2w.port_H), directed=True)
m.p4ebw = Arc(ports=(m.Pump4w.port_P, m.EBgw.port_P), directed=True)

m.p5r1w = Arc(ports=(m.Pump5w.port_Qin, m.B1w.port_Q), directed=True)
m.p5c0_Qw = Arc(ports=(m.Pump5w.port_Qout, m.Pipe2w.port_Q), directed=True)
m.p5c0_Hw = Arc(ports=(m.Pump5w.port_H, m.Pipe2w.port_H), directed=True)
m.p5ebw = Arc(ports=(m.Pump5w.port_P, m.EBgw.port_P), directed=True) # pv node

# m.p6r1w = Arc(ports=(m.Pump6w.port_Qin, m.B1w.port_Q), directed=True)
# m.p6c0_Qw = Arc(ports=(m.Pump6w.port_Qout, m.Pipe2w.port_Q), directed=True)
# m.p6c0_Hw = Arc(ports=(m.Pump6w.port_H, m.Pipe2w.port_H), directed=True)
# m.p6ebw = Arc(ports=(m.Pump6w.port_P, m.EBgw.port_P), directed=True) # pv node

m.p7r0w = Arc(ports=(m.Pump7w.port_Qin, m.B0w.port_Q), directed=True)
m.p7c0_Qw = Arc(ports=(m.Pump7w.port_Qout, m.Pipe0w.port_Q), directed=True)
m.p7c0_Hw = Arc(ports=(m.Pump7w.port_H, m.Pipe0w.port_H), directed=True)
m.p7ebw = Arc(ports=(m.Pump7w.port_P, m.EBgw.port_P), directed=True)

m.p8r0w = Arc(ports=(m.Pump8w.port_Qin, m.B0w.port_Q), directed=True)
m.p8c2_Qw = Arc(ports=(m.Pump8w.port_Qout, m.Pipe0w.port_Q), directed=True)
m.p8c2_Hw = Arc(ports=(m.Pump8w.port_H, m.Pipe0w.port_H), directed=True)
m.p8ebw = Arc(ports=(m.Pump8w.port_P, m.EBgw.port_P), directed=True)

# m.p9r0w = Arc(ports=(m.Pump9w.port_Qin, m.B0w.port_Q), directed=True)
# m.p9c2_Qw = Arc(ports=(m.Pump9w.port_Qout, m.Pipe0w.port_Q), directed=True)
# m.p9c2_Hw = Arc(ports=(m.Pump9w.port_H, m.Pipe0w.port_H), directed=True)
# m.p9ebw = Arc(ports=(m.Pump9w.port_P, m.EBgw.port_P), directed=True)

m.t2r1w = Arc(ports=(m.Turb2w.port_Qout, m.B1w.port_Q), directed=True)
m.t2c2_Qw = Arc(ports=(m.Turb2w.port_Qin, m.Pipe2w.port_Q), directed=True)
m.t2c2_Hw = Arc(ports=(m.Turb2w.port_H, m.Pipe2w.port_H), directed=True)
m.t2ebw = Arc(ports=(m.Turb2w.port_P, m.EBgw.port_P), directed=True)

m.c0r1_Qw = Arc(ports=(m.Pipe0w.port_Q, m.B1w.port_Q), directed=True)
m.c0r1_zw = Arc(ports=(m.B1w.port_z, m.Pipe0w.port_zhigh), directed=True)
m.c0r0_zw = Arc(ports=(m.B0w.port_z, m.Pipe0w.port_zlow), directed=True)

m.c2r4_Qw = Arc(ports=(m.Pipe2w.port_Q, m.B4w.port_Q), directed=True)
m.c2r4_zw = Arc(ports=(m.B4w.port_z, m.Pipe2w.port_zhigh), directed=True)
m.c2r1_zw = Arc(ports=(m.B1w.port_z, m.Pipe2w.port_zlow), directed=True)

m.r4i4w = Arc(ports=(m.Irrigation4w.port_Qin, m.B4w.port_Q), directed=True) #Irrigation of B4
m.r1i1w = Arc(ports=(m.Irrigation1w.port_Qin, m.B1w.port_Q), directed=True) #Irrigation of B1

m.gridebw = Arc(ports=(m.Gridw.port_P, m.EBgw.port_P), directed=True)
m.pvebw = Arc(ports=(m.PVw.port_P, m.EBgw.port_P), directed=True) # pv node
# m.batebw = Arc(ports=(m.Batw.port_P, m.EBgw.port_P), directed=True) # pv node

# Connections

m.p4r1s = Arc(ports=(m.Pump4s.port_Qin, m.B1s.port_Q), directed=True) #Pump B1 to B4
m.p4c2_Qs = Arc(ports=(m.Pump4s.port_Qout, m.Pipe2s.port_Q), directed=True)
m.p4c2_Hs = Arc(ports=(m.Pump4s.port_H, m.Pipe2s.port_H), directed=True)
m.p4ebs = Arc(ports=(m.Pump4s.port_P, m.EBgs.port_P), directed=True)

m.p5r1s = Arc(ports=(m.Pump5s.port_Qin, m.B1s.port_Q), directed=True)
m.p5c2_Qs = Arc(ports=(m.Pump5s.port_Qout, m.Pipe2s.port_Q), directed=True)
m.p5c2_Hs = Arc(ports=(m.Pump5s.port_H, m.Pipe2s.port_H), directed=True)
m.p5ebs = Arc(ports=(m.Pump5s.port_P, m.EBgs.port_P), directed=True) # pv node

# m.p6r1s = Arc(ports=(m.Pump6s.port_Qin, m.B1s.port_Q), directed=True)
# m.p6c2_Qs = Arc(ports=(m.Pump6s.port_Qout, m.Pipe2s.port_Q), directed=True)
# m.p6c2_Hs = Arc(ports=(m.Pump6s.port_H, m.Pipe2s.port_H), directed=True)
# m.p6ebs = Arc(ports=(m.Pump6s.port_P, m.EBgs.port_P), directed=True) # pv node

m.p7r0s = Arc(ports=(m.Pump7s.port_Qin, m.B0s.port_Q), directed=True) #Pump B0 to B1 
m.p7c0_Qs = Arc(ports=(m.Pump7s.port_Qout, m.Pipe0s.port_Q), directed=True)
m.p7c0_Hs = Arc(ports=(m.Pump7s.port_H, m.Pipe0s.port_H), directed=True)
m.p7ebs = Arc(ports=(m.Pump7s.port_P, m.EBgs.port_P), directed=True)

m.p8r0s = Arc(ports=(m.Pump8s.port_Qin, m.B0s.port_Q), directed=True)
m.p8c0_Qs = Arc(ports=(m.Pump8s.port_Qout, m.Pipe0s.port_Q), directed=True)
m.p8c0_Hs = Arc(ports=(m.Pump8s.port_H, m.Pipe0s.port_H), directed=True)
m.p8ebs = Arc(ports=(m.Pump8s.port_P, m.EBgs.port_P), directed=True) # pv node

# m.p9r0s = Arc(ports=(m.Pump9s.port_Qin, m.B0s.port_Q), directed=True)
# m.p9c0_Qs = Arc(ports=(m.Pump9s.port_Qout, m.Pipe0s.port_Q), directed=True)
# m.p9c0_Hs = Arc(ports=(m.Pump9s.port_H, m.Pipe0s.port_H), directed=True)
# m.p9ebs = Arc(ports=(m.Pump9s.port_P, m.EBgs.port_P), directed=True) # pv node

m.t2r1s = Arc(ports=(m.Turb2s.port_Qout, m.B1s.port_Q), directed=True)
m.t2c2_Qs = Arc(ports=(m.Turb2s.port_Qin, m.Pipe2s.port_Q), directed=True)
m.t2c2_Hs = Arc(ports=(m.Turb2s.port_H, m.Pipe2s.port_H), directed=True)
m.t2ebs = Arc(ports=(m.Turb2s.port_P, m.EBgs.port_P), directed=True)

m.c0r1_Qs = Arc(ports=(m.Pipe0s.port_Q, m.B1s.port_Q), directed=True)
m.c0r1_zs = Arc(ports=(m.B1s.port_z, m.Pipe0s.port_zhigh), directed=True)
m.c0r0_zs = Arc(ports=(m.B0s.port_z, m.Pipe0s.port_zlow), directed=True)

m.c2r4_Qs = Arc(ports=(m.Pipe2s.port_Q, m.B4s.port_Q), directed=True)
m.c2r4_zs = Arc(ports=(m.B4s.port_z, m.Pipe2s.port_zhigh), directed=True)
m.c2r1_zs = Arc(ports=(m.B1s.port_z, m.Pipe2s.port_zlow), directed=True)

m.r4i4s = Arc(ports=(m.Irrigation4s.port_Qin, m.B4s.port_Q), directed=True) #Irrigation of B4
m.r1i1s = Arc(ports=(m.Irrigation1s.port_Qin, m.B1s.port_Q), directed=True) #Irrigation of B1

m.gridebs = Arc(ports=(m.Grids.port_P, m.EBgs.port_P), directed=True)
m.pvebs = Arc(ports=(m.PVs.port_P, m.EBgs.port_P), directed=True) # pv node
# m.batebs = Arc(ports=(m.Bats.port_P, m.EBgs.port_P), directed=True) # pv node

pyo.TransformationFactory("network.expand_arcs").apply_to(m) # apply arcs to model


#%% RUN THE OPTIMIZATION
# from pyomo.environ import value
import os
import time

# Objective function
def obj_fun(m):
    return sum((m.Gridw.Pbuy[t]*m.costw[t]/1e6 - m.Gridw.Psell[t]*m.excw[t]/1e6 + 
              m.Grids.Pbuy[t]*m.costs[t]/1e6 - m.Grids.Psell[t]*m.excs[t]/1e6)/2  for t in l_t ) # + (m.Batw.Pdim*cp_bat + m.Batw.Edim*ce_bat) #+ m.PV.Pdim*cost_new_pv
            
    # return sum( m.Grids.Pbuy[t]*m.costs[t]/1e6 - m.Grids.Psell[t]*m.excs[t]/1e6 for t in l_t ) + (m.Bats.Pdim*cp_bat + m.Bats.Edim*ce_bat) #+ m.PV.Pdim*cost_new_pv
    # return sum( m.Gridw.Pbuy[t]*m.costw[t]/1e6 - m.Gridw.Psell[t]*m.excw[t]/1e6 for t in l_t ) + (m.Batw.Pdim*cp_bat + m.Batw.Edim*ce_bat) #+ m.PV.Pdim*cost_new_pv

# return sum((m.Grid.Pbuy[t]*m.cost[t]/1e6 - m.Grid.Psell[t]*m.exc[t]/1e6) + 0*1/1e6*((m.PV.Pinst+m.PV.Pdim)*m.PV.forecast[t]*m.PV.eff + m.PV.P[t]) for t in l_t ) #+ (m.Bat.Pdim*cp_bat + m.Bat.Edim*ce_bat)/365/20#+ m.PV.Pdim*cost_new_pv
# 	return sum( m.Grids.Pbuy[t]*m.costs[t]/1e6 - m.Grids.Psell[t]*m.excs[t]/1e6 for t in l_t ) + (m.Bats.Pdim*cp_bat + m.Bats.Edim*ce_bat) #+ m.PV.Pdim*cost_new_pv
m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)

instance = m.create_instance()
# instance.write(filename='instance.nl', io_options={'symbolic_solver_labels':True})

# read_sol(m, instance, symbol_map_filename, suffixes=[".*"])

start_time = time.time()

# os.environ['NEOS_EMAIL'] = 'carla.cinto@upc.edu'
# solver_manager = pyo.SolverManagerFactory('neos')
# # results = solver_manager.solve(instancre, solver="knitro")
# results = solver_manager.solve(instance, solver="couenne")
# # results = solver_manager.solve(instance, solver="ipopt", options_string='max_iter=10000000')
# # results = solver_manager.solve(instance, solver="bonmin")
# # results = solver_manager.solve(instance, solver="minlp")

with open("bonmin.opt", "w") as file:
    file.write('''bonmin.algorithm B-Ecp
                bonmin.ecp_abs_tol 0.001
                bonmin.warm_start optimum
                tol 0.001
                ''')
                
# bonmin.warm_start optimum
solver = pyo.SolverFactory('bonmin')
results = solver.solve(instance, keepfiles= True, tee=True)
results.write()

# os.environ['NEOS_EMAIL'] = 'carla.cinto@upc.edu'
# solver_manager = pyo.SolverManagerFactory('neos')
# results = solver_manager.solve(instance, solver="knitro")
# results = solver_manager.solve(instance, solver="couenne")
# results = solver_manager.solve(instance, solver="ipopt", options_string='max_iter=10000000')
# results = solver_manager.solve(instance, solver="bonmin")
# results = solver_manager.solve(instance, solver="minlp")
# results.write()

# with open("couenne.opt", "w") as file:
#     file.write('''time_limit 100000
#                 convexification_cuts 2
#                 convexification_points 2
#                 delete_redundant yes
#                 use_quadratic no
#                 feas_tolerance 1e-1
#                 ''')
# solver = pyo.SolverFactory('asl:couenne')
# results = solver.solve(instance, tee=True)
# results.write()

# with open("couenne.opt", "w") as file:
#     file.write('''time_limit 100000
#                 convexification_cuts 2
#                 convexification_points 2
#                 delete_redundant yes
#                 use_quadratic no
#                 feas_tolerance 1e-1
#                 ''')
# solver = pyo.SolverFactory('asl:couenne')
# results = solver.solve(instance, tee=True)
# results.write()
# os.remove('couenne.opt') #Delete options

# solver = pyo.SolverFactory('asl:SCIP')
# results = solver.solve(instance, tee=True)
# results.write()

# instance = m.create_instance()
# solver = pyo.SolverFactory('ipopt')
# solver.options['max_iter'] = 1000000
# results = solver.solve(instance, tee=True)

exec_time = time.time() - start_time

#%%
def sol_read(filename, model):
    import pyomo.environ
    from pyomo.core import ComponentUID
    from pyomo.opt import ProblemFormat, ReaderFactory, ResultsFormat
    from pyomo.core.base.var import _GeneralVarData
    from pyomo.core import SymbolMap
    from six.moves import cPickle as pickle
    import pandas as pd

    """
    Reads a .sol solution file and returns a DataFrame with the variables.

    Parameters: 
        filename (str): Name of the file without extension. Note that all .nl, .sol, and .col must have the same name.
        model (pyomo.environ.AbstractModel): Pyomo model with variables and parameters.

    Returns:
        pd.DataFrame: DataFrame with variables and their values for each step time.
        str: Status description from the solver.
    """

    # Generating mapping file -
    def write_nl(model, nl_filename, **kwds):
        symbol_map_filename = nl_filename + ".symbol_map.pickle"
        _, smap_id = model.write(nl_filename, format=ProblemFormat.nl, io_options=kwds)
        symbol_map = model.solutions.symbol_map[smap_id]

        tmp_buffer = {}  # To speed up the process

        symbol_cuid_pairs = tuple(
            (symbol, ComponentUID(var, cuid_buffer=tmp_buffer))
            for symbol, var_weakref in symbol_map.bySymbol.items()
            if isinstance((var := var_weakref()), _GeneralVarData)  # Filter only variables
        )

        with open(symbol_map_filename, "wb") as f:
            pickle.dump(symbol_cuid_pairs, f)

        return symbol_map_filename
    
# Reading .sol file and returning results --- 
    def read_sol(model, sol_filename, symbol_map_filename, suffixes=[".*"]):
        if suffixes is None:
            suffixes = []

        with ReaderFactory(ResultsFormat.sol) as reader:
            results = reader(sol_filename, suffixes=suffixes)

        with open(symbol_map_filename, "rb") as f:
            symbol_cuid_pairs = pickle.load(f)

        symbol_map = SymbolMap()
        symbol_map.addSymbols((cuid.find_component(model), symbol) for symbol, cuid in symbol_cuid_pairs)
        results._smap = symbol_map

        return results

    # Reading the .col file to extract variable names
    def read_col_file(col_filename):
        with open(col_filename, "r") as col_file:
            variable_names = [line.strip() for line in col_file.readlines()]
        return variable_names

    # --- If Var not initialized, initialize to 0
    for v in model.component_objects(pyomo.environ.Var, active=True):
        for index in v:
            if v[index].value is None:
                v[index].set_value(0.0)  # Initialize variables to 0 if they have no value

    # 1. Reading symbol_map_filename
    nl_filename = filename + '.nl'
    col_filename = filename + '.col'
    symbol_map_filename = write_nl(model, nl_filename)

    # 2. Reading .sol
    sol_filename = filename + ".sol"
    symbol_map_filename = filename + ".nl.symbol_map.pickle"
    results = read_sol(model, sol_filename, symbol_map_filename)

    # Extract solver condition directly from results
    condition = results['Solver'][0]
    # 3. Reading variable names from .col file
    variable_names = read_col_file(col_filename)

    # 4. Reading the variable values from the .sol file
    variable_values = {}

    for solution in results['Solution']:
        for idx, (var, value) in enumerate(solution['Variable'].items()):
            if idx < len(variable_names):  # Ensure there is a mapping available
                real_var_name = variable_names[idx]  # Assign the correct name from .col
                variable_values[real_var_name] = value['Value']  # Store in dictionary
                
    #- 5. --- Read Parameter Values ---
    parameter_values = {}
    if "Parameter" in solution:
        for param, value in solution["Parameter"].items():
            parameter_values[param] = value["Value"]

    # 5. Organizing data for DataFrame
    organized_data = {}
    max_time_index = 0

    for var, value in variable_values.items():
        # Extract base variable name without temporal index
        if '[' in var and ']' in var:
            base_name = var.split('[')[0]
            time_index = int(var.split('[')[1].split(']')[0])
        else:
            base_name = var
            time_index = 0  # Non-temporal variable

        # Initialize variable list
        if base_name not in organized_data:
            organized_data[base_name] = []

        while len(organized_data[base_name]) <= time_index:
            organized_data[base_name].append(None)

        organized_data[base_name][time_index] = value

        max_time_index = max(max_time_index, time_index)

    # 6. Adjusting variable index
    for key in organized_data:
        while len(organized_data[key]) <= max_time_index:
            organized_data[key].append(None)

    # 7. Converting to data frame
    df = pd.DataFrame(organized_data)

    # Return the DataFrame and a clear condition message
    return df,condition

#%%
df, condition = sol_read('EB3/instance', m)

#%% Parameters

parameters = {}

for param in m.component_objects(pyo.Param, active=True):
    param_name = param.name
    parameters[param_name] = {}  # Store values in a dictionary
    for index in param:
        parameters[param_name][index]= param[index]
param_df = pd.DataFrame.from_dict(parameters, orient='index').T

#%% GET RESULTS
from Utilities import get_results

file = './results/SegriaSud/Lower/'
df_out, df_param, df_size = get_results(file=file+'/EB12', instance=instance, results=results, l_t=l_t, exec_time=exec_time)

#%% PLOTS
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Nimbus Roman No9 L"],
    "font.size": 9,
    'axes.spines.top': False,
    'axes.spines.right': False
})
labels_hours = ['0','','','','','','6','','','','','','12','','','','','','18','','','','','23']

cbcolors = sns.color_palette('colorblind')

file = 'EB12'

df_meteo_aug = pd.read_csv('data/meteo/LesPlanes_meteo_hour_aug.csv')
df_cons_aug = pd.read_excel('data/irrigation/EB5_irrigation_s.xlsx')
df_grid_aug = pd.read_csv('data/costs/PVPC_aug.csv')
df_B1cons_aug= pd.read_csv('data/irrigation/B1B2B3_irrigation_s.csv')

df_meteo_jan = pd.read_csv('data/meteo/LesPlanes_meteo_hour_jan.csv')
df_cons_jan = pd.read_excel('data/irrigation/EB5_irrigation_w.xlsx')
df_grid_jan = pd.read_csv('data/costs/PVPC_jan.csv')
df_B1cons_jan= pd.read_csv('data/irrigation/B1B2B3_irrigation_w.csv')



df_param = pd.read_csv('results/SegriaSud/Lower/'+file+'_param.csv')
w_lim = df_param['B4w.Wmin'][0]
w_lim = df_param['B4s.Wmin'][0]

df = pd.read_csv('results/SegriaSud/Lower/'+file+'.csv')
df['PVw.Pf'] = -df_meteo_jan['Irr']/1000*527.5e3*0.98
df['PVs.Pf'] = -df_meteo_aug['Irr']/1000*527.5e3*0.98


if 'Turb2w.Qout' in df.columns:
    df['Rev5s.Qout'] = df['Pump5s.Qout'] - df['Turb2s.Qout']
    df['Rev5s.Pe'] = df['Pump5s.Pe'] + df['Turb2s.Pe']
    df['Rev5w.Qout'] = df['Pump5w.Qout'] - df['Turb2w.Qout']
    df['Rev5w.Pe'] = df['Pump5w.Pe'] + df['Turb2w.Pe']
else:
    df['Rev5s.Qout'] = df['Pump5s.Qout']
    df['Rev5s.Pe'] = df['Pump5s.Pe']
    df['Rev5w.Qout'] = df['Pump5w.Qout']
    df['Rev5w.Pe'] = df['Pump5w.Pe']
    
# if 'Turb4w.Qout' in df.columns:
#     df['Rev4s.Qout'] = df['Pump4s.Qout'] - df['Turb4s.Qout']
#     df['Rev4s.Pe'] = df['Pump4s.Pe'] + df['Turb4s.Pe']
#     df['Rev4w.Qout'] = df['Pump4w.Qout'] - df['Turb4w.Qout']
#     df['Rev4w.Pe'] = df['Pump4w.Pe'] + df['Turb4w.Pe']
# else:
#     df['Rev4s.Qout'] = df['Pump4s.Qout']
#     df['Rev4s.Pe'] = df['Pump4s.Pe']
#     df['Rev4w.Qout'] = df['Pump4w.Qout']
#     df['Rev4w.Pe'] = df['Pump4w.Pe']

df['Grid14s.Pe'] = - df['PVs.P'] - df['Rev5s.Pe'] -df['Pump4s.Pe']
df['Grid14w.Pe'] = - df['PVw.P'] - df['Rev5w.Pe'] -df['Pump4w.Pe']

df['Irrigation1s.Qin'] = -df['Irrigation1s.Qout'] # no se per que passa pero Qin a vegades es reseteja a 0. La resta tot be
df['Irrigation4w.Qin'] = -df['Irrigation4w.Qout']

df['Irrigation1s.Qin'] = -df['Irrigation1s.Qout'] # no se per que passa pero Qin a vegades es reseteja a 0. La resta tot be
df['Irrigation4w.Qin'] = -df['Irrigation4w.Qout']

for c in df.columns:
    df[c] = df[c].apply(lambda x: 0 if (x>-1e-5 and x<1e-5) else x)

df_W = df[[col for col in df.columns if 'w.' in col]]
df_S = df[[col for col in df.columns if 's.' in col]]
df_W['t'] = df_W.index
df_S['t'] = df_S.index
df_W['PVPC'] = df_grid_jan['PVPC']
df_S['PVPC'] = df_grid_aug['PVPC']
df_W['Excedentes'] = df_grid_jan['Excedentes']
df_S['Excedentes'] = df_grid_aug['Excedentes']

df_W.columns = [col.replace('w.','.') for col in df_W.columns]
df_S.columns = [col.replace('s.','.') for col in df_S.columns]

#%%
i=0
# season=['S','W']
season=['S']
# for df in [df_S, df_W]:
for df in [df_S]:
    fig = plt.figure(figsize=(3.4, 2.2))
    plt.rcParams['axes.spines.right'] = True
    ax1 = fig.add_subplot(1,1,1)
    df.apply(lambda x: x/1000).plot(y=['PV.Pf'], kind='bar', ax=ax1, stacked=False, ylabel='P (kW)', 
                                color='#C0E3C0', alpha=1, edgecolor=None)
    df.apply(lambda x: x/1000).plot(y=['Rev5.Pe','Pump4.Pe','PV.P','Grid14.Pe'], kind='bar', stacked=True, ax=ax1, ylabel='P (kW)',
                                color=[cbcolors[1],cbcolors[4], cbcolors[2], cbcolors[7]], edgecolor=None)
    # df.apply(lambda x: x/1000).plot(y=['Grid.P','Rev4.Pe'], kind='bar', stacked=True, ylabel= 'P(kW)', ax=ax1, color= [cbcolors[7],cbcolors[0]])
    ax1.axhline(0,color='k') 
    # ax1.set_ylim(-200,200)
    bars = ax1.patches
    patterns =(None, None, None, None)
    hatches = [p for p in patterns for i in range(len(df))]
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax1.legend(['_','$\hat{P}_{PV}$','$P_{rev5}$','$P_{pump4}$','$P_{PV}$','$P_{Grid}$'],loc='upper center', bbox_to_anchor=(0.5, -0.2),
        fancybox=False, shadow=False, ncol=3)
    
    ax2 = plt.twinx()
    ax2.set_ylim(45,300)
    sns.lineplot(df,x='t' ,y='PVPC', ax=ax2, color='tab:red')#, label='Buy')
    sns.lineplot(df,x='t' ,y='Excedentes', ax=ax2, color='tab:red', linestyle='dashed')#, label='Buy')
    ax2.set_xticks(range(24), labels=labels_hours, rotation=90)
    ax2.text(24.5,0.1,'Time (h)')
    
    plt.ylabel('Price (€/MWh)')
    plt.xlabel('Hour')
    # plt.title('Power consumption')
    plt.subplots_adjust(left=0.21, right=0.8, top=0.97, bottom=0.42)
    plt.show()
    
#%%
    # plt.rcParams['savefig.format']='pdf'
    # plt.savefig('results/EB3/Proves/T5' + file + season[i] + '_P', dpi=300)
    # plt.rcParams['savefig.format']='svg'
    # plt.savefig('results/EB3/Proves/T5' + file + season[i] + '_P', dpi=300)
    
     
    fig = plt.figure(figsize=(3.4, 1.9))
    plt.rcParams['axes.spines.right'] = False
    gs = gridspec.GridSpec(2,1,height_ratios=[2,1])
    
    ax1 = fig.add_subplot(gs[0])
    ax1.set_ylim(-0.5,0.5)
    df.plot(y=['Pump2.Qout','Irrigation5.Qin','Irrigation1.Qin','Rev1.Qout'], kind='bar', stacked=True, ax=ax1, ylabel='Q (m$^3$/s)', 
            color=[cbcolors[0],cbcolors[2],cbcolors[6],cbcolors[1]], edgecolor=None)
    #sns.lineplot(df_cons_jan, x='Hour' ,y=-df_cons_jan['Qirr']/3600, color='tab:red')
    # sns.lineplot(df_cons_aug, x='Hour' ,y=-df_cons_aug['Qirr']/3600, ax=ax1, color='tab:red')
    bars = ax1.patches
    patterns =(None,None,None)
    hatches = [p for p in patterns for i in range(len(df))]
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    # ax1.legend(['$Q_{p,PV}$', '$Q_{irr}$', '$Q_{p,g}$'], ncols=3, loc='upper center', bbox_to_anchor=(0.5, -0.88),
          # fancybox=False, shadow=False)
    ax1.axhline(0,color='k')
    ax1.set_xticks(range(24), labels=['']*24, rotation=90)
    ax1.tick_params(axis='x', labelbottom='off')
    
    fig = plt.figure(figsize=(3.4, 1.9))
    plt.rcParams['axes.spines.right'] = False
    gs = gridspec.GridSpec(2,1,height_ratios=[2,1])       
    ax1 = fig.add_subplot(gs[1],sharex=ax1)
    ax1.figsize=(3.4,2)
    ax1.set_ylim(160000*0.9,185814)
    ax1.set_yticks([160e3*0.9,160e3,185814])
    ax1.axhline(w_lim,color='#AFAFAF', alpha=1, linewidth=1)
    ax1.axhline(160000,color='#AFAFAF', alpha=1, linewidth=1)
    ax1.axhline(160e3*0.95, color='#AFAFAF', linestyle='--', alpha=1, linewidth=1)
    ax1.axhline(160e3*1.05, color='#AFAFAF', linestyle='--', alpha=1, linewidth=1)
    sns.lineplot(df_S, x='t', y='B5.W', ax=ax1, color='tab:red', linewidth=1.5)
    # sns.lineplot(B5W_dt, x='Hour', y='mean', ax=ax1, color='blue', linestyle='--', linewidth= 1.5)
    ax1.set_xticks(range(24), labels=labels_hours, rotation=90)
    ax1.text(1.02,-0.45,'Time (h)', transform=ax1.transAxes)
    plt.show()
    
    # fig = plt.figure(figsize=(3.4, 1.9))
    # plt.rcParams['axes.spines.right'] = False
    # gs = gridspec.GridSpec(2,1,height_ratios=[2,1])       
    # ax1 = fig.add_subplot(gs[1],sharex=ax1)
    # ax1.figsize=(3.4,2)
    # ax1.set_ylim(124000*0.9,185814)
    # ax1.set_yticks([124e3*0.9,124e3,185814])
    # ax1.axhline(w_lim,color='#AFAFAF', alpha=1, linewidth=1)
    # ax1.axhline(124000,color='#AFAFAF', alpha=1, linewidth=1)
    # ax1.axhline(124e3*0.95, color='#AFAFAF', linestyle='--', alpha=1, linewidth=1)
    # ax1.axhline(124e3*1.05, color='#AFAFAF', linestyle='--', alpha=1, linewidth=1)
    # sns.lineplot(df_W, x='t', y='B5.W', ax=ax1, color='tab:red', linewidth=1.5)
    # # sns.lineplot(B5W_dt, x='Hour', y='mean', ax=ax1, color='blue', linestyle='--', linewidth= 1.5)
    # ax1.set_xticks(range(24), labels=labels_hours, rotation=90)
    # ax1.text(1.02,-0.45,'Time (h)', transform=ax1.transAxes)
    
    
    df.plot(y=['Pump2.Qout','Irrigation5.Qin','Rev1.Qout'], kind='line', stacked=True, ax=ax1, ylabel='Q (m$^3$/s)',
            color=[cbcolors[0],cbcolors[2],cbcolors[1]], edgecolor=None)
    plt.show()
    
    # # plt.ylabel('$W_{R1}$ (m$^3$)')
    # plt.xlabel(None)
    # plt.setp(ax1.get_xticklabels(), visible=False)
    # # plt.xticks(range(24),labels=labels_hours,rotation=90)
    # plt.subplots_adjust(left=0.18, right=0.84, top=0.97, bottom=0.26)
    # plt.show()

    # # plt.rcParams['savefig.format']='pdf'
    # # plt.savefig('results/ISGT/' + file + season[i] + '_Q', dpi=300)
    # # plt.rcParams['savefig.format']='svg'
    # # plt.savefig('results/ISGT/' + file + season[i] + '_Q', dpi=300)
    
    
    # i=i+1



# df = pd.read_csv('data/Analogiques.csv')
# df['date'] = pd.to_datetime(df['Fecha'], format='%Y/%m/%d')
# df['month'] = df['date'].dt.month
# df['weekday'] = df['date'].dt.dayofweek
# df['Volum bassa'] = df['Nivell bassa']*13000/4.5
# df['Qdiff'] = df['Volum bassa'] - df['Volum bassa'].shift(1) 
# df['Qreg'] = df['Cabal impulsio'] - df['Qdiff']
# df['Qreg'] = df['Qreg'].apply(lambda x: 0 if x<0 else x)
# df['Qreg_lag'] = df['Qreg'].shift(1)
# df.loc[df['Qreg']>300,'Qreg'] = df['Qreg_lag']

# df['Season'] = df['month']/4
# df['Season'] = df['Season'].astype(int)
# fig = plt.figure(figsize=(3.4, 2.3))
# sns.lineplot(df, x='hour', y='Qreg', style='Season', color='tab:grey')
# plt.legend(labels=['Winter','_','Spring','_','Summer','_','Fall','_'], ncol=1, loc='upper right')
# plt.xlabel('Hour')
# plt.ylabel('Irrigation demand (m$^3$/h)')
# plt.xticks(range(24),labels=labels_hours, rotation=90)
# plt.xlim([0,24])
# plt.ylim([0,250])
# plt.tight_layout()
# plt.subplots_adjust(left=0.15, right=1, top=0.95, bottom=0.18)
# plt.show()
# plt.rcParams['savefig.format']='pdf'
# plt.savefig('results/ISGT/Irrigation_season', dpi=300)

# df['Winter'] = df['date'].apply(lambda x: 0 if (x.month in [3,4,5,6,7,8]) else 1)
# fig = plt.figure(figsize=(3.4, 1.1))
# sns.lineplot(df, x='hour', y='Qreg', style='Winter', hue='Winter', palette=[cbcolors[1],cbcolors[0]], errorbar=('ci',90))
# plt.legend(labels=['S','_','W','_'], ncol=1)
# plt.text(24.5,-50,'Time (h)')
# plt.xlabel(None)
# plt.ylabel('Demand (m$^3$/h)')
# plt.xticks(range(24),labels=labels_hours, rotation=90)
# plt.yticks([0,100,200])
# # plt.ylim([0,200])
# plt.xlim([0,24])
# plt.tight_layout()
# plt.subplots_adjust(left=0.15, right=0.84, top=0.92, bottom=0.22)
# plt.show()
# plt.rcParams['savefig.format']='pdf'
# plt.savefig('results/ISGT/Irrigation_WS', dpi=300)
# plt.rcParams['savefig.format']='svg'
# plt.savefig('results/ISGT/Irrigation_WS', dpi=300)


#%%
logging.INFO = 2000
log_infeasible_constraints(instance)

# with open('eq', 'w') as f:
#     instance.pprint(f)