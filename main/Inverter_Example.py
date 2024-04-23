# Import pyomo
import pyomo.environ as pyo
from pyomo.network import Arc, Port
from Devices.SolarPV import SolarPV
from Devices.MainGrid import Grid
from Devices.EB import EB
from Devices.Inverter import Inverter

m = pyo.ConcreteModel()

# time
l_t = list(range(5))
m.t = pyo.Set(initialize=l_t)

# electricity cost
l_cost = [10, 5, 1, 5, 10]
m.cost = pyo.Param(m.t, initialize=l_cost)
cost_new_pv = 2

# ===== Create the system =====

m.PV = pyo.Block()
m.Grid = pyo.Block()
m.EB = pyo.Block()
m.INV_grid = pyo.Block()
m.INV_bat = pyo.Block()
m.INV_pump = pyo.Block()
m.INV_pv = pyo.Block()

data_pv = {'Pinst': 150e3, 'Pmax': 500e3, 'forecast': [0.0, 0.4, 0.8, 1.0, 0.1], 'eff': 0.2}  # PV
SolarPV(m.PV, m.t, data_pv)

Grid(m.Grid, m.t, {'Pmax': 100e3})  # grid

EB(m.EB, m.t)

# DataSets
# Stützpunkte und Pcondloss und Pswitchloss für piecewise linear Funktion
data_inv_grid = {'Pmax': 500e3,
                 'x_cond_loss':     [-500.0e3, -300.0e3, -100.0e3, 0, 100e3, 300.0e3, 500.0e3],
                 'y_cond_loss':     [3.03e3, 1.5e3, 0.5e3, 0, 0.5e3, 1.5e3, 3.03e3],
                 'x_switch_loss':   [-500.0e3, -300.0e3, -100.0e3, 0, 100e3, 300.0e3, 500.0e3],
                 'y_switch_loss':   [3.19e3, 2.3e3, 1.5e3, 0, 1.5e3, 2.3e3, 3.19e3]}

data_inv_bat = {'Pmax': 270e3,
                 'x_cond_loss':     [-270e3, -200.0e3, -100.0e3, 0, 100e3, 200.0e3, 270e3],
                 'y_cond_loss':     [0.8e3, 0.57e3, 0.25e3, 0, 0.25e3, 0.57e3, 0.8e3],
                 'x_switch_loss':   [-270.0e3, -2.70e3,  0, -2.70e3, 270.0e3],
                 'y_switch_loss':   [   2.8e3,  0.38e3,  0,  0.38e3,  2.8e3]}

data_inv_pv = {'Pmax': 270e3,
                 'x_cond_loss':     [-270e3, -200.0e3, -100.0e3, 0, 100e3, 200.0e3, 270e3],
                 'y_cond_loss':     [     0,        0,        0, 0, 0.27e3, 0.62e3, 0.91e3],
                 'x_switch_loss':   [-270e3, -200.0e3, -100.0e3, 0,       100e3, 200.0e3, 270e3],
                 'y_switch_loss':   [     0,        0,        0, 0.23,   0.58e3,  0.79e3, 0.95e3]}

data_inv_pump = {'Pmax': 270e3,
                 'x_cond_loss': [-170.0e3, -100.0e3, -34.0e3, -17.0e3, 0],
                 'y_cond_loss': [1.21e3, 1.24e3, 1.84e3, 0.52e3, 0],
                 'x_switch_loss': [-170.0e3, 34.0e3, -1.70e3, 0],
                 'y_switch_loss': [1.86e3, 1.84e3, 1.05e3, 0.7]}
# Battery

# Pump


Inverter(m.INV_grid, m.t, data_inv_grid)
#Inverter(m.INV_bat, m.t, data_inv_grid)
#Inverter(m.INV_pump, m.t, data_inv_grid)
#Inverter(m.INV_pv, m.t, data_inv_grid)

m.grid_to_inv = Arc(ports=(m.Grid.port_P, m.INV_grid.port_Pin), directed=True)
m.pv_to_inv = Arc(ports=(m.PV.port_P, m.INV_grid.port_Pout), directed=True)

#m.gridinv = Arc(ports=(m.Grid.port_P, m.EB.port_P), directed=True)
#m.PVinv = Arc(ports=(m.PV.port_P, m.EB.port_P), directed=True)
#m.eb1 = Arc(ports=(m.INV_grid.port_P, m.EB.port_P), directed=True)
#m.eb2 = Arc(ports=(m.INV_grid.port_P, m.EB.port_P), directed=False)



# PV -> INVERTER -> GRID
pyo.TransformationFactory("network.expand_arcs").apply_to(m)  # apply arcs to model


def obj_fun(m):
    return sum((m.Grid.Pbuy[t] * m.cost[t] - m.Grid.Psell[t] * m.cost[t] / 2) for t in l_t) + m.PV.Pdim * cost_new_pv


m.goal = pyo.Objective(rule=obj_fun, sense=pyo.minimize)

instance = m.create_instance()
solver = pyo.SolverFactory('ipopt')
solver.solve(instance, tee=False)

instance.Grid.P.pprint()
instance.PV.P.pprint()
# instance.INV.Pout.pprint()
# instance.INV.Pin.pprint()
