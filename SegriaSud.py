# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 17:37:52 2023

@author: colives
"""

import SystClass

# System definition
sgr_sud = SystClass.system(9.81e3)

# Reservoirs definition
sgr_sud.add_rsvr(58, 90, 30)
# sgr_sud.add_rsvr(34, 50, 20)
sgr_sud.add_rsvr(35, 50, 20)

# EBs definition
sgr_sud.add_EB(100e6)

# Pipes definition
sgr_sud.add_pipe(23, 5e-5, 0, 1)
# sgr_sud.add_pipe(20, 50, 0, 1)
# sgr_sud.add_pipe(20, 50, 0, 1)

# Pumps definition
# PDH-800-90-M (A)
sgr_sud.add_pump(106.3, 3e-5, 272.13e3, 1035.21, 1480, 0.87, 0, 0)
# sgr_sud.add_pump_simple(25e4, 50, 0.5, 0, 0)
# sgr_sud.add_pump_simple(25e4, 2, 0.9, 0, 0)
# sgr_sud.add_turbine_simple(25e6, 2000, 0.5, 1, 0)
# Pump to dim
# sgr_sud.add_new_pump(2, 0)

# PVs definition
sgr_sud.add_PV(0, 5e5, 1e4)

# Batteries definition
#sgr_sud.add_battery(0, 1e3)

sgr_sud.builder(solver='mindtpy')
reslt = sgr_sud.run()
    
print()
print(f'Potencia xarxa EB0: {reslt["P_g0"]}')
print(f'Potencia bomba: {reslt["Ph_b0"]}')
# print(f'Potencia turbina: {reslt["P_trb0"]}')
print(f'Potencia PV: {reslt["P_pv_g0"]}')

# for i in sgr_sud.rsvrs.keys():
#     print(sgr_sud.rsvrs[i].x)
#     for k in sgr_sud.rsvrs[i].var:
#         print(k)
#     print('\n')

# for i in sgr_sud.pumps.keys():
#     print(sgr_sud.pumps[i].x)
#     for k in sgr_sud.pumps[i].var:
#         print(k)
#     sgr_sud.pumps[i].eq_write()
#     for k in sgr_sud.pumps[i].eqs:
#         print(k)