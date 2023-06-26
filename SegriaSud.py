# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 17:37:52 2023

@author: colives
"""

import SystClass

# converts B from Q of m3/h to m3/s
def B_to_SI(B):
    return B*3600**2


# System definition
sgr_sud = SystClass.system(9.81e3)

# Reservoirs definition
sgr_sud.add_rsvr(142869, 142869, 142869*0.1, 335.50, 328) # Bassa 1
sgr_sud.add_rsvr(85268*0.5, 85268, 85268*0.1, 423.50, 414) # Bassa 3
# sgr_sud.add_rsvr(34, 50, 20)
# sgr_sud.add_rsvr(35, 50, 20)

# EBs definition
sgr_sud.add_EB(100e6)

# Pipes definition
sgr_sud.add_pipe(0, 95, 0, 1)
# sgr_sud.add_pipe(0, 95, 0, 1)
# sgr_sud.add_pipe(20, 5e-5, 0, 1)
# sgr_sud.add_pipe(20, 5e-5, 0, 1)

# Pumps definition
sgr_sud.add_pump(106.3, B_to_SI(2e-5), 275.9e3, (960/3600,1025/3600,1160/3600), 1480, 0.85, 0, 0)
# sgr_sud.add_pump_simple(25e4, 50, 0.5, 0, 0)
# sgr_sud.add_pump_simple(25e4, 2, 0.9, 0, 0)
# sgr_sud.add_turbine_simple(0.1e6, 2000, 0.8, 1, 0)
# Pump to dim
# sgr_sud.add_new_pump(2, 0)

# PVs definition
# sgr_sud.add_PV(0, 5e5, 0)

# Batteries definition
#sgr_sud.add_battery(0, 1e3)

sgr_sud.builder(solver='mindtpy')
reslt = sgr_sud.run()
    
print()
print(f'Potencia xarxa EB0: {reslt["P_g0"]}')
print(f'Potencia el bomba: {reslt["Pe_b0"]}')
print(f'Rend bomba: {reslt["nu_b0"]}')
print(f'Cabal bomba (m3/h): {[i*3600 for i in reslt["Q_b0"]]}')
print(f'H bomba: {reslt["H_b0"]}')
print(f'rpm bomba: {reslt["rpm_0"]}')
# print(f'Potencia turbina: {reslt["P_trb0"]}')
# print(f'Potencia PV: {reslt["P_pv_g0"]}')

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