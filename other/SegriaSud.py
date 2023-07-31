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

# EBs definition
sgr_sud.add_EB(100e6)
# sgr_sud.add_EV(0,100e3)

# Pipes definition
sgr_sud.add_pipe(5e-3, 95, 0, 1)

# Pumps definition
sgr_sud.add_pump(106.3, B_to_SI(2e-5), 275.9e3, (960/3600,1025/3600,1160/3600), 1480, 0.85, 0, 0)
sgr_sud.add_pump(106.3, B_to_SI(2e-5), 275.9e3, (960/3600,1025/3600,1160/3600), 1480, 0.85, 0, 0)
sgr_sud.add_pump(200, B_to_SI(2e-5), 275.9e3, (960/3600,1025/3600,1160/3600), 1480, 0.85, 0, 0)
# sgr_sud.add_pump_simple(25e4, 50, 0.5, 0, 0)
# sgr_sud.add_pump_simple(25e4, 2, 0.9, 0, 0)
# sgr_sud.add_turbine_simple(0.1e6, 2000, 0.8, 1, 0)
# Pump to dim
# sgr_sud.add_new_pump(0, 0)

# PVs definition
sgr_sud.add_PV(0, 523e3, 0)
# sgr_sud.add_PV(0, 300e3, 0)

# Batteries definition
#sgr_sud.add_battery(0, 1e3)

sgr_sud.builder(solver='mindtpy')
reslt = sgr_sud.run()
    
print()
print(f'Potencia xarxa EB0: {reslt["P_g0"]}')
print(f'Potencia el bomba: {reslt["Pe_b0"]}')
print(f'Rend bomba: {reslt["nu_b0"]}')
print(f'Cabal bomba (m3/h): {[i*3600 for i in reslt["Q_b0"]]}')
print(f'Cabal pipe (m3/h): {[i*3600 for i in reslt["Q_p0"]]}')
# print(f'Cabal bomba (m3/h): {[i*3600 for i in reslt["Q_b1"]]}')
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

#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

import json
with open("ext_data.json","r") as file:
	ext_data=json.load(file)

ext_data.pop('cost_pv_inst')

df = pd.DataFrame(ext_data)
df['t'] = df.index

for k in ['P_g0', 'P_pv_g0', 'Pe_b0', 'Pe_b1', 'W_r0', 'W_r1']:
 df = pd.concat([df, pd.DataFrame(reslt[k], columns=[k])], axis=1)


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif"
})

fig=plt.figure(figsize=(11.69, 8.27))
ax=sns.barplot(data=df, x='t', y='cost_elec', color='tab:green')
ax.axhline(y=0, color='black', linewidth=1)
plt.title('Electricity cost')
plt.ylabel('$C_{elec}$ (m.u./Wh)')
plt.xlabel('Time')
plt.tight_layout()

fig=plt.figure(figsize=(11.69, 8.27))
ax=sns.barplot(data=df, x='t', y='P_g0', color='tab:red')
ax.axhline(y=0, color='black', linewidth=1)
plt.title('Grid power (W)')
plt.ylabel('$P_{g}$ (W)')
plt.xlabel('Time')
plt.tight_layout()

fig=plt.figure(figsize=(11.69, 8.27))
ax=sns.barplot(data=df, x='t', y='P_pv_g0', color='tab:blue')
ax.axhline(y=0, color='black', linewidth=1)
plt.title('PV power (W)')
plt.ylabel('$P_{PV}$ (W)')
plt.xlabel('Time')
plt.tight_layout()

fig=plt.figure(figsize=(11.69, 8.27))
ax=sns.barplot(data=df, x='t', y='weather_0', color='tab:blue')
ax.axhline(y=0, color='black', linewidth=1)
plt.title('Irradiation')
plt.ylabel('$f_{PV}$ (pu)')
plt.xlabel('Time')
plt.tight_layout()



fig = plt.figure(figsize=(11.69, 8.27))
gs=GridSpec(2,1)

ax = fig.add_subplot(gs[0])
ax.axhline(y=142869, color='black', linewidth=1)
ax=sns.lineplot(data=df, x='t', y='W_r0', color='tab:blue', label='Res_0', marker='o',markersize=5)
ax.axhline(y=142869*0.1, color='black', linewidth=1)
ax.set_ylabel('W (m$^3$)')
ax.set_xlabel('Time')
ax.set_xticks(range(5))
ax.set_xticklabels(['1','2','3','4','5'])

ax = fig.add_subplot(gs[1])
ax.axhline(y=85268, color='black', linewidth=1)
ax2=sns.lineplot(data=df, x='t', y='W_r1', color='tab:red', label='Res_1', marker='o',markersize=5)
ax.axhline(y=85268*0.1, color='black', linewidth=1)
ax.axhline(y=0, color='black', linewidth=1)
ax.set_ylabel('W (m$^3$)')
ax.set_xlabel('Time')
ax.set_xticks(range(5))
ax.set_xticklabels(['1','2','3','4','5'])
plt.tight_layout()


