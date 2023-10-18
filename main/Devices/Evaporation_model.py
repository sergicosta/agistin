
#https://www.agraria.unirc.it/documentazione/materiale_didattico/1462_2016_412_24509.pdf
#https://www.scielo.br/j/eagri/a/gBhrCcJRCHdLMsyjRsdedXRGRy/?lang=en&format=pd
 
"""
Implementation model of evaporation and rainfall of the water in the reservoirs.

Evaporation:
    - It's used the Penman Method as an aproximation of the evaporation in l/m^2 per t
    - In the examples calculate the evaporation per hour during one day 
    of "h" hours of sun.

    - Requires:
        - Net radiance [W/m^2]
        - Wind speed [m/s]
        - Relative humidity [%]
        - Atmospheric pressure [kPa]
        -Temperature [ºC]
        
Rainfall:
    - It requieres the measure of rainfall in [l/m^2]

Once calculated the rainfall and evaporation per an hour, it's defined 
a variable of water flow, that can be implemented in the reservoir volume
expression

    

"""

import matplotlib.pyplot as plt
import pandas as pd

path = 'Data.xlsx'

data = pd.read_excel(path,sheet_name = 'Hoja2')

wnd = data['VIENTO']
Temp = data['TEMPERATURA']
Rdt = data['RADIACION']
Press = data['PRESION']
Humid = data['HUMEDAD']
rain = data['PRECIPITACION']

latent = 2256/1000
A = 1000
hsum = 0   
i = 0

#Contador de hores de sol segons la radiació
for i in Rdt:
    if hsum >=14:
        break
    if i >= 50:
        hsum = hsum +1
        i = i + 1

h=hsum

def slope(Temp):
    slp = []
    for T in Temp:
        slope = (4098*(0.06108*2.718**(17.27*T/(T+273.3))))/(T+273.3)**2
        slp.append(slope)
    return slp
slp = slope(Temp)


def act_sat_vapor(Temp,Humid):
    ea = []
    for T,rh in zip(Temp,Humid):
        act_vapor = (rh/100)*0.6108*2.718**(17.27*T/(T+273.3))
        ea.append(act_vapor)
    return ea
ea = act_sat_vapor(Temp,Humid)

#Mean saturate vapor (STP10)
def mean_sat_vapor(Temp):
    es = []
    for T in Temp:
        ess = 0.6108*2.718**(17.27*T/(T+273.3))
        es.append(ess)
    return es
es = mean_sat_vapor(Temp) 

#Psychometric constant (STP6) P in kPa
def psychct(Press):
    psy = []
    for P in Press:
        pssyy = 0.000665*P/10
        psy.append(pssyy)
    return psy
psy = psychct(Press)


def diff_ea_es (ea,es):
    diff = []
    for EA,ES in zip(ea,es):
        ea_es = EA-ES
        diff.append(ea_es)
    return diff
diff = diff_ea_es(ea,es)


def Evaporation(slp,psy,Temp,ea,es,wnd,Rdt):
    evaporation = []
    for s,p,t,dff,w,r in zip (slp,psy,Temp,diff,wnd,Rdt):
        evap = 1/h*1/997*(86.4*(s/(s+p))*(r/latent)+((p/(p+s))*0.26*(0.5+0.54*w)*dff*10))
        
        evaporation.append(evap)
    return evaporation
evaporation = Evaporation(slp,psy,Temp,ea,es,wnd,Rdt)

print(evaporation)

plt.figure(1)
plt.plot(evaporation)

total = 0
for i in evaporation:
    total = total+i
print(total)


We = (rain-evaporation)*A

    
print(We)



       
       
        
        