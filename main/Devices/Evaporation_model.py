#doi:10.1016/j.jhydrol.2006.06.012
"""
Implementation model of evaporation in reservoirs.


    - It's used a simplified version of the Penman Method by Jhon D. Valiantzas 2006
    - The method use routine weahter data for the calculation of the evaporation [mm/day]

    - Requires:
        - Radiance [W/m^2]
        - Wind speed [m/s]
        - Relative humidity [%]
        - Temperature [ºC]
        - Max Temperature [ºC]
        - Min Temperature [ºC]
        - Julian Day
        
"""

import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np

file = 'WI_Maials.xlsx'
path = 'C:/Users/paugm/OneDrive/Documentos/Universitat/CITCEA/Dades Metereologiques/'
df = pd.read_excel(path+file)
df = df.drop(df.columns[[0,1,2,3,4,5,11,12,13,17,18,19]], axis=1)


df['DATA_Hora Inicial (T.U.)'] = pd.to_datetime(df['DATA_Hora Inicial (T.U.)']).dt.date
df = df.groupby('DATA_Hora Inicial (T.U.)').mean()
df.index = pd.to_datetime(df.index)

T = df['T mitjana (ºC)']                            #In ºC
Rs = df['Irradiància solar global (W/m2)']*0.0864   #Set to MJ/m2/day
RH = df['Humitat Relativa mitjana (%)']             #In % 
u = df['Mitjana velocitat vent a 2 m (m/s)']        #m/s
A = 3500                                            #m2
P = 101.9                                           #kPa
a = 0.08
Tmax = df['TMAX (ºC)']
Tmin = df['TMIN (ºC)']
df['J'] = df.index.dayofyear
J = df['J']
latitude = 41.17043116361487
latitude = np.radians(latitude)


#Valiantzas  2006

import seaborn as sns
import matplotlib.pyplot as plt


def declination (J):
    return 0.409*np.sin(2*np.pi/365*J-1.39)         #EQ 52 Valiantzas2006
declination = declination(J)        

def hourangle (declination,latitude):
    return np.arccos(-np.tan(latitude)*np.tan(declination)) #EQ 51 Valiantzas2006
ws = hourangle(declination,latitude)

def ExtraRadiation (J,latitude):
    return 37.59*(1+0.033*np.cos(2*3.14/365*J))*(ws*np.sin(latitude)*np.sin(declination)+np.sin(ws)*np.cos(latitude)*np.cos(declination)) #EQ 54 Valiantzas2006

Ra = ExtraRadiation(J,latitude)

def Evaliantzas(T,Tmax,Tmin,Rs,Ra,a,RH,u):      #EQ 31 Valiantzas2006
    return 0.051*(1-a)*Rs*np.sqrt(T+9.5)-0.188*(T+13)*(Rs/Ra-0.194)*(1-0.00014*(0.7*Tmax+0.3*Tmin+46)**2*np.sqrt(RH/100))+0.049*(Tmax+16.3)*(1-RH/100)*(0.5+0.536*u)

df['E_Valiantzas']= Evaliantzas(T,Tmax,Tmin,Rs,Ra,a,RH,u)*A/1000    #liters/day

df['Año'] = df.index.year
df['Mes'] = df.index.month

plt.figure(1)
sns.lineplot(data=df,x='Mes', y = 'E_Valiantzas')
plt.title('Evaporation per day during a year by Valiantzas2006')
plt.xlabel('Month')
plt.ylabel('Evaporation [l/day]')

#%% Jensen 2010 !!!NOT WORKING!!!)

def es(T):
    return 0.6108*np.exp(17.37*T/(T+273))           #Zotarelli2015

es = es(T)

def slope(es,T):
    return (4098*es)/(T + 273.3)**2                 #Zotarelli2015

slope = slope(es,T)

def psy(P,T):
    return 0.0016286*(P/(2.501*10E-3*T))         #Zotarelli2015

psy = psy(P,T)

def Ejensen(slope,Rs,psy,T,u2,es,RH):               #EQ 11,12 Jensen 2010
    return ((0.408*slope*(Rs))+(psy*(900/(T+273))*u2*es*(1-RH/100)))/(slope+(psy*(1+0.34*u2)))*1.1
    # return es*(1-RH/100)

plt.figure(1)
df_year = df[df.index.year == 2020]
df.index = pd.to_datetime(df.index)
sns.lineplot(data = df_year, x =df_year.index, y='E_Valiantzas',label='Valiantzas2006')

sns.lineplot(data=df_year,x=df_year.index, y='E_Jensen',label='Jensen2010')
plt.xlabel('Date')
plt.ylabel('Evaporation [Liter/day]')
plt.title('Comparation of Evaporation Methods')
plt.xticks(rotation=45)
plt.legend()
plt.show
    
df['E_Jensen'] = Ejensen(slope,Rs,psy,T,u,es,RH)*A/1000


plt.figure(2)
sns.lineplot(data=df_year,x=df_year.index,y=df_year['Humitat Relativa mitjana (%)']/10)
sns.lineplot(data=df_year,x=df_year.index,y=df_year['Irradiància solar global (W/m2)']/100)
sns.lineplot(data=df_year,x=df_year.index,y=df_year['Mitjana velocitat vent a 2 m (m/s)'])
sns.scatterplot(data=es)
sns.scatterplot(data=df_year,x=df_year['Humitat Relativa mitjana (%)'],y='E_Valiantzas')
plt.show
