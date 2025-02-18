import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
from datetime import date



file = './results/EB3/EB3.csv'
df = pd.read_csv(file)

dfw = pd.DataFrame()
dfs = pd.DataFrame()

dfw['Pump1.Qout'] = df['Pump1w.Qout']
dfw['Pump1.PumpOn'] = df['Pump1w.PumpOn']

dfw['Pump2.Qout'] = df['Pump2w.Qout']
dfw['Pump2.PumpOn'] = df['Pump2w.PumpOn']

dfs['Pump1.Qout'] = df['Pump1s.Qout']
dfs['Pump1.PumpOn'] = df['Pump1s.PumpOn']

dfs['Pump2.Qout'] = df['Pump2s.Qout']
dfs['Pump2.PumpOn'] = df['Pump2s.PumpOn']


dfw1 = pd.concat([dfw]*115,ignore_index=True)
dfw2 = pd.concat([dfw]*(366-298),ignore_index=True)
dfs = pd.concat([dfs]*183,ignore_index=True)

df = pd.concat([dfw1, dfs, dfw2], ignore_index=True)

def rellenar_nan(df):
   
    df = df.fillna(df.shift(24))  
    df.ffill(inplace=True)  
    df.bfill(inplace=True)  
    return df

def files(dfs, dfw):
    dfw1= pd.concat([dfw]*115, ignore_index= True)
    dfw2 = pd.concat([dfw]*(366-298),ignore_index=True)
    dfs = pd.concat([dfs]*183,ignore_index=True)
    df= pd.concat([dfw1, dfs, dfw2], ignore_index= True)
    dec31_index = pd.date_range(start='2024-01-01 00:00:00', end='2024-12-31 23:00:00', freq='H')
    # dec31_index = dec31_index[~((dec31_index.month == 2) & (dec31_index.day == 29))]
    df.index= dec31_index
    df= df.drop(columns=['Hour'])
    return df 

meteo_files = 'data/meteo/LesPlanes_meteo_hour_aug.csv'
meteo_filew= 'data/meteo/LesPlanes_meteo_hour_jan.csv'
df_meteos = pd.read_csv(meteo_files)
df_meteow= pd.read_csv(meteo_filew)
df_meteo = files(df_meteos, df_meteow)

file_PVPCs = 'data/costs/PVPC_aug.csv'
file_PVPCw = 'data/costs/PVPC_jan.csv'
df_PVPCs = pd.read_csv(file_PVPCs)
df_PVPCw = pd.read_csv(file_PVPCw)
df_PVPC= files(df_PVPCs, df_PVPCw)

# file_exec = './data/costs/export_ExcedentariaDelAutoconsumo_2024-10-01_10_13.csv'
# df_exec['datetime'].astype(str)
# df_exec['datetime'] = df_exec['datetime'].str[:-6]  #Delete 1+utc hour
# df_exec['datetime'] = pd.to_datetime(df_exec['datetime'])
# df_exec['Hour'] = df_exec['datetime'].dt.hour
# df_exec['value'] = df_exec['value']*(1-0.3*df_exec['Hour'].apply(lambda x: 1 if (x in [8,9,10,11,12,13,14,15,16]) else 0))

# df_exec = df_exec.set_index('datetime')
# df_exec = df_exec.drop(['id', 'name', 'geoid', 'geoname'], axis=1)
# df_exec.index = range(8784)

# df_exec['value'] = df_exec['value']*(1-0.3*df_exec['Hour'].apply(lambda x: 1 if (x in [8,9,10,11,12,13,14,15,16]) else 0))



# carpeta = './data/irrigation/'  
# archivos_csv = glob.glob(os.path.join(carpeta, 'LesPlanes_irrigation_*.csv'))

# dataframes = []

# mes_a_numero = {
#     'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
#     'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
#     'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
# }

# dias_en_mes = {
#     1: 31, 2: 28, 3: 31, 4: 30, 5: 31,
#     6: 30, 7: 31, 8: 31, 9: 30, 10: 31,
#     11: 30, 12: 31
# }

# for archivo in archivos_csv:
#     df_mes = pd.read_csv(archivo)
    
#     nombre_archivo = os.path.basename(archivo)
#     mes = nombre_archivo.split('_')[2][:3].lower() 
    
#     month = mes_a_numero.get(mes)
    
#     if month is None:
#         print(f"Mes no reconocido en el archivo: {archivo}")
#         continue 
    
#     year = 2022
#     num_dias = dias_en_mes[month]
    
#     fecha_inicio = f'{year}-{month:02d}-01'
#     fecha_fin = f'{year}-{month:02d}-{num_dias}'
    
#     rango_fechas = pd.date_range(start=fecha_inicio, end=pd.to_datetime(fecha_fin) + pd.Timedelta(hours=23), freq='H')

#     df_mes['datetime'] = pd.date_range(start=fecha_inicio, periods=len(df_mes), freq='H')
    
#     df_mes.set_index('datetime', inplace=True)
    
#     df_mes.drop(columns=['Hour'], inplace=True)

#     df_mes_full = df_mes.reindex(rango_fechas).ffill() 

#     dataframes.append(df_mes_full)

# df_unificado = pd.concat(dataframes)

# df_con = df_unificado.sort_index()*1.4

# df_con.index = range(8760)

# #Irrigation 

# #Fixing Irrigation peaks demand
# df_con['datetime'] = pd.Timestamp('2023-01-01 00:00:00') + pd.to_timedelta(df_con.index, unit='h')

# df_con.set_index('datetime',inplace=True)

# for month, group in df_con.groupby(df_con.index.to_period('M')):
#     # Verificamos que el grupo tenga al menos un día (día 1)
#     if len(group) > 0:
#         # Obtener el valor de 'Qirr' a las 23:00 horas del día 1
#         day_1_23_value = group.loc[group.index[0] + pd.Timedelta(hours=23), 'Qirr']
        
#         # Verificamos que el valor existe para evitar errores
#         if not pd.isna(day_1_23_value):
#             # Asignar este valor a todas las horas del día 1
#             for hour in range(24):  # Iteramos sobre todas las horas del día
#                 df_con.loc[group.index[0] + pd.Timedelta(hours=hour), 'Qirr'] = day_1_23_value

# df_con = df_con.reset_index(drop=True)
# df_con.name = 'Qirr'



#%%Case Base and evaluation per year

'''

Case Base avaluation for a year evaluation per day

'''
#Loading files and data from optimization results
file = './results/EB3/EB3.csv'
df = pd.read_csv(file)

df_param = pd.read_csv('./results/EB3/EB3_param.csv')
df_reg= pd.read_csv('./data/irrigation/Irrigation_SegSud/Qirr.csv')

h = 8784      #Test hours

SummerO = 91+1     #Initial summer day
SummerF = 244+1    #Final summer day

#Loading parameters

W0 = int(df_param['B4w.W0'].iloc[0])
zmax = int(df_param['B4w.zmax'].iloc[0])
zmin = int(df_param['B4w.zmin'].iloc[0])
pump1_eff = int(df_param['Pump1w.eff'].iloc[0]*100)
pump2_eff = int(df_param['Pump2w.eff'].iloc[0]*100)
Wmin = int(df_param['B4w.Wmin'].iloc[0])
Wmax = int(df_param['B4w.Wmax'].iloc[0])
K_pipe = int(df_param['Pipe1w.K'].iloc[0])
PV_inst = int(df_param['PVw.Pinst'].iloc[0])
PV_eff = int(df_param['PVw.eff'].iloc[0])
z_R0 = 430



#To list and loading pump optimization result
dfw = pd.DataFrame()
dfs = pd.DataFrame()
    
dfw['Pump1.Qout'] = df['Pump1w.Qout']
dfw['Pump2.Qout'] = df['Pump2w.Qout']
dfs['Pump1.Qout'] = df['Pump1s.Qout']
dfs['Pump2.Qout'] = df['Pump2s.Qout']



dfw1 = pd.concat([dfw]*(SummerO-1),ignore_index=True)    #Days from 01/01 to 28/2
dfw2 = pd.concat([dfw]*(366-SummerF+1),ignore_index=True) #Days from 1/10 to 31/12
dfs = pd.concat([dfs]*(SummerF-SummerO),ignore_index=True)    #Days from 1/03 to 30/9

df_pumps = pd.concat([dfw1, dfs, dfw2], ignore_index=True)

Q_out1 = df_pumps['Pump1.Qout'].head(h)
Q_out2 = df_pumps['Pump2.Qout'].head(h)

Irr = df_meteo.head(h)
Qirr = df_reg['Qirr'].head(h)

PVPC = df_PVPC['PVPC'].head(h)
Exec = df_PVPC['Excedentes'].head(h)



#Creation of Results DataFrame

df_res = pd.DataFrame()

k = [None]*h
df_res['k'] = k
df_res['B5.W'] = [0]*h
df_res['B5.z'] = [0]*h
df_res['Pipe1.H'] = [0]*h
df_res['Pump1.Pe'] = [0]*h
df_res['Pump2.Pe'] = [0]*h
df_res['Pump1.Qout'] = df_pumps['Pump1.Qout'].head(h)
df_res['Pump2.Qout'] = df_pumps['Pump2.Qout'].head(h)

df_res['PV.P'] = df_param['PVw.Pinst'].iloc[0] *df_meteo['Irr']*df_param['PVw.eff'].iloc[0]/1000
df_res['Grid.P'] = [0]*h
df_res['Cost'] = [0]*h
df_res['Irrigation.Q'] = df_reg['Qirr'].head(h)
df_res['B5.W'] = df_res['B5.W'].astype(float)
df_res['B5.z'] = df_res['B5.z'].astype(float)
df_res['Pipe1.H'] = df_res['Pipe1.H'].astype(float)
df_res['Pump1.Pe'] = df_res['Pump1.Pe'].astype(float)
df_res['Pump2.Pe'] = df_res['Pump2.Pe'].astype(float)
df_res['PV.P'] = df_res['PV.P'].astype(float)
df_res['Grid.P'] = df_res['Grid.P'].astype(float)
df_res['Cost'] = df_res['Cost'].astype(float)

costs = 0
cost = [0]*h


#Extra parameters
Qmin = 0.01000188945
Qmax = 0.97298804618
Wmin = 0.9*160e3
Wmax = 185814
B5_W = W0

df_res['datetime'] = pd.date_range(start='1/1/2024', periods=len(df_res), freq='H')

k_values = []
B5_W = 160000  
costs = 0

# Adaptation of pumping program to reservoir levels and Irrigation

for dia, df_dia in df_res.groupby(df_res['datetime'].dt.date):
    Irr = df_dia['Irrigation.Q'].sum()
    Pump1 = df_dia['Pump1.Qout'].sum()
    Pump2 = df_dia['Pump2.Qout'].sum()
    min_Pump1 = df_dia.loc[df_dia['Pump1.Qout'] > 1e-5, 'Pump1.Qout'].min()
    min_Pump2 = df_dia.loc[df_dia['Pump2.Qout'] > 1e-5, 'Pump2.Qout'].min()
    max_Pump1 = df_dia.loc[df_dia['Pump1.Qout'] > 1e-5, 'Pump1.Qout'].max()
    max_Pump2 = df_dia.loc[df_dia['Pump2.Qout'] > 1e-5, 'Pump2.Qout'].max() 

    k = 1
    if Irr <= 1:
        k = 0
    elif B5_W - Irr + (Pump1 + Pump2) * 3600 < Wmin and (Pump1 > 1e-5 or Pump2 > 1e-5):
        k = (Wmin - B5_W + Irr) / ((Pump1 + Pump2) * 3600)
        if max_Pump1 * k > Qmax and max_Pump2 * k > Qmax:
            k = Qmax / max(max_Pump1, max_Pump2)
        elif max_Pump1 * k > Qmax:
            k = Qmax / max_Pump1
        elif max_Pump2 * k > Qmax:
            k = Qmax / max_Pump2
    elif B5_W - Irr + (Pump1 + Pump2) * 3600 > Wmax and (Pump1 > 1e-5 or Pump2 > 1e-5):
        k = (Wmax - B5_W + Irr) / ((Pump1 + Pump2) * 3600)
        if min_Pump1 * k < Qmin and min_Pump2 * k < Qmin:
            k = 0
        elif min_Pump1 * k < Qmin:
            if min_Pump1 * k > Qmin * 0.5:
                k = Qmin / min_Pump1
            else: k=0
        elif min_Pump2 * k < Qmin:
            if min_Pump2 * k > Qmin * 0.5:
                k = Qmin / min_Pump2
            else: k=0
    else:
        k = 1

    k_values.append((dia, k))

    B5_W += (Pump1 + Pump2) * k * 3600 - Irr

    df_res.loc[df_res['datetime'].dt.date == dia, 'B5.W'] = B5_W
    df_res.loc[df_res['datetime'].dt.date == dia, 'k'] = k

# Multiplication of k pumping correcting factor to pumping program

df_res['Pump1.Qout'] = df_res['k'] * df_res['Pump1.Qout']
df_res['Pump2.Qout'] = df_res['k'] * df_res['Pump2.Qout']



# Calculation of electrical consumption with adapted pumping program.

for i in range(0, h):
    t = df_res.index[i]
    
    # Initial value of reservoir level
    if i == 0:
        df_res.loc[t, 'B5.W'] = (df_param['B5w.W0'].iloc[0] + 
                                         (df_pumps.loc[0, 'Pump1.Qout'] + df_pumps.loc[0, 'Pump2.Qout']) * 3600)
    else:
        prev_t = df_res.index[i - 1]
        df_res.loc[t, 'B5.W'] = (df_res.loc[prev_t, 'B5.W'] +  
                                         (df_res.loc[t, 'Pump1.Qout'] + df_res.loc[t, 'Pump2.Qout']) * 3600 - 
                                         df_res.loc[t, 'Irrigation.Q'])
    
    # Variación del programa de bombeo
    df_res.loc[t, 'B5.z'] = ((df_res.loc[t, 'B5.W'] - df_param['B5w.Wmin'].iloc[0]) / 
                                     (df_param['B5w.Wmax'].iloc[0] - df_param['B5w.Wmin'].iloc[0]) * 
                                     (df_param['B5w.zmax'].iloc[0] - df_param['B5w.zmin'].iloc[0]) + 
                                     df_param['B5w.zmax'].iloc[0])

    df_res.loc[t, 'Pipe1.H'] = (df_res.loc[t, 'B5.z'] - z_R0 + 
                                df_param['Pipe1w.K'].iloc[0] * ((df_res.loc[t, 'Pump1.Qout'] + df_res.loc[t, 'Pump2.Qout']))**2)

    df_res.loc[t, 'Pump1.Pe'] = (9810 * df_res.loc[t, 'Pipe1.H'] * df_res.loc[t, 'Pump1.Qout'] / df_param['Pump1w.eff'].iloc[0])
    
    df_res.loc[t, 'Pump2.Pe'] = (9810 * df_res.loc[t, 'Pipe1.H'] * df_res.loc[t, 'Pump2.Qout'] / df_param['Pump2w.eff'].iloc[0])
    
    # Cálculo de costos
    df_res.loc[t, 'Grid_aux.P'] = df_res.loc[t, 'Pump2.Pe'] - df_res.loc[t, 'PV.P']
    
    if df_res.loc[t, 'Grid_aux.P'] >= 0:
        costs += df_res.loc[t, 'Grid_aux.P'] * PVPC[i] / 1e6 + df_res.loc[t, 'Pump1.Pe'] * PVPC[i] / 1e6
        df_res.loc[t, 'Cost'] = df_res.loc[t, 'Grid_aux.P'] * PVPC[i] / 1e6 + df_res.loc[t, 'Pump1.Pe'] * PVPC[i] / 1e6
    else:
        costs += df_res.loc[t, 'Pump1.Pe'] * PVPC[i] / 1e6
        df_res.loc[t, 'Cost'] = df_res.loc[t, 'Pump1.Pe'] * PVPC[i] / 1e6





cost1 = cost
print(costs)

#Results plots

df_res['datetime'] = pd.date_range(start='1/1/2024', periods=8784, freq='H')

# Create a new column with formatted date as 'mm/dd HH' and set it as index
df_res['date_mmdd'] = df_res['datetime'].dt.strftime('%m/%d %H')
df_res.set_index('date_mmdd', inplace=True)


daily_data = pd.DataFrame()

# Agrupar por fecha y calcular la suma del coste total diario
daily_data['date'] = df_res.groupby(df_res['datetime'].dt.date)['Cost'].sum().reset_index()['datetime']

# Calcular el coste total diario y la media del nivel del reservorio
daily_data['CB_total_cost'] = df_res.groupby(df_res['datetime'].dt.date)['Cost'].sum().values
daily_data['B5.W'] = df_res.groupby(df_res['datetime'].dt.date)['B5.W'].mean().values
daily_data['Irrigation.Q'] = df_res.groupby(df_res['datetime'].dt.date)['Irrigation.Q'].mean().values

daily_data['k'] = df_res.groupby(df_res['datetime'].dt.date)['k'].mean().values

daily_datab=daily_data
df_resb=df_res


#Total costs dataframes

df_costs = pd.DataFrame()

df_costs['CaseBCosts'] = daily_data['CB_total_cost']
df_costs['CaseB_Total'] = [df_costs['CaseBCosts'].sum()]+[None]*(len(df_costs)-1)

# Graficar utilizando Seaborn
# plt.figure(figsize=(10, 6))
# sns.lineplot(data=daily_data, x='date', y='CB_total_cost', marker='o')
# plt.xlabel('Día')
# plt.ylabel('Coste total diario')
# plt.title('Costes diarios acumulados')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


plt.figure(2)
sns.lineplot(data=daily_data, x='date', y='B5.W')
plt.xlabel('Día')
plt.ylabel('Nivel  diario')
plt.title('Nivel diarios ')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(3)
sns.lineplot(data=daily_data, x='date', y='Irrigation.Q')
plt.xlabel('Día')
plt.ylabel('Qirr')
plt.title('Qirr ')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# # plt.figure(4)
# # sns.lineplot(data=daily_data,x='date',y = 'k')
# # plt.xticks(rotation=45)
# # plt.show()

# plt.figure(5)
# df_res.index = pd.to_datetime(df_res.index,format='%m/%d %H')
# sns.lineplot(data = df_res[(df_res.index.month >= 1) & (df_res.index.month <= 12)], x=df_res[(df_res.index.month >= 1) & (df_res.index.month <= 12)].index, y = 'Pump2.Qout')
# sns.lineplot(data = df_res[(df_res.index.month >= 1) & (df_res.index.month <= 12)], x=df_res[(df_res.index.month >= 1) & (df_res.index.month <= 12)].index, y = 'Pump1.Qout')

# plt.axhline(y=Qmin, color='red', linestyle='--', label='Qmin')  # Línea horizontal
# plt.xticks(rotation=45)
# plt.show()

# counter1 = ((df_res['Pump1.Qout']<(0.6250*0.0556-1e-5)) & (df_res['Pump1.Qout'] >1e-5)).sum()
# counter2 = ((df_res['Pump2.Qout']<0.6250*0.0556) & (df_res['Pump2.Qout'] >1e-5)).sum()

# counter1m = ((df_res['Pump1.Qout']>Qmax+1e-5)).sum()
# counter2m = (df_res['Pump2.Qout']>Qmax+1e-5).sum()

# counter = counter1 + counter2
# counterm = counter1m + counter2m


#%%Case1 with Turbin1

'''
Case 1 with turbine 
'''


file = './results/EB3/EB3.csv'
df = pd.read_csv(file)

df_param = pd.read_csv('./results/EB3/EB3_param.csv')
df_reg= pd.read_csv('./data/irrigation/Irrigation_SegSud/Qirr.csv')


h = 8784      #Test hours

SummerO = 91+1     #days of summer and winter
SummerF = 244+1


# df_con = df_con.reset_index(drop=True)
# df_con.name = 'Qirr'
        

#Param

W0 = int(df_param['B5w.W0'].iloc[0])
zmax = int(df_param['B5w.zmax'].iloc[0])
zmin = int(df_param['B5w.zmin'].iloc[0])
pump1_eff = int(df_param['Pump1w.eff'].iloc[0]*100)
pump2_eff = int(df_param['Pump2w.eff'].iloc[0]*100)
turb1_eff = int(df_param['Turb1w.eff'].iloc[0]*100)
Wmin = int(df_param['B5w.Wmin'].iloc[0])
Wmax = int(df_param['B5w.Wmax'].iloc[0])
K_pipe = int(df_param['Pipe1w.K'].iloc[0])
PV_inst = int(df_param['PVw.Pinst'].iloc[0])
PV_eff = int(df_param['PVw.eff'].iloc[0])
z_R0 = 430



#To list
dfw = pd.DataFrame()
dfs = pd.DataFrame()
    
dfw['Pump1.Qout'] = df['Pump1w.Qout']
dfw['Turb1.Qout'] = df['Turb1w.Qout']
dfw['Pump2.Qout'] = df['Pump2w.Qout']

dfs['Pump1.Qout'] = df['Pump1s.Qout']
dfs['Turb1.Qout'] = df['Turb1s.Qout']

dfs['Pump2.Qout'] = df['Pump2s.Qout']



dfw1 = pd.concat([dfw]*(SummerO-1),ignore_index=True)    #Days from 01/01 to 28/2
dfw2 = pd.concat([dfw]*(366-SummerF+1),ignore_index=True) #Days from 1/10 to 31/12
dfs = pd.concat([dfs]*(SummerF-SummerO),ignore_index=True)    #Days from 1/03 to 30/9

df_pumps = pd.concat([dfw1, dfs, dfw2], ignore_index=True)

Q_out1 = df_pumps['Pump1.Qout'].head(h)
Q_out2 = df_pumps['Pump2.Qout'].head(h)
Qt_out1 = df_pumps['Turb1.Qout'].head(h)

Irr = df_meteo.head(h)
Qirr = df_reg['Qirr'].head(h)

PVPC = df_PVPC['PVPC'].head(h)
Exec = df_PVPC['Excedentes'].head(h)



df_res = pd.DataFrame()

k = [None]*h
df_res['k'] = k

df_res['B5.W'] = [0]*h
df_res['B5.z'] = [0]*h
df_res['Pipe1.H'] = [0]*h
df_res['Pump1.Pe'] = [0]*h
df_res['Pump2.Pe'] = [0]*h
df_res['Pump1.Qout'] = df_pumps['Pump1.Qout'].head(h)
df_res['Pump2.Qout'] = df_pumps['Pump2.Qout'].head(h)
df_res['Turb1.Qout'] = df_pumps['Turb1.Qout'].head(h)
df_res['Irrigation.Q'] = df_reg['Qirr'].head(h)
df_res['Turb1.Pe'] = [0]*h
df_res['PV.P'] = df_param['PVw.Pinst'].iloc[0] *df_meteo['Irr'] * df_param['PVw.eff'].iloc[0]/1000
df_res['Grid.P'] = [0]*h
df_res['Cost'] = [0]*h
df_res['Grid_aux.P'] = [0]*h

df_res['B5.W'] = df_res['B5.W'].astype(float)
df_res['B5.z'] = df_res['B5.z'].astype(float)
df_res['Pipe1.H'] = df_res['Pipe1.H'].astype(float)
df_res['Pump1.Pe'] = df_res['Pump1.Pe'].astype(float)
df_res['Pump2.Pe'] = df_res['Pump2.Pe'].astype(float)
df_res['Turb1.Pe'] = df_res['Turb1.Pe'].astype(float)
df_res['PV.P'] = df_res['PV.P'].astype(float)
df_res['Grid.P'] = df_res['Grid.P'].astype(float)
df_res['Grid_aux.P'] = df_res['Grid_aux.P'].astype(float)
df_res['Cost'] = df_res['Cost'].astype(float)

counter = 0
costs = 0
cost = [0]*h

        

Qmin = 0.01000188945
Qmax = 0.97298804618
Wmin = 0.9*160e3
Wmax = 185814
B5_W = W0

df_res['datetime'] = pd.date_range(start='1/1/2024', periods=len(df_res), freq='H')

k_values = []
B5_W = 160e3  
costs = 0

# Adaptation of pumping program to reservoir levels and Irrigation

for dia, df_dia in df_res.groupby(df_res['datetime'].dt.date):
    
    Irr = df_dia['Irrigation.Q'].sum()
    Pump1 = df_dia['Pump1.Qout'].sum()
    Pump2 = df_dia['Pump2.Qout'].sum()
    Turb1 = df_dia['Turb1.Qout'].sum()
    min_Pump1 = df_dia.loc[df_dia['Pump1.Qout'] > 1e-5, 'Pump1.Qout'].min()
    min_Pump2 = df_dia.loc[df_dia['Pump2.Qout'] > 1e-5, 'Pump2.Qout'].min()
    max_Pump1 = df_dia.loc[df_dia['Pump1.Qout'] > 1e-5, 'Pump1.Qout'].max()
    max_Pump2 = df_dia.loc[df_dia['Pump2.Qout'] > 1e-5, 'Pump2.Qout'].max() 

    k = 1
    if Irr <= 1:
        k = 0
    elif B5_W - Irr + (Pump1 + Pump2-Turb1) * 3600 < Wmin and (Pump1 > 1e-5 or Pump2 > 1e-5):
        k = (Wmin - B5_W + Irr) / ((Pump1 + Pump2-Turb1) * 3600)
        if max_Pump1 * k > Qmax and max_Pump2 * k > Qmax:
            k = Qmax / max(max_Pump1, max_Pump2)
        elif max_Pump1 * k > Qmax:
            k = Qmax / max_Pump1
        elif max_Pump2 * k > Qmax:
            k = Qmax / max_Pump2
    elif B5_W - Irr + (Pump1 + Pump2-Turb1) * 3600 > Wmax and (Pump1 > 1e-5 or Pump2 > 1e-5):
        k = (Wmax - B5_W + Irr) / ((Pump1 + Pump2-Turb1) * 3600)
        if min_Pump1 * k < Qmin and min_Pump2 * k < Qmin:
            k = 0
        elif min_Pump1 * k < Qmin:
            if min_Pump1 * k > Qmin * 0.5:
                k = Qmin / min_Pump1
            else:
                k = 0
  
        elif min_Pump2 * k < Qmin:
            if min_Pump2 * k > Qmin * 0.5:
                k = Qmin / min_Pump2
            else:
                k = 0
    else:
        k = 1

    k_values.append((dia, k))

    B5_W += (Pump1 + Pump2-Turb1) * k * 3600 - Irr

    df_res.loc[df_res['datetime'].dt.date == dia, 'B5.W'] = B5_W
    df_res.loc[df_res['datetime'].dt.date == dia, 'k'] = k

# Multiplication of k pumping correcting factor to pumping program

df_res['Pump1.Qout'] = df_res['k'] * df_res['Pump1.Qout']
df_res['datetime'] = pd.to_datetime(df_res['datetime'])
mask = (df_res['datetime'].dt.month>=4) & (df_res['datetime'].dt.month<=8 )     #Only applies for summer months
df_res.loc[mask,'Pump2.Qout'] = df_res.loc[mask,'k'] * df_res.loc[mask,'Pump2.Qout']
df_res.loc[mask,'Turb1.Qout'] = df_res.loc[mask,'k'] * df_res.loc[mask,'Turb1.Qout']


#Calculation of energy consumption with pumping and turbine adaptation

for i in range(0,h):
    if i == 0:
        df_res.loc[0, 'B5.W'] = (df_param['B5w.W0'].iloc[0] + (df_res.loc[0,'Pump1.Qout'] + df_res.loc[0, 'Pump2.Qout'] - df_res.loc[0, 'Turb1.Qout'] ) * 3600)
        df_res.loc[0,'Pipe1.H'] = 137

    else:
        
        df_res.loc[i,'Pump2.Qout'] = (df_res.loc[i,'PV.P']*df_param['Pump2w.eff'].iloc[0]/(9810*df_res.loc[i-1,'Pipe1.H']))
        
        if df_res.loc[i,'Pump2.Qout'] < Qmin-1e-5 and df_res.loc[i,'Pump2.Qout']>1e-6:
            df_res.loc[i,'Pump2.Qout'] = 0
            
        elif df_res.loc[i,'Pump2.Qout'] > Qmax+1e-5:
            df_res.loc[i,'Pump2.Qout'] = Qmax
        
        else: pass
            
        df_res.loc[i, 'B5.W'] = (df_res.loc[i-1,'B5.W'] +  (df_res.loc[i,'Pump1.Qout'] + df_res.loc[i, 'Pump2.Qout'] - df_res.loc[i, 'Turb1.Qout'] ) * 3600 - df_res.loc[i, 'Irrigation.Q'])
            
        if df_res.loc[i,'B5.W'] > Wmax:
            df_res.loc[i,'Pump2.Qout'] = 0
            df_res.loc[i,'Pump1.Qout'] = 0
        elif df_res.loc[i,'B5.W'] <=Wmin:
            df_res.loc[i,'Turb1.Qout'] = 0
        else:pass
        
        df_res.loc[i, 'B5.W'] = (df_res.loc[i-1,'B5.W'] +  (df_res.loc[i,'Pump1.Qout'] + df_res.loc[i, 'Pump2.Qout'] - df_res.loc[i, 'Turb1.Qout'] ) * 3600 - df_res.loc[i, 'Irrigation.Q'])

        df_res.loc[i, 'B5.z'] = ((df_res.loc[i, 'B5.W'] - df_param['B5w.Wmin'].iloc[0]) /(df_param['B5w.Wmax'].iloc[0] - df_param['B5w.Wmin'].iloc[0]) *(df_param['B5w.zmax'].iloc[0] - df_param['B5w.zmin'].iloc[0]) + 
            df_param['B5w.zmax'].iloc[0])

        df_res.loc[i, 'Pipe1.H'] = (df_res.loc[i, 'B5.z'] - df_param['B4w.zmax'].iloc[0] + df_param['Pipe1w.K'].iloc[0] * ((df_res.loc[i, 'Pump1.Qout'] +  df_res.loc[i, 'Pump2.Qout'] +df_res.loc[i, 'Turb1.Qout']))**2)
        
        df_res.loc[i, 'Pump1.Pe'] = ( 9810 * df_res.loc[i, 'Pipe1.H'] *df_res.loc[i, 'Pump1.Qout'] / df_param['Pump1w.eff'].iloc[0])
        
        # df_res.loc[i, 'Pump2.Pe'] = (9810 * df_res.loc[i, 'Pipe1.H'] * df_res.loc[i, 'Pump2.Qout'] / df_param['Pump2w.eff'].iloc[0])
    # df_res.loc[i,'Grid_aux.P'] = df_res.loc[i,'Pump2.Pe'] - df_res.loc[i,'PV.P']                         #EQ7
    df_res.loc[i, 'Turb1.Pe'] = (9810 * df_res.loc[i, 'Pipe1.H'] * df_res.loc[i, 'Turb1.Qout'] * df_param['Turb1w.eff'].iloc[0])
    
    if df_res.loc[i,'Grid_aux.P'] > 0:
        costs = costs + df_res.loc[i,'Grid_aux.P']*PVPC[i]/1e6 +df_res.loc[i,'Pump1.Pe']*PVPC[i]/1e6 +df_res.loc[i, 'Turb1.Pe']*Exec[i]/1e6
        df_res.loc[i,'Cost'] = df_res.loc[i,'Grid_aux.P']*PVPC[i]/1e6 + df_res.loc[i,'Pump1.Pe']*PVPC[i]/1e6 + df_res.loc[i, 'Turb1.Pe']*Exec[i]/1e6
    
    else:
        costs = costs + df_res.loc[i,'Pump1.Pe']*PVPC[i]/1e6 - df_res.loc[i, 'Turb1.Pe']*Exec[i]/1e6
        df_res.loc[i,'Cost'] =  df_res.loc[i,'Pump1.Pe']*PVPC[i]/1e6 - df_res.loc[i, 'Turb1.Pe']*Exec[i]/1e6
    
cost1 = cost
print(costs)

df_res['datetime'] = pd.date_range(start='1/1/2024', periods=8784, freq='H')

# Create a new column with formatted date as 'mm/dd HH' and set it as index
df_res['date_mmdd'] = df_res['datetime'].dt.strftime('%m/%d %H')
df_res.set_index('date_mmdd', inplace=True)

daily_data = pd.DataFrame()

# Agrupar por fecha y calcular la suma del coste total diario
daily_data['date'] = df_res.groupby(df_res['datetime'].dt.date)['Cost'].sum().reset_index()['datetime']

# Calcular el coste total diario y la media del nivel del reservorio
daily_data['C1_total_cost'] = df_res.groupby(df_res['datetime'].dt.date)['Cost'].sum().values
daily_data['B5.W'] = df_res.groupby(df_res['datetime'].dt.date)['B5.W'].mean().values
daily_data['Irrigation.Q'] = df_res.groupby(df_res['datetime'].dt.date)['Irrigation.Q'].mean().values
daily_data['k'] = df_res.groupby(df_res['datetime'].dt.date)['k'].mean().values

daily_data1=daily_data
df_res1=df_res

df_costs = pd.DataFrame()
df_costs['Case1Costs'] = daily_data['C1_total_cost']
df_costs['Case1_Total'] = [df_costs['Case1Costs'].sum()]+[None]*(len(df_costs)-1)


# Graficar utilizando Seaborn
# plt.figure(figsize=(10, 6))
# sns.lineplot(data=daily_data, x='date', y='C1_total_cost', marker='o')
# plt.xlabel('Día')
# plt.ylabel('Coste total diario')
# plt.title('Costes diarios acumulados')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


# plt.figure(2)
# sns.lineplot(data=daily_data, x='date', y='B5.W')
# plt.xlabel('Día')
# plt.ylabel('Nivel  diario')
# plt.title('Nivel diarios ')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# plt.figure(3)
# sns.lineplot(data=daily_data, x='date', y='Irrigation.Q')
# plt.xlabel('Día')
# plt.ylabel('Qirr')
# plt.title('Qirr ')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()



# counter1 = ((df_res['Pump1.Qout']<(0.6250*0.0556-1e-5)) & (df_res['Pump1.Qout'] >1e-5)).sum()
# counter2 = ((df_res['Pump2.Qout']<0.6250*0.0556-1e-5) & (df_res['Pump2.Qout'] >1e-5)).sum()

# counter1m = ((df_res['Pump1.Qout']>Qmax+1e-5)).sum()
# counter2m = (df_res['Pump2.Qout']>Qmax+1e-5).sum()

# counter = counter1 + counter2
# counterm = counter1m + counter2m



#%%Case 2

'''
Case 2 avaluation for a year.
'''

from datetime import datetime, timedelta


file = './results/ISGT/Case2/ISGT.csv'
df = pd.read_csv(file)

df_param = pd.read_csv('./results/ISGT/Case2/ISGT_param.csv')

carpeta = './data/irrigation/LesPlanes_irrigation_'

# df_conw = pd.read_csv(carpeta + 'jan.csv')
# df_cons = pd.read_csv(carpeta + 'aug.csv')


h = 8760      #Test hours

SummerO = 91     #days of summer and winter
SummerF = 244


df_con = df_con.reset_index(drop=True)
df_con.name = 'Qirr'


#Param

W0 = int(df_param['B5w.W0'].iloc[0])
zmax = int(df_param['B5w.zmax'].iloc[0])
zmin = int(df_param['B5w.zmin'].iloc[0])
pump1_eff = int(df_param['Pump1w.eff'].iloc[0]*100)
pump2_eff = int(df_param['Pump2w.eff'].iloc[0]*100)
Wmin = int(df_param['B5w.Wmin'].iloc[0])
Wmax = int(df_param['B5w.Wmax'].iloc[0])
K_pipe = int(df_param['Pipe1w.K'].iloc[0])
PV_inst = int(df_param['PVw.Pinst'].iloc[0])
PV_eff = int(df_param['PVw.eff'].iloc[0])
z_R0 = 30



#To list
dfw = pd.DataFrame()
dfs = pd.DataFrame()
    
dfw['Pump1.Qout'] = df['Pump1w.Qout']
dfw['Pump2.Qout'] = df['Pump2w.Qout']

dfs['Pump1.Qout'] = df['Pump1s.Qout']
dfs['Pump2.Qout'] = df['Pump2s.Qout']



dfw1 = pd.concat([dfw]*(SummerO),ignore_index=True)    
dfw2 = pd.concat([dfw]*(365-SummerF),ignore_index=True)
dfs = pd.concat([dfs]*(SummerF-SummerO),ignore_index=True)   

df_pumps = pd.concat([dfw1, dfs, dfw2], ignore_index=True)

Q_out1 = df_pumps['Pump1.Qout'].head(h)
Q_out2 = df_pumps['Pump2.Qout'].head(h)

Irr = df_meteo.head(h)
Qirr = df_con['Qirr'].head(h)

PVPC = df_PVPC['PVPC'].head(h)
Exec = df_PVPC['Excedentes'].head(h)    




df_res = pd.DataFrame()

df_res['k'] = [None]*h

df_res['B5.W'] = [0]*h
df_res['B5.z'] = [0]*h
df_res['Pipe1.H'] = [0]*h
df_res['Pump1.Pe'] = [0]*h
df_res['Pump2.Pe'] = [0]*h
df_res['Pump1.Qout'] = df_pumps['Pump1.Qout'].head(h)
df_res['Pump2.Qout'] = df_pumps['Pump2.Qout'].head(h)
df_res['PV.P'] = df_param['PVw.Pinst'].iloc[0] *df_meteo * df_param['PVw.eff'].iloc[0]/1000
df_res['Grid.P'] = [0]*h
df_res['Cost'] = [0]*h
df_res['Irrigation.Q'] = df_con['Qirr'].head(h)
df_res['B5.W'] = df_res['B5.W'].astype(float)
df_res['B5.z'] = df_res['B5.z'].astype(float)
df_res['Pipe1.H'] = df_res['Pipe1.H'].astype(float)
df_res['Pump1.Pe'] = df_res['Pump1.Pe'].astype(float)
df_res['Pump2.Pe'] = df_res['Pump2.Pe'].astype(float)
df_res['PV.P'] = df_res['PV.P'].astype(float)
df_res['Grid.P'] = df_res['Grid.P'].astype(float)
df_res['Cost'] = df_res['Cost'].astype(float)

counter = 0
costs = 0
cost = [0]*h


Qmin = 0.0556*0.6250
Qmax = 0.0556*1.825
Wmin = 7000
Wmax = 13000
B5_W = W0

df_res['datetime'] = pd.date_range(start='1/1/2023', periods=len(df_res), freq='H')

k_values = []
B5_W = 10000  
costs = 0

# Adaptation of pumping program to reservoir levels and Irrigation
for dia, df_dia in df_res.groupby(df_res['datetime'].dt.date):
    Irr = df_dia['Irrigation.Q'].sum()
    Pump1 = df_dia['Pump1.Qout'].sum()
    Pump2 = df_dia['Pump2.Qout'].sum()
    min_Pump1 = df_dia.loc[df_dia['Pump1.Qout'] > 1e-5, 'Pump1.Qout'].min()
    min_Pump2 = df_dia.loc[df_dia['Pump2.Qout'] > 1e-5, 'Pump2.Qout'].min()
    max_Pump1 = df_dia.loc[df_dia['Pump1.Qout'] > 1e-5, 'Pump1.Qout'].max()
    max_Pump2 = df_dia.loc[df_dia['Pump2.Qout'] > 1e-5, 'Pump2.Qout'].max() 

    k = 1
    if Irr <= 1:
        k = 0
    elif B5_W - Irr + (Pump1 + Pump2) * 3600 < Wmin and (Pump1 > 1e-5 or Pump2 > 1e-5):
        k = (Wmin - B5_W + Irr) / ((Pump1 + Pump2) * 3600)

        if max_Pump1 * k > Qmax and max_Pump2 * k > Qmax:
            k = Qmax / max(max_Pump1, max_Pump2)
        elif max_Pump1 * k > Qmax:
            k = Qmax / max_Pump1
        elif max_Pump2 * k > Qmax:
            k = Qmax / max_Pump2
    elif B5_W - Irr + (Pump1 + Pump2) * 3600 > Wmax and (Pump1 > 1e-5 or Pump2 > 1e-5):
        k = (Wmax - B5_W + Irr) / ((Pump1 + Pump2) * 3600)
        if min_Pump1 * k < Qmin and min_Pump2 * k < Qmin:
            k = 0
        elif min_Pump1 * k < Qmin:
            if min_Pump1 * k > Qmin * 0.5:
                k = Qmin / min_Pump1
            else: k=0

        elif min_Pump2 * k < Qmin:
            if min_Pump2 * k > Qmin * 0.5:
                k = Qmin / min_Pump2
            else: k=0

    else:
        k = 1

    k_values.append((dia, k))

    B5_W += (Pump1 + Pump2) * k * 3600 - Irr

    df_res.loc[df_res['datetime'].dt.date == dia, 'B5.W'] = B5_W
    df_res.loc[df_res['datetime'].dt.date == dia, 'k'] = k

# Multiplication of "k" pumping correcting factor to pumping program

df_res['Pump1.Qout'] = df_res['k'] * df_res['Pump1.Qout']
df_res['Pump2.Qout'] = df_res['k'] * df_res['Pump2.Qout']


for i in range(0,h):
    if i == 0:
        df_res.loc[0, 'B5.W'] = (df_param['B5w.W0'].iloc[0] + (df_res.loc[0,'Pump1.Qout'] + df_res.loc[0, 'Pump2.Qout'] ) * 3600)
        
    else:
        
        df_res.loc[i, 'B5.W'] = (df_res.loc[i-1,'B5.W'] +  (df_res.loc[i,'Pump1.Qout'] + df_res.loc[i, 'Pump2.Qout'] ) * 3600 -df_con.loc[i, 'Qirr'])
        
        df_res.loc[i, 'B5.z'] = ((df_res.loc[i, 'B5.W'] - df_param['B5w.Wmin'].iloc[0]) /(df_param['B5w.Wmax'].iloc[0] - df_param['B5w.Wmin'].iloc[0]) *(df_param['B5w.zmax'].iloc[0] - df_param['B5w.zmin'].iloc[0]) + 
            df_param['B5w.zmax'].iloc[0])

        df_res.loc[i, 'Pipe1.H'] = df_res.loc[i, 'B5.z'] - df_param['B4w.zmax'].iloc[0] + df_param['Pipe1w.K'].iloc[0] * (((df_res.loc[i, 'Pump1.Qout'] +  df_res.loc[i, 'Pump2.Qout']))**2)

        df_res.loc[i, 'Pump1.Pe'] = ( 9810 * df_res.loc[i, 'Pipe1.H'] *df_res.loc[i, 'Pump1.Qout'] / df_param['Pump1w.eff'].iloc[0])
        
        df_res.loc[i, 'Pump2.Pe'] = (9810 * df_res.loc[i, 'Pipe1.H'] * df_res.loc[i, 'Pump2.Qout'] / df_param['Pump2w.eff'].iloc[0])

        df_res.loc[i, 'Grid.P'] = df_res.loc[i, 'Pump1.Pe'] + df_res.loc[i, 'Pump2.Pe'] - df_res.loc[i, 'PV.P']         
    
        if df_res.loc[i,'Grid.P'] >= 0:
            costs = costs + df_res.loc[i, 'Grid.P'] * PVPC[i]/1e6
            df_res.loc[i,'Cost'] = df_res.loc[i, 'Grid.P'] *PVPC[i]/1e6
        
        else:
            costs = costs + df_res.loc[i, 'Grid.P']*Exec[i]/1e6
            df_res.loc[i,'Cost'] = df_res.loc[i, 'Grid.P']*Exec[i]/1e6
        
cost2 = cost
print(costs)

df_res['datetime'] = pd.date_range(start='1/1/2023', periods=8760, freq='H')

# Create a new column with formatted date as 'mm/dd HH' and set it as index
df_res['date_mmdd'] = df_res['datetime'].dt.strftime('%m/%d %H')
df_res.set_index('date_mmdd', inplace=True)

daily_data = pd.DataFrame()

# Agrupar por fecha y calcular la suma del coste total diario
daily_data['date'] = df_res.groupby(df_res['datetime'].dt.date)['Cost'].sum().reset_index()['datetime']

# Calcular el coste total diario y la media del nivel del reservorio
daily_data['C2_total_cost'] = df_res.groupby(df_res['datetime'].dt.date)['Cost'].sum().values
daily_data['B5.W'] = df_res.groupby(df_res['datetime'].dt.date)['B5.W'].mean().values
daily_data['Irrigation.Q'] = df_res.groupby(df_res['datetime'].dt.date)['Irrigation.Q'].mean().values

daily_data2=daily_data
df_res2=df_res


df_costs['Case2Costs'] = daily_data['C2_total_cost']
df_costs['Case2_Total'] = [df_costs['Case2Costs'].sum()]+[None]*(len(df_costs)-1)


# Graficar utilizando Seaborn
# plt.figure(figsize=(10, 6))
# sns.lineplot(data=daily_data, x='date', y='C2_total_cost', marker='o')
# plt.xlabel('Día')
# plt.ylabel('Coste total diario')
# plt.title('Costes diarios acumulados')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


# plt.figure(2)
# sns.lineplot(data=daily_data, x='date', y='B5.W')
# plt.xlabel('Día')
# plt.ylabel('Nivel  diario')
# plt.title('Nivel diarios ')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# plt.figure(3)
# sns.lineplot(data=daily_data, x='date', y='Irrigation.Q')
# plt.xlabel('Día')
# plt.ylabel('Qirr')
# plt.title('Qirr ')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


# plt.figure(5)
# df_res.index = pd.to_datetime(df_res.index,format='%m/%d %H')
# sns.lineplot(data = df_res[(df_res.index.month >= 1) & (df_res.index.month <= 12)], x=df_res[(df_res.index.month >= 1) & (df_res.index.month <= 12)].index, y = 'Pump2.Qout')
# sns.lineplot(data = df_res[(df_res.index.month >= 1) & (df_res.index.month <= 12)], x=df_res[(df_res.index.month >= 1) & (df_res.index.month <= 12)].index, y = 'Pump1.Qout')
# plt.axhline(y=Qmin, color='red', linestyle='--', label='Qmin')  # Línea horizontal
# plt.xticks(rotation=45)
# plt.show()



# counter1 = ((df_res['Pump1.Qout']<(0.6250*0.0556-1e-5)) & (df_res['Pump1.Qout'] >1e-5)).sum()
# counter2 = ((df_res['Pump2.Qout']<0.6250*0.0556-1e-5) & (df_res['Pump2.Qout'] >1e-5)).sum()

# counter1m = ((df_res['Pump1.Qout']>Qmax+1e-5)).sum()
# counter2m = (df_res['Pump2.Qout']>Qmax+1e-5).sum()

# counter = counter1 + counter2
# counterm = counter1m + counter2m



#%%

'Case 3: PAT + PV to grid.'


file = './results/ISGT/Case3/ISGT.csv'
df = pd.read_csv(file)

df_param = pd.read_csv('./results/ISGT/Case3/ISGT_param.csv')

carpeta = './data/irrigation/LesPlanes_irrigation_'


h = 8760      #Test hours

SummerO = 91     #days of summer and winter
SummerF = 244

df_con = df_con.reset_index(drop=True)
df_con.name = 'Qirr'

#Param

W0 = int(df_param['B5w.W0'].iloc[0])
zmax = int(df_param['B5w.zmax'].iloc[0])
zmin = int(df_param['B5w.zmin'].iloc[0])
pump1_eff = int(df_param['Pump1w.eff'].iloc[0]*100)
pump2_eff = int(df_param['Pump2w.eff'].iloc[0]*100)
turb1_eff = int(df_param['Turb1w.eff'].iloc[0]*100)
Wmin = int(df_param['B5w.Wmin'].iloc[0])
Wmax = int(df_param['B5w.Wmax'].iloc[0])
K_pipe = int(df_param['Pipe1w.K'].iloc[0])
PV_inst = int(df_param['PVw.Pinst'].iloc[0])
PV_eff = int(df_param['PVw.eff'].iloc[0])
z_R0 = 30



#To list
dfw = pd.DataFrame()
dfs = pd.DataFrame()
    
dfw['Pump1.Qout'] = df['Pump1w.Qout']
dfw['Turb1.Qout'] = df['Turb1w.Qout']
dfw['Pump2.Qout'] = df['Pump2w.Qout']

dfs['Pump1.Qout'] = df['Pump1s.Qout']
dfs['Turb1.Qout'] = df['Turb1s.Qout']

dfs['Pump2.Qout'] = df['Pump2s.Qout']



dfw1 = pd.concat([dfw]*(SummerO-1),ignore_index=True)    #Days from 01/01 to 28/2
dfw2 = pd.concat([dfw]*(365-SummerF+1),ignore_index=True) #Days from 1/10 to 31/12
dfs = pd.concat([dfs]*(SummerF-SummerO),ignore_index=True)    #Days from 1/03 to 30/9

df_pumps = pd.concat([dfw1, dfs, dfw2], ignore_index=True)

Q_out1 = df_pumps['Pump1.Qout'].head(h)
Q_out2 = df_pumps['Pump2.Qout'].head(h)
Qt_out1 = df_pumps['Turb1.Qout'].head(h)

Irr = df_meteo.head(h)
Qirr = df_con['Qirr'].head(h)

PVPC = df_PVPC['value'].head(h)
Exec = df_exec['value'].head(h)



df_res = pd.DataFrame()

df_res['k'] = [None]*h
df_res['B5.W'] = [0]*h
df_res['B5.z'] = [0]*h
df_res['Pipe1.H'] = [0]*h
df_res['Pump1.Pe'] = [0]*h
df_res['Pump2.Pe'] = [0]*h
df_res['Pump1.Qout'] = df_pumps['Pump1.Qout'].head(h)
df_res['Pump2.Qout'] = df_pumps['Pump2.Qout'].head(h)
df_res['Turb1.Qout'] = df_pumps['Turb1.Qout'].head(h)
df_res['Turb1.Pe'] = [0]*h
df_res['PV.P'] = df_param['PVw.Pinst'].iloc[0] *df_meteo * df_param['PVw.eff'].iloc[0]/1000
df_res['Grid.P'] = [0]*h
df_res['Cost'] = [0]*h
df_res['Irrigation.Q'] = df_con['Qirr'].head(h)
df_res['B5.W'] = df_res['B5.W'].astype(float)
df_res['B5.z'] = df_res['B5.z'].astype(float)
df_res['Pipe1.H'] = df_res['Pipe1.H'].astype(float)
df_res['Pump1.Pe'] = df_res['Pump1.Pe'].astype(float)
df_res['Pump2.Pe'] = df_res['Pump2.Pe'].astype(float)
df_res['Turb1.Pe'] = df_res['Turb1.Pe'].astype(float)
df_res['PV.P'] = df_res['PV.P'].astype(float)
df_res['Grid.P'] = df_res['Grid.P'].astype(float)
df_res['Cost'] = df_res['Cost'].astype(float)

Qmin = 0.0556*0.6250
Qmax = 0.0556*1.825
Wmin = 7000
Wmax = 13000
B5_W = W0

df_res['datetime'] = pd.date_range(start='1/1/2023', periods=len(df_res), freq='H')

k_values = []
B5_W = 10000  
costs = 0

counter = 0
costs = 0
cost = [0]*h

for dia, df_dia in df_res.groupby(df_res['datetime'].dt.date):
    
    Irr = df_dia['Irrigation.Q'].sum()
    Pump1 = df_dia['Pump1.Qout'].sum()
    Pump2 = df_dia['Pump2.Qout'].sum()
    Turb1 = df_dia['Turb1.Qout'].sum()
    min_Pump1 = df_dia.loc[df_dia['Pump1.Qout'] > 1e-5, 'Pump1.Qout'].min()
    min_Pump2 = df_dia.loc[df_dia['Pump2.Qout'] > 1e-5, 'Pump2.Qout'].min()
    max_Pump1 = df_dia.loc[df_dia['Pump1.Qout'] > 1e-5, 'Pump1.Qout'].max()
    max_Pump2 = df_dia.loc[df_dia['Pump2.Qout'] > 1e-5, 'Pump2.Qout'].max() 

    k = 1
    if Irr <= 1:
        k = 0
    elif B5_W - Irr + (Pump1 + Pump2-Turb1) * 3600 < Wmin and (Pump1 > 1e-5 or Pump2 > 1e-5):
        k = (Wmin - B5_W + Irr) / ((Pump1 + Pump2-Turb1) * 3600)
        if max_Pump1 * k > Qmax and max_Pump2 * k > Qmax:
            k = Qmax / max(max_Pump1, max_Pump2)
        elif max_Pump1 * k > Qmax:
            k = Qmax / max_Pump1
        elif max_Pump2 * k > Qmax:
            k = Qmax / max_Pump2
    elif B5_W - Irr + (Pump1 + Pump2-Turb1) * 3600 > Wmax and (Pump1 > 1e-5 or Pump2 > 1e-5):
        k = (Wmax - B5_W + Irr) / ((Pump1 + Pump2-Turb1) * 3600)
        if min_Pump1 * k < Qmin and min_Pump2 * k < Qmin:
            k = 0
        elif min_Pump1 * k < Qmin:
            if min_Pump1 * k > Qmin * 0.5:
                k = Qmin / min_Pump1
            else:
                k = 0
        elif min_Pump2 * k < Qmin:
            if min_Pump2 * k > Qmin * 0.5:
                k = Qmin / min_Pump2
            else:
                k = 0
    else:
        k = 1

    k_values.append((dia, k))

    B5_W += (Pump1 + Pump2-Turb1) * k * 3600 - Irr

    df_res.loc[df_res['datetime'].dt.date == dia, 'B5.W'] = B5_W
    df_res.loc[df_res['datetime'].dt.date == dia, 'k'] = k

# Multiplication of k pumping correcting factor to pumping program

df_res['Pump1.Qout'] = df_res['k'] * df_res['Pump1.Qout']
df_res['Pump2.Qout'] = df_res['k'] * df_res['Pump2.Qout']
df_res['Turb1.Qout'] = df_res['k'] * df_res['Turb1.Qout']


winter = list(range(1, 90*24)) + list(range(181*24, 366*24))


for i in range(0,h):
    if i == 0:
        df_res.loc[0, 'B5.W'] = (df_param['B5w.W0'].iloc[0] + (df_res.loc[0,'Pump1.Qout'] + df_res.loc[0, 'Pump2.Qout'] - df_res.loc[0, 'Turb1.Qout'] ) * 3600)
        
    else:
        
        
        df_res.loc[i, 'B5.W'] = (df_res.loc[i-1,'B5.W'] +  (df_res.loc[i,'Pump1.Qout'] + df_pumps.loc[i, 'Pump2.Qout'] - df_res.loc[i, 'Turb1.Qout'] ) * 3600 -df_res.loc[i, 'Irrigation.Q'])
        
        df_res.loc[i, 'B5.z'] = ((df_res.loc[i, 'B5.W'] - df_param['B5w.Wmin'].iloc[0]) /(df_param['B5w.Wmax'].iloc[0] - df_param['B5w.Wmin'].iloc[0]) *(df_param['B5w.zmax'].iloc[0] - df_param['B5w.zmin'].iloc[0]) + 
            df_param['B5w.zmax'].iloc[0])


        df_res.loc[i, 'Pipe1.H'] = (df_res.loc[i, 'B5.z'] - df_param['B4s.zmax'].iloc[0] + df_param['Pipe1w.K'].iloc[0] * ((df_res.loc[i, 'Pump1.Qout'] +  df_res.loc[i, 'Pump2.Qout'] +df_res.loc[i, 'Turb1.Qout']))**2)

        df_res.loc[i, 'Pump1.Pe'] = ( 9810 * df_res.loc[i, 'Pipe1.H'] *df_pumps.loc[i, 'Pump1.Qout'] / df_param['Pump1w.eff'].iloc[0])
        
        df_res.loc[i, 'Pump2.Pe'] = (9810 * df_res.loc[i, 'Pipe1.H'] * df_pumps.loc[i, 'Pump2.Qout'] / df_param['Pump2w.eff'].iloc[0])

    
        if df_res.loc[i, 'PV.P'] < (df_res.loc[i, 'Pump1.Pe'] + df_res.loc[i, 'Pump2.Pe']) and i in winter:
            if df_res.loc[i, 'PV.P'] >= df_res.loc[i, 'Pump1.Pe']:
                df_res.loc[i, 'Pump2.Qout'] = 0
                df_res.loc[i, 'Pump2.Pe'] = 0
        
            if df_res.loc[i, 'PV.P'] >= df_res.loc[i, 'Pump2.Pe']:
                df_res.loc[i, 'Pump1.Qout'] = 0
                df_res.loc[i, 'Pump1.Pe'] = 0
            else:
                # Desactivar ambas bombas
                df_res.loc[i, 'Pump1.Qout'] = 0
                df_res.loc[i, 'Pump2.Qout'] = 0
                df_res.loc[i, 'Pump1.Pe'] = 0
                df_res.loc[i, 'Pump2.Pe'] = 0
        else:
            df_res.loc[i, 'Pump1.Qout'] = df_res.loc[i, 'Pump1.Qout']
            df_res.loc[i, 'Pump2.Qout'] = df_res.loc[i, 'Pump2.Qout']
            df_res.loc[i, 'Pump1.Pe'] = df_res.loc[i, 'Pump1.Pe']
            df_res.loc[i, 'Pump2.Pe'] = df_res.loc[i, 'Pump2.Pe']


        df_res.loc[i, 'B5.W'] = (df_res.loc[i-1,'B5.W'] +  (df_pumps.loc[i,'Pump1.Qout'] + df_pumps.loc[i, 'Pump2.Qout'] - df_pumps.loc[i, 'Turb1.Qout'])*df_res.loc[i,'k'] * 3600 -df_con.loc[i, 'Qirr'])


    
            ##Dues bombes activades

        df_res.loc[i, 'Turb1.Pe'] = (9810 * df_res.loc[i, 'Pipe1.H'] * df_pumps.loc[i, 'Turb1.Qout']*df_res.loc[i,'k'] * df_param['Turb1w.eff'].iloc[0])
    
        
        df_res.loc[i, 'Grid.P'] = df_res.loc[i, 'Pump1.Pe'] + df_res.loc[i, 'Pump2.Pe'] - df_res.loc[i, 'PV.P'] - df_res.loc[i,'Turb1.Pe']            
        
        if df_res.loc[i,'Grid.P'] >= 0:
            costs = costs + df_res.loc[i, 'Grid.P'] * PVPC[i]/1e6
            df_res.loc[i,'Cost'] = df_res.loc[i, 'Grid.P'] *PVPC[i]/1e6
        
        else:
            costs = costs + df_res.loc[i, 'Grid.P']*Exec[i]/1e6
            df_res.loc[i,'Cost'] = df_res.loc[i, 'Grid.P']*Exec[i]/1e6
        


cost3 = cost
print(costs)

df_res['datetime'] = pd.date_range(start='1/1/2023', periods=8760, freq='H')

# Create a new column with formatted date as 'mm/dd HH' and set it as index
df_res['date_mmdd'] = df_res['datetime'].dt.strftime('%m/%d %H')
df_res.set_index('date_mmdd', inplace=True)

daily_data = pd.DataFrame()

# Agrupar por fecha y calcular la suma del coste total diario
daily_data['date'] = df_res.groupby(df_res['datetime'].dt.date)['Cost'].sum().reset_index()['datetime']

# Calcular el coste total diario y la media del nivel del reservorio
daily_data['C3_total_cost'] = df_res.groupby(df_res['datetime'].dt.date)['Cost'].sum().values
daily_data['B5.W'] = df_res.groupby(df_res['datetime'].dt.date)['B5.W'].mean().values
daily_data3=daily_data
df_res3=df_res

df_costs['Case3Costs'] = daily_data['C3_total_cost']
df_costs['Case3_Total'] = [df_costs['Case3Costs'].sum()]+[None]*(len(df_costs)-1)



# Graficar utilizando Seaborn
# plt.figure(figsize=(10, 6))
# sns.lineplot(data=daily_data, x='date', y='C3_total_cost', marker='o')
# plt.xlabel('Día')
# plt.ylabel('Coste total diario')
# plt.title('Costes diarios acumulados')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


# plt.figure(2)
# sns.lineplot(data=daily_data, x='date', y='B5.W')
# plt.xlabel('Día')
# plt.ylabel('Nivel  diario')
# plt.title('Nivel diarios ')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


#Nombre de vegades que Qpump1/Qpump2 < Qmin

# counter1 = ((df_res['Pump1.Qout']<(0.6250*0.0556)-1e-4) & (df_res['Pump1.Qout'] >= 1e-6)).sum()
# counter2 = ((df_res['Pump2.Qout']<(0.6250*0.0556)-1e-4) & (df_res['Pump2.Qout'] >= 1e-6)).sum()
# counter = counter1 + counter2


# df_costs = df_costs.drop(['CaseBCosts','Case1Costs','Case2Costs','Case3Costs'], axis=1)
# df_costs = df_costs.dropna(how='all')

df_long = df_costs.melt(var_name='Case', value_name='Value')
df_long['Case'] = df_long['Case'].str.replace('_Total', '')

#%%
#Plots
import matplotlib.dates as mdates



plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Nimbus Roman No9 L"],
    "font.size": 9,
    'axes.spines.top': False,
    'axes.spines.right': False
})
# labels_hours = ['0','','','','','','6','','','','','','12','','','','','','18','','','','','23']

cbcolors = sns.color_palette('colorblind')

plt.figure(figsize=(3.4,2.1))
sns.lineplot(data=daily_datab,x='date',y='Irrigation.Q')
plt.fill_between( daily_datab['date'], daily_datab['Irrigation.Q'], color=cbcolors[0], alpha=0.2)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m'))  
plt.gca().xaxis.set_major_locator(mdates.MonthLocator()) 
plt.xlim(daily_data3['date'].min(), daily_data3['date'].max())        
plt.axhline(y=0,color='black',linewidth=0.5)
plt.ylabel('Flow (m3/h)')
plt.xlabel('Month', labelpad=-8, x=1.07)  
plt.subplots_adjust(left=0.18, right=0.88, top=0.97, bottom=0.20)
plt.show()


plt.figure(figsize=(3.4,2.1))
sns.lineplot(data=daily_datab,x='date',y='CB_total_cost')
plt.fill_between( daily_datab['date'], daily_datab['CB_total_cost'], color=cbcolors[0], alpha=0.2)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m'))  
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  
plt.xlim(daily_data3['date'].min(), daily_data3['date'].max())       
plt.axhline(y=0,color='black',linewidth=0.5)
plt.ylabel('Cost (€/day)')
plt.xlabel('Month', labelpad=-8, x=1.07)  
plt.subplots_adjust(left=0.18, right=0.88, top=0.97, bottom=0.20)
plt.show()

plt.figure(figsize=(3.4,2.1))
sns.lineplot(data=daily_data1,x='date',y='C1_total_cost')
plt.fill_between( daily_data1['date'], daily_data1['C1_total_cost'], color=cbcolors[0], alpha=0.2)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m'))  
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xlim(daily_data3['date'].min(), daily_data3['date'].max())         
plt.axhline(y=0,color='black',linewidth=0.5)
plt.ylabel('Cost (€/day)')
plt.xlabel('Month', labelpad=-8, x=1.07)  
plt.subplots_adjust(left=0.18, right=0.88, top=0.97, bottom=0.20)
plt.show()

plt.figure(figsize=(3.4,2.1))
sns.lineplot(data=daily_data2,x='date',y='C2_total_cost')
plt.fill_between(daily_data2['date'], daily_data2['C2_total_cost'], color=cbcolors[0], alpha=0.2)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m'))  
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xlim(daily_data3['date'].min(), daily_data3['date'].max())       
plt.axhline(y=0,color='black',linewidth=0.5)
plt.ylabel('Cost (€/day)')
plt.xlabel('Month', labelpad=-8, x=1.07)  
plt.subplots_adjust(left=0.18, right=0.88, top=0.97, bottom=0.20)
plt.show()

plt.figure(figsize=(3.4, 2.1))
sns.lineplot(data=daily_data3, x='date', y='C3_total_cost')
plt.fill_between(daily_data3['date'], daily_data3['C3_total_cost'], color=cbcolors[0], alpha=0.2)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xlim(daily_data3['date'].min(), daily_data3['date'].max())
plt.axhline(y=0, color='black', linewidth=0.5)
plt.ylabel('Cost (€/day)')
plt.xlabel('Month', labelpad=-8, x=1.07)  
plt.subplots_adjust(left=0.18, right=0.88, top=0.97, bottom=0.20)
plt.show()



labels = ['Case Base','Case 1','Case 2','Case 3']
plt.figure(figsize=(3.8,2.1))
sns.barplot(data=df_long,x=labels,y='Value',palette=cbcolors)
plt.axhline(y=0,color='black',linewidth=0.5)
plt.subplots_adjust(left=0.19, right=0.98, top=0.97, bottom=0.18)
plt.ylabel('Cost (€/year)')





