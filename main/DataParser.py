"""
AGISTIN project 

.\DataParser.py

DataParser generates a .json file form excel files. The .json generated feeds 
the builder function.
"""

import pandas as pd

def write_list(f, df, df_time, k, val):
    for it in df_time[k]:
        aux = it.split('_')
        if aux[0] == df[k]["Name"][val]:
            f.write(f'"{aux[1]}":{list(df_time[k][it])},')

NameTest = 'Test1'
T = 5
dt = 1

df = pd.read_excel(f'Cases/{NameTest}.xlsx', sheet_name=None)
df_time = pd.read_excel(f'Cases/{NameTest}_time.xlsx', sheet_name=None)
special = ['SolarPV','Source']

with open(f'Cases/{NameTest}.json', 'w') as f:
    f.write('{\n')
    for k in df.keys():
        for val in range(len(df[k])):
            f.write(f'"{df[k]["Name"][val]}":{{\n')
            f.write(f'\t "data":{{"type":"{k}",')
            for it in df[k].columns.values:
                if it in ('Name','CONNECTION'):
                    pass
                else:
                    f.write(f'"{it}":{df[k][it][val]},')
            if k == 'Reservoir':
                f.write(f'"dt":{dt}')
            if k in special:
                #  Time values as data
                write_list(f, df, df_time, k, val)
            f.write('},\n')
            #  Time values as Initial data
            f.write('\t "init_data":{')
            if k not in special:
                write_list(f, df, df_time, k, val)
            f.write('},\n')
            #  CONNECTIONS
            f.write('\t "conns":{')
            try:
                con = df[k]['CONNECTION'][val]
                cons = con.split(';')
                for aux in cons:
                    if len(aux) == 0:
                        pass
                    else:
                        trp = aux.split(',')
                        f.write(f'"{trp[0]}":["{trp[1]}","{trp[2]}"],')
            except KeyError: # no CONNECTION
                pass
            except AttributeError: # CONNECTION is NaN
                pass 
            f.write('}\n')
            f.write('\t },\n')
    f.write('}\n')