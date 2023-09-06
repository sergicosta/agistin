"""
AGISTIN project 

.\DataParser.py

DataParser generates a .json file that feeds the builder function.
"""

import pandas as pd

NameTest = 'Example3'
T = 5
dt = 1

df = pd.read_excel(f'Cases/{NameTest}.xlsx', sheet_name=None)
df_time = pd.read_excel(f'Cases/{NameTest}_time.xlsx', sheet_name=None)

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
            f.write('},\n')
            f.write('\t "init_data":{},\n')
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