# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:37:28 2023

@author: Daniel V Pombo (EPRI)

.\ Utilities.py

Generic functions and utilities used recurively.
"""

import pandas as pd
from pyomo.environ import value

def clear_clc(): 
    """
    Applies clear and clc similar to Matlab in Spyder
    """
    try:
        from IPython import get_ipython
        get_ipython().magic('clear')
        get_ipython().magic('reset -f')
    except:
        pass 


def model_to_file(model, filename):
    with open(filename,'w') as f:
        model.pprint(f)
        

def get_results(file, instance, results, l_t, exec_time):
    df_out = pd.DataFrame(l_t, columns=['t'])
    df_param = pd.DataFrame()
    df_size = pd.DataFrame()
    for i in range(len(instance._decl_order)):
        e = instance._decl_order[i][0]
        if e is None:
            continue
        name = e.name
        
        if "pyomo.core.base.block.ScalarBlock" not in str(e.type):
            continue
        
        for ii in range(len(e._decl_order)):
            v = e._decl_order[ii][0]
            vals = 0
            
            if "pyomo.core.base.var.IndexedVar" in str(v.type): #Var(t)
                vals = v.get_values()
            elif "pyomo.core.base.param.IndexedParam" in str(v.type): #Param(t)
                vals = v.extract_values()
            elif "pyomo.core.base.var.ScalarVar" in str(v.type): #Var
                vals = v.get_values()
                df_size = pd.concat([df_size, pd.DataFrame.from_dict(vals, orient='index', columns=[v.name])], axis=1)
                continue
            elif "pyomo.core.base.param.ScalarParam" in str(v.type): #Param
                vals = v.extract_values()
                df_param = pd.concat([df_param, pd.DataFrame.from_dict(vals, orient='index', columns=[v.name])], axis=1)
                continue
            else:
                continue
            
            df_out = pd.concat([df_out, pd.DataFrame.from_dict(vals, orient='index', columns=[v.name])], axis=1)
    
    
    file = './results/LesPlanes_PVPump_and_Reversible'
    df_out.to_csv(file+'.csv')
    df_param.to_csv(file+'_param.csv')
    df_size.to_csv(file+'_size.csv')
    results.write(filename=file+'_results.txt')
    with open(file+'_results.txt','a') as f:
        f.write('\nExecution time:\n'+str(exec_time)+' s\n')
        f.write('\nGOAL VALUE:\n'+str(value(instance.goal))+'\n')
        f.close()
        
    return df_out, df_param, df_size