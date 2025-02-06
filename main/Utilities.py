# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:37:28 2023

@author: Daniel V Pombo (EPRI)

.\ Utilities.py

Generic functions and utilities used recurively.
"""

import pandas as pd
from pyomo.environ import value

def MeteoAPI2df(var,long,lat,days):
    '''
    Prediction of any wether variable to 16 days.
    3 inputs required:
        var = global_tilted_irradiance
        lat = 41.67
        long = 0.47
        days = 2
    Web info : https://open-meteo.com/en/docs#hourly=global_tilted_irradiance 
    '''
    import json
    import requests

    uri = "https://api.open-meteo.com/v1/forecast?latitude="+lat+"&longitude="+long+"&hourly="+var+"&forecast_days="+days+""

    def download_and_read_json(uri):
        try:
            response = requests.get(uri)
            response.raise_for_status()  
    
            data = response.json()
            
            return data
        
        except requests.exceptions.RequestException as e:
            print(f"Download error: {e}")
        except ValueError as e:
            print(f"Error reading JSON file: {e}")
    
    
        
    json_data = download_and_read_json(uri)
    
    df = pd.DataFrame({'time': json_data['hourly']['time'],
                       'global_tilted_irradiance': json_data['hourly']['global_tilted_irradiance']})
    
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%dT%H:%M')
    
    df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M')
    
    return df



def REEAPI2df(start_date, end_date):
    ''' 
    PVPC prices €/MWh in advanced and historical, source:REE. 
    2 inputs:
        
        start_date = 2024-07-23T00:00
        end_date = 2024-07-23T10:00
    
    Web info: https://www.ree.es/es/apidatos 

    '''
    import json
    import requests

    uri = "https://apidatos.ree.es/es/datos/mercados/precios-mercados-tiempo-real?start_date="+start_date+"&end_date="+end_date+"&time_trunc=hour"
    
    def download_and_read_json(uri):
        try:
            response = requests.get(uri)
            response.raise_for_status()  
    
            data = response.json()
    
            return data
        except requests.exceptions.RequestException as e:
            print(f"Download error: {e}")
        except ValueError as e:
            print(f"Error reading JSON file: {e}")
        
    
    json_data = download_and_read_json(uri)
    
    values_list = []
    for item in json_data["included"]:
        if item["type"] == "PVPC (€/MWh)" and "attributes" in item and "values" in item["attributes"]:
            values_list.extend(item["attributes"]["values"])
    
    # Convertir la lista en un DataFrame
    df = pd.DataFrame(values_list)
    
    # Convertir la columna 'datetime' a formato de fecha y hora
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%dT%H:%M:%S.%f%z')
    
    # Formatear la columna 'datetime' a 'yyyy-mm-dd hh:mm'
    df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M')
    
    return df


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
    
    if not file==None:
        df_out.to_csv(file+'.csv')
        df_param.to_csv(file+'_param.csv')
        df_size.to_csv(file+'_size.csv')
        results.write(filename=file+'_results.txt')
        with open(file+'_results.txt','a') as f:
            f.write('\nExecution time:\n'+str(exec_time)+' s\n')
            f.write('\nGOAL VALUE:\n'+str(value(instance.goal))+'\n')
            f.close()
        model_to_file(instance,file+'_model.txt')
        
    return df_out, df_param, df_size

def get_n_variables(model):
    
    n_vars = 0
    # n_pars = 0
    
    for i in range(len(model._decl_order)):
        e = model._decl_order[i][0]
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
                n_vars = n_vars + len(vals)
                
            # elif "pyomo.core.base.param.IndexedParam" in str(v.type): #Param(t)
            #     vals = v.extract_values()
            #     n_pars = n_pars + len(vals)
                
            elif "pyomo.core.base.var.ScalarVar" in str(v.type): #Var
                vals = v.get_values()
                n_vars = n_vars + 1
                
            # elif "pyomo.core.base.param.ScalarParam" in str(v.type): #Param
            #     vals = v.extract_values()
            #     n_pars = n_pars + 1
                
                
    return n_vars#, n_pars


def sol_read(filename, model):
    import pyomo.environ
    from pyomo.core import ComponentUID
    from pyomo.opt import ProblemFormat, ReaderFactory, ResultsFormat
    from pyomo.core.base.var import _GeneralVarData
    from pyomo.core import SymbolMap
    from six.moves import cPickle as pickle
    import pandas as pd

    """
    Reads a .sol solution file and returns a DataFrame with the variables.

    Parameters: 
        filename (str): Name of the file without extension. Note that all .nl, .sol, and .col must have the same name.
        model (pyomo.environ.AbstractModel): Pyomo model with variables and parameters.

    Returns:
        pd.DataFrame: DataFrame with variables and their values for each step time.
        str: Status description from the solver.
    """

    # Generating mapping file -
    def write_nl(model, nl_filename, **kwds):
        symbol_map_filename = nl_filename + ".symbol_map.pickle"
        _, smap_id = model.write(nl_filename, format=ProblemFormat.nl, io_options=kwds)
        symbol_map = model.solutions.symbol_map[smap_id]

        tmp_buffer = {}  # To speed up the process

        symbol_cuid_pairs = tuple(
            (symbol, ComponentUID(var, cuid_buffer=tmp_buffer))
            for symbol, var_weakref in symbol_map.bySymbol.items()
            if isinstance((var := var_weakref()), _GeneralVarData)  # Filter only variables
        )

        with open(symbol_map_filename, "wb") as f:
            pickle.dump(symbol_cuid_pairs, f)

        return symbol_map_filename

    # Reading .sol file and returning results --- 
    def read_sol(model, sol_filename, symbol_map_filename, suffixes=[".*"]):
        if suffixes is None:
            suffixes = []

        with ReaderFactory(ResultsFormat.sol) as reader:
            results = reader(sol_filename, suffixes=suffixes)

        with open(symbol_map_filename, "rb") as f:
            symbol_cuid_pairs = pickle.load(f)

        symbol_map = SymbolMap()
        symbol_map.addSymbols((cuid.find_component(model), symbol) for symbol, cuid in symbol_cuid_pairs)
        results._smap = symbol_map

        return results

    # Reading the .col file to extract variable names
    def read_col_file(col_filename):
        with open(col_filename, "r") as col_file:
            variable_names = [line.strip() for line in col_file.readlines()]
        return variable_names

    # --- If Var not initialized, initialize to 0
    for v in model.component_objects(pyomo.environ.Var, active=True):
        for index in v:
            if v[index].value is None:
                v[index].set_value(0.0)  # Initialize variables to 0 if they have no value

    # 1. Reading symbol_map_filename
    nl_filename = filename + '.nl'
    col_filename = filename + '.col'
    symbol_map_filename = write_nl(model, nl_filename)

    # 2. Reading .sol
    sol_filename = filename + ".sol"
    symbol_map_filename = filename + ".nl.symbol_map.pickle"
    results = read_sol(model, sol_filename, symbol_map_filename)

    # Extract solver condition directly from results
    condition = results['Solver'][0]
    # 3. Reading variable names from .col file
    variable_names = read_col_file(col_filename)

    # 4. Reading the variable values from the .sol file
    variable_values = {}

    for solution in results['Solution']:
        for idx, (var, value) in enumerate(solution['Variable'].items()):
            if idx < len(variable_names):  # Ensure there is a mapping available
                real_var_name = variable_names[idx]  # Assign the correct name from .col
                variable_values[real_var_name] = value['Value']  # Store in dictionary

    # 5. Organizing data for DataFrame
    organized_data = {}
    max_time_index = 0

    for var, value in variable_values.items():
        # Extract base variable name without temporal index
        if '[' in var and ']' in var:
            base_name = var.split('[')[0]
            time_index = int(var.split('[')[1].split(']')[0])
        else:
            base_name = var
            time_index = 0  # Non-temporal variable

        # Initialize variable list
        if base_name not in organized_data:
            organized_data[base_name] = []

        while len(organized_data[base_name]) <= time_index:
            organized_data[base_name].append(None)

        organized_data[base_name][time_index] = value

        max_time_index = max(max_time_index, time_index)

    # 6. Adjusting variable index
    for key in organized_data:
        while len(organized_data[key]) <= max_time_index:
            organized_data[key].append(None)

    # 7. Converting to data frame
    df = pd.DataFrame(organized_data)

    # Return the DataFrame and a clear condition message
    return df,condition
