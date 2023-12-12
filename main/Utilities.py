# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 11:37:28 2023

@author: Daniel V Pombo (EPRI)

.\ Utilities.py

Generic functions and utilities used recurively.
"""

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