# -*- coding: utf-8 -*-

import pyomo.environ as pyo

def Builder(model, l_t, *sets):
    
    model.t = pyo.Set(initialize=l_t)
    
    for st in sets:
        st.builder(model)
        
    for st in sets:
        st.builderConstr(model)