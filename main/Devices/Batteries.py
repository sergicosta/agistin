# AGISTIN project 
# .\Devices\Batteries.py
"""
Battery pyomo block containing the characteristics of a battery.
"""

import pyomo.environ as pyo
from pyomo.network import Arc, Port



def Battery(b, t, data, init_data):
    """
    Simple Battery for testing and example purposes.
    It is used in Example21.
    
    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: data ``dict``
    :param init_data: init_data ``dict``
        
    data:
         - 'dt': time step :math:`\Delta t`
         - 'E0': Initial energy :math:`E_0`
         - 'SOCmin': Minimum allowed SOC :math:`SOC_{min}` in p.u.
         - 'SOCmax': Maximum allowed SOC :math:`SOC_{max}` in p.u.
         - 'Pmax': Maximum delivered/absorbed power :math:`P_{max}`
         - 'Emax': Maximum battery energy :math:`E_{max}`
         - 'Pinst': Power already installed :math:`P_{inst}`
         - 'Einst': Energy storage already installed :math:`E_{inst}`
         - 'eff_ch': Effiency of charge :math:`\eta_{ch}`
         - 'eff_dc': Effiency of discharge :math:`\eta_{dc}`
             
    init_data:
         - 'E': Energy :math:`E(t)` as a ``list``
         - 'P': Power :math:`P(t)` as a ``list``
         
    Pyomo declaration
        - Parameters: 
            - dt
            - E0
            - SOCmin
            - SOCmax
            - Pmax
            - Emax
            - Pinst
            - Einst
            - eff_ch
            - eff_dc
        - Variables: 
            - E (t) bounded :math:`E(t) \in [E_{max}\, SOC_{min}, E_{max}\, SOC_{max}]`
            - P (t) bounded :math:`P(t) \in [-P_{max}, P_{max}]`
            - Pch (t) bounded :math:`P_{ch}(t) \in [0, P_{max}]`
            - Pdc (t) bounded :math:`P_{dc}(t) \in [0, P_{max}]`
            - SOC (t) bounded :math:`SOC(t) \in [SOC_{min}, SOC_{max}]`
            - Edim bounded :math:`E_{dim} \in [0, E_{max} - E_{inst}]`
            - Pdim bounded :math:`P_{dim} \in [0, P_{max} - P_{inst}]`
        - Ports: 
            - port_P @ P (Extensive)
        - Constraints:
            - c_P: :math:`P(t) = P_{ch}(t)\, \eta_{ch} - P_{disc}(t)\, \eta_{dc}`
            - c_P0: :math:`0 = P_{ch}(t) \, P_{dc}(t)`
            - c_SOC: :math:`SOC(t) = E(t) /(E_{dim}+E_{inst})`
            - c_ch: :math:`P_{ch}(t) \leq (P_{inst} + P_{dim})`
            - dc: :math:`P_{dc}(t) \leq (P_{inst} + P_{dim})`
            - c_Emax: :math:`E(t) \leq (E{inst} + E{dim})\, SOC{max}`
            - c_Emin: :math:`E(t) \geq (E{inst} + E{dim})\, SOC{min}`
            - c_E: 
                - :math:`E(t) = E(t-1) + \Delta t \, P(t) \quad` if  :math:`t>0`
                - :math:`E(t) = E_0 + \Delta t \, P(t) \quad` otherwise

     """       
    
    # Parameters
    b.dt = pyo.Param(initialize=data['dt'])
    b.E0 = pyo.Param(initialize=data['E0'])
    b.SOCmax = pyo.Param(initialize=data['SOCmax'])
    b.SOCmin = pyo.Param(initialize=data['SOCmin'])
    b.Pmax = pyo.Param(initialize=data['Pmax'])
    b.Emax = pyo.Param(initialize=data['Emax'])
    b.Pinst = pyo.Param(initialize = data['Pinst'])
    b.Einst = pyo.Param(initialize = data['Einst'])
    b.eff_ch = pyo.Param(initialize = data['eff_ch'])
    b.eff_dc = pyo.Param(initialize = data['eff_dc'])
    
    # Variables
    b.E  = pyo.Var(t, initialize= init_data['E'], within=pyo.NonNegativeReals)
    b.P = pyo.Var(t, initialize= init_data['P'], bounds = (-data['Pmax'],data['Pmax']), within=pyo.Reals)
    b.Pch = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, bounds = (0,data['Pmax']), within=pyo.NonNegativeReals)
    b.Pdc = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, bounds = (0,data['Pmax']), within=pyo.NonNegativeReals)
    b.SOC = pyo.Var(t, initialize={k:init_data['E'][h]/(data['Einst']) for h,k in enumerate(range(len(t)))}, bounds=(data['SOCmin'], data['SOCmax']), within=pyo.NonNegativeReals)
    b.Pdim = pyo.Var(initialize = 0, bounds =(0,data['Pmax']-data['Pinst']),within = pyo.NonNegativeReals)
    b.Edim = pyo.Var(initialize = 0, bounds = (0,data['Emax']-data['Einst']),within = pyo.NonNegativeReals)

    # Ports
    b.port_P = Port(initialize={'P': (b.P, Port.Extensive)})

    # Constraints
    def Constraint_P(_b, _t):
        return _b.P[_t] == _b.Pch[_t]*_b.eff_ch - _b.Pdc[_t]*_b.eff_dc
    b.c_P = pyo.Constraint(t, rule = Constraint_P)
    
    def Constraint_P0(_b, _t):
        return 0 == _b.Pch[_t] * _b.Pdc[_t]
    b.c_P0 = pyo.Constraint(t, rule = Constraint_P0)
    
    def Constraint_SOC(_b, _t):
        return _b.SOC[_t] == _b.E[_t] /(_b.Einst + _b.Edim)
    b.c_SOC = pyo.Constraint(t, rule = Constraint_SOC)
    
    def Constraint_E(_b, _t):
        if _t>0:
            return _b.E[_t] == _b.E[_t-1] + _b.dt*(_b.Pch[_t]*_b.eff_ch - _b.Pdc[_t]*_b.eff_dc)
        else:
            return b.E[_t] == _b.E0 + _b.dt*(_b.Pch[_t]*_b.eff_ch - _b.Pdc[_t]*_b.eff_dc)
    b.c_E = pyo.Constraint(t, rule = Constraint_E)
    
    def Constraint_ch(_b,_t):
        return _b.Pch[_t] <= (_b.Pinst + _b.Pdim)
    b.c_ch = pyo.Constraint(t, rule = Constraint_ch)
    
    def Constraint_disc(_b,_t):
        return  _b.Pdc[_t] <= (_b.Pinst + _b.Pdim)
    b.c_dc = pyo.Constraint(t, rule = Constraint_disc)
    
    def ConstraintE_max(_b,_t):
        return _b.E[_t] <= (_b.Einst + _b.Edim)*_b.SOCmax
    b.MaxEnergy = pyo.Constraint(t, rule = ConstraintE_max)
    
    def ConstraintE_min(_b,_t):
        return _b.E[_t] >= (_b.Einst + _b.Edim)*_b.SOCmin
    b.MinEnergy = pyo.Constraint(t, rule = ConstraintE_min)
  
#__________________________________________________________________________________________________________________________________
# #%% Battery_MV 

# def Battery_MV(b, t, data, init_data):
    
#     """
#     This battery model in intended to be used for he optmization examples using frequency data, for example
#     for the provision of frecuency containtment reserve. The model was implemented by Manuel Valois he 19.03.2024
    
#     The current version is missing he deliitaion of a maximun and minimium SOC 
    
#     :param b: pyomo ``Block()`` to be set
#     :param t: pyomo ``Set()`` referring to time
#     :param data: data ``dict``
#     :param init_data: init_data ``dict``
        
#     data:
#          - 'E0': Initial energy :math:`E_0`
#          - 'Einst': Energy storage already installed :math:`E_{inst}`
#          - 'Emax': Maximum battery energy that can be installed :math:`E_{max}`
#          - 'SOCmin': Minimum allowed SOC :math:`SOC_{min}` in p.u.
#          - 'SOCmax': Maximum allowed SOC :math:`SOC_{max}` in p.u.
#          - 'Pmax': Maximum delivered/absorbed power :math:`P_{max}`
#          - 'Pinst': Power already installed :math:`P_{inst}`

             
#     init_data:
#          - 'E': Energy :math:`E(t)` as a ``list``
#          - 'P': Power :math:`P(t)` as a ``list``
         
#     Pyomo declaration
#         - Parameters: 
#             - dt
#             - E0
#             - Emax
#             - Pmax
#             - Pdim
#             - Edim
#             - eff_ch
#             - eff_disc
#         - Variables: 
#             - E (t) bounded :math:`E(t) \in [E_{max}\cdot SOC_{min}, E_{max}\cdot SOC_{max}]`
#             - P (t) bounded :math:`P(t) \in [-P_{max}, P_{max}]`
#             - Pch (t) bounded :math:`P_{ch}(t) \in [0, P_{max}]`
#             - Pdisc (t) bounded :math:`P_{disc}(t) \in [0, P_{max}]`
#             - SOC (t) bounded :math:`SOC(t) \in [SOC_{min}, SOC_{max}]`
#             - Edim bounded :math:`E_{dim} \in [0, E_ {max} - E_{inst}]` : Aditional Energy Capacity to be determined by the optimization
#             - Pdim bounded :math:`P_{dim} \in [0, P_ {max} - P_{inst}]`
#         - Ports: 
#             - port_P @ P (Extensive)
#         - Constraints:
#             - c_P: :math:`P(t) = P_{ch}(t) - P_{disc}(t)`
#             - c_P0: :math:`0 = P_{ch}(t) \cdot P_{disc}(t)`
#             - c_SOC: :math:`SOC(t) = E(t) /(E_{dim}+E_{inst}`
#             - c_ch: :math:`Pch(t) \leq (P{inst} + P{dim})`
#             - c_disc: :math:`Pdisc(t) \leq (P{inst} + P{dim})`
#             - c_Emax: :math:`E(t) \leq (E{inst} + E{dim})\cdot SOC{max}`
#             - c_Emin: :math:`E(t) \leq (E{inst} + E{dim})\cdot SOC{min}`
#             - c_E: 
#                 - :math:`E(t) = E(t-1) + \Delta t \cdot P(t) \quad` if  :math:`t>0`
#                 - :math:`E(t) = E_0 + \Delta t \cdot P(t) \quad` otherwise

#      """     
    
#     # Parameters
#     b.E0 = pyo.Param(initialize=data['E0'])       # Energy storage at the battery at the beggiing of the simulation (_t=0)
#     b.Pmax = pyo.Param(initialize=data['Pmax'])   # Battery maximun charging and discharging power  
#     b.Emax = pyo.Param(initialize=data['Emax'])   # Battery maximun energy capacity
#     b.Einst = pyo.Param(initialize=data['Einst']) # Battery installed rated energy
#     b.Pinst = pyo.Param(initialize=data['Pinst']) # Battery installed rated power
#     b.slope_fcr = pyo.Param(initialize=data['slope_fcr']) # statik for the FCR ( 100%  power by a 200mH deviation)
#     b.F = pyo.Param(t, initialize=data['F'])      # Grid frequency (time serie)
    
#     b.SOCmin = pyo.Param(initialize=0.25*(data['Pinst'])/(data['Einst'])) 
#     b.SOCmax = pyo.Param(initialize=((data['Einst']) - 0.25*(data['Pinst']))/(data['Einst'])) # bounds = (0.0,data['SOCmax']),
    
#     # Variables
      
#     b.Pdemanded = pyo.Var(t, initialize= 0.0,  within=pyo.Reals) 
#     b.Pcharged = pyo.Var(t, initialize= 0.0,  within=pyo.NonNegativeReals)
#     b.Pdischarged = pyo.Var(t, initialize= 0.0,  within=pyo.NonNegativeReals)
#     #b.EstrgOut = pyo.Var(t, initialize= 0.0,  within=pyo.NonNegativeReals)
#     b.EstrgOut = pyo.Var(t, initialize= 0.0, bounds = (0.0,data['Einst']),  within=pyo.NonNegativeReals)
#     b.SOC = pyo.Var(t, initialize= 0.0, bounds = (data['SOCmin'],data['SOCmax']),  within=pyo.NonNegativeReals)
#     #b.SOC = pyo.Var(t, initialize= 0.0,  within=pyo.NonNegativeReals)
#     b.Pout = pyo.Var(t, initialize= 0.0,  within=pyo.Reals)
    
    
#     # Ports
#     b.port_P = Port(initialize={'P': (b.Pout, Port.Extensive)})                                     # Output signal of the model
  
#   # Constraints
#     #b.constraint_lower = Constraint(expr=b.EstrgOut[t] >= 0.0)
    
#     def Constraint_P(b, t):                                                       #This constraint calculates the amount of power to be delivered
#             return b.Pdemanded[t] == (50 - (b.F[t]))*(b.Pinst)*b.slope_fcr
#     b.Constraint_P = pyo.Constraint(t, rule=Constraint_P) 
    
#     def Constraint_Pbalance(b, t):                                                       #This constraint calculates the amount of power to be delivered
#             return b.Pdemanded[t] - b.Pcharged[t] + b.Pdischarged[t] == 0.0
#     b.Constraint_Pbalance = pyo.Constraint(t, rule=Constraint_Pbalance) 
    
#     def Constraint_Pbalance_1(b, t):                                                       #This constraint calculates the amount of power to be delivered
#             return b.Pcharged[t]*b.Pdischarged[t] == 0.0
#     b.Constraint_Pbalance_1 = pyo.Constraint(t, rule=Constraint_Pbalance_1)
 

#     def Constraint_EstrgOut(b, t): 
#         if t==0:                                               # This constraint calculated the power at the end of the time step
#            return b.EstrgOut[t] ==  (b.E0) + (50 - (b.F[t]))*(b.Pinst)*b.slope_fcr
#         else:
#            return b.EstrgOut[t] ==  b.EstrgOut[t-1] + (50 - (b.F[t]))*(b.Pinst)*b.slope_fcr
#     b.Constraint_EstrgOut = pyo.Constraint(t, rule = Constraint_EstrgOut)
     
   
#     def Constraint_SOC(b, t,):                                                    # This constraint calculated the current SOC of the battery
#             return b.SOC[t] == b.EstrgOut[t]/(b.Einst)
#     b.c_SOC = pyo.Constraint(t, rule = Constraint_SOC)

# # From this point we start having problems

# # When including this constraint, the result from t=14 to 20 are not correct
# #    the battery is supposed to be empty  

# # Problem definition: The outpout power is not calculated correctly since the SOC is also not correct.
# # The SCO and the EstrgOut are not matching, which cause the minimun and maximun SOC and not respected
# # Additionally, after the full charging or discharging of the battery, the results of the next time step are 0 or 24000, which not correct. 
# # The results after reaching the maximun or the minimun must be for example the minimum SOC + Edemanded
# # Constraint could defined within the variables or as a fuction, but some how the results are deferent, and for example the SOCmin when defined within the variable, is not taken into account
# # give a look to time step 1, 21-22, 23-26, 14-20, 49-50, which the error are usually occurring 
#     def Constraint_Pout(b, t,):          # the                                           # Based on the SCO and the power demanded, this constraint computes te amoung of energy/power that can be delivered
#         if t==0:
#             return b.Pout[t] == (50 - (b.F[t]))*(b.Pinst)*b.slope_fcr
#         else:
#             return b.Pout[t] == (b.SOC[t] - b.SOC[t-1])*(b.Einst)
#     b.c_Pout = pyo.Constraint(t, rule = Constraint_Pout)  
# """      
#     def Constraint_SOC_2(b, t,):                                                    # This constraint calculated the current SOC of the battery
#             return b.SOC[t] <= b.SOCmax
#     b.c_SOC_2 = pyo.Constraint(t, rule = Constraint_SOC_2)
    
#     def Constraint_SOC_1(b, t,):                                                    # This constraint calculated the current SOC of the battery
#             return b.SOC[t] >= b.SOCmin
#     b.c_SOC_1 = pyo.Constraint(t, rule = Constraint_SOC_1)

  
#     def Constraint_SOC_3(b, t,):                                                    # This constraint calculated the current SOC of the battery
#             return  b.EstrgOut[t] <= b.SOCmax*(b.Einst)
#     b.c_SOC_3 = pyo.Constraint(t, rule = Constraint_SOC_3)


# ----------------------------------------------------------------------------------- 









    
  
#     def Constraint_SOC_1(b, t,):                                                    # This constraint calculated the current SOC of the battery
#             return b.EstrgOut[t] >= b.SOCmin*(b.Einst)
#     b.c_SOC_1 = pyo.Constraint(t, rule = Constraint_SOC_1)



#     def Constraint_Pbalance_2(b, t):                                                       #This constraint calculates the amount of power to be delivered
#             return b.Pcharged[t] <= b.Pinst
#     b.Constraint_Pbalance_2 = pyo.Constraint(t, rule=Constraint_Pbalance_2)
    
#     def Constraint_Pbalance_3(b, t):                                                       #This constraint calculates the amount of power to be delivered
#             return b.Pdischarged[t] <= b.Pinst
#     b.Constraint_Pbalance_3 = pyo.Constraint(t, rule=Constraint_Pbalance_3)
  
#     def Constraint_EstrgOut(b, t): 
#         if t==0:                                               # This constraint calculated the power at the end of the time step
#            return b.EstrgOut[t] ==  (b.E0) + b.Pcharged[t] - b.Pdischarged[t]
#         else:
#            return b.EstrgOut[t] ==  b.EstrgOut[t-1] + b.Pcharged[t] - b.Pdischarged[t]
#     b.Constraint_EstrgOut = pyo.Constraint(t, rule = Constraint_EstrgOut)


  
    



#     def Constraint_EstrgOut_1(b, t):                                               # This constraint calculated the power at the end of the time step
#          return b.EstrgOut[t] >= b.SOCmin*b.Einst
#     b.Constraint_EstrgOut_1 = pyo.Constraint(t, rule = Constraint_EstrgOut_1)


# """    


# """
#     def Constraint_EstrgOut(b, t): 
#         if t==0:                                               # This constraint calculated the power at the end of the time step
#            return b.EstrgOut[t] ==  (b.E0) + (50 - (b.F[t]))*(b.Pinst)*b.slope_fcr
#         else:
#            return b.EstrgOut[t] ==  b.EstrgOut[t-1] + (50 - (b.F[t]))*(b.Pinst)*b.slope_fcr
#     b.Constraint_EstrgOut = pyo.Constraint(t, rule = Constraint_EstrgOut)

#     def Constraint_EstrgOut_2(b, t):                                               # This constraint calculated the power at the end of the time step
#          return b.EstrgOut[t] <= b.SOCmax*b.Einst
#     b.Constraint_EstrgOut_2 = pyo.Constraint(t, rule = Constraint_EstrgOut_2)
    

 


    
#     def Constraint_Pbalance_4(b, t):                                                       #This constraint calculates the amount of power to be delivered
#             return  b.EstrgOut[t] >= b.Pdischarged[t]
#     b.Constraint_Pbalance_4 = pyo.Constraint(t, rule=Constraint_Pbalance_4)
 

  







     
#     def Constraint_EstrgOut(b, t):                                                # This constraint calculated the power at the end of the time step
#         if t==0:
#             return b.EstrgOut[t] == b.E0 + (50 - (b.F[t]))*(b.Pinst)*b.slope_fcr 
#         else:
#             return b.EstrgOut[t] == b.EstrgOut[t-1] + (50 - (b.F[t]))*(b.Pinst)*b.slope_fcr
#     b.c_EstrgOut = pyo.Constraint(t, rule = Constraint_EstrgOut)
 



#     def Constraint_SOC(b, t,):                                                    # This constraint calculated the current SOC of the battery
#         if t==0:
#             return b.SOC[t] == (b.E0 + ((50 - (b.F[t]))*(b.Pinst)*b.slope_fcr))/ b.Einst
#         else:
#             return b.SOC[t] == (b.EstrgOut[t-1] + (50 - (b.F[t]))*(b.Pinst)*b.slope_fcr)/ b.Einst
#     b.c_SOC = pyo.Constraint(t, rule = Constraint_SOC)




#     def Constraint_Pout(_b, _t,):                                                   # Based on the SCO and the power demanded, this constraint computes te amoung of energy/power that can be delivered
#         if _t==0:
#             return _b.Pout[_t] == (50 - (_b.F[_t]))*(_b.Pinst)*_b.slope_fcr
#         else:
#             return _b.Pout[_t] == (_b.EstrgOut[_t] - _b.EstrgOut[_t-1])
#     b.c_Pout = pyo.Constraint(t, rule = Constraint_Pout) 

#     def Constraint_EstrgOut_1(_b, _t):                                                # This constraint calculated the power at the end of the time step
#            return _b.EstrgOut[_t] >= 0.0
#     b.c_EstrgOut_1 = pyo.Constraint(t, rule = Constraint_EstrgOut_1)

 
#     def Constraint_Pout(_b, _t,):                                                   # Based on the SCO and the power demanded, this constraint computes te amoung of energy/power that can be delivered
#         if _t==0:
#             return _b.Pout[_t] == (50 - (_b.F[_t]))*(_b.Pinst)*_b.slope_fcr
#         else:
#             return _b.Pout[_t] == (_b.EstrgOut[_t] - _b.EstrgOut[_t-1])
#     b.c_Pout = pyo.Constraint(t, rule = Constraint_Pout)  
    


 



 
#     def Constraint_Pout(_b, _t,):                                                   # Based on the SCO and the power demanded, this constraint computes te amoung of energy/power that can be delivered
#         if _t==0:
#             return _b.Pout[_t] == -(50 - (_b.F[_t]))*(_b.Pinst)*_b.slope_fcr
#         else:
#             return _b.Pout[_t] == (_b.EstrgOut[_t] - _b.EstrgOut[_t-1])
#     b.c_Pout = pyo.Constraint(t, rule = Constraint_Pout)   



    
#     def Constraint_SOC_1(_b, _t,):                                                    # This constraint calculated the current SOC of the battery
#             return _b.EstrgOut[_t] >= _b.SOCmin*(_b.Einst)
#     b.c_SOC_1 = pyo.Constraint(t, rule = Constraint_SOC_1) 



#     def Constraint_SOC_3(_b, _t,):                                                    # This constraint calculated the current SOC of the battery
#             return  _b.SOC[_t] <= _b.SOCmax[_t]
#     b.c_SOC_3 = pyo.Constraint(t, rule = Constraint_SOC_3)

#     def Constraint_SOC_4(_b, _t,):                                                    # This constraint calculated the current SOC of the battery
#             return  _b.SOC[_t] >= _b.SOCmin[_t]
#     b.c_SOC_4 = pyo.Constraint(t, rule = Constraint_SOC_4)
  
#  """ 
  
  
  
  
  
  
  
  
  
  
  
  
  
  





# """  
#     def Constraint_P(_b, _t):                                                       #This constraint calculates the amount of power to be delivered
#             return _b.Pdemanded[_t] == (50 - (_b.F[_t]))*(_b.Pinst)*_b.slope_fcr
#     b.Constraint_P = pyo.Constraint(t, rule=Constraint_P) 

#     def Constraint_SOCmin(_b, _t,):                                                    # This constraint calculated the current SOC of the battery
#             return _b.SOCmin[_t] == 0.25*(_b.Pinst)/b.Einst
#     b.c_SOCmin = pyo.Constraint(t, rule = Constraint_SOCmin)
    
#     def Constraint_SOCmax(_b, _t,):                                                    # This constraint calculated the current SOC of the battery
#             return _b.SOCmax[_t] == (b.Einst - 0.25*(_b.Pinst))/b.Einst
#     b.c_SOCmax = pyo.Constraint(t, rule = Constraint_SOCmax)
      
#     def Constraint_EstrgOut(_b, _t):                                                # This constraint calculated the power at the end of the time step
#         if _t==0:
#             return _b.EstrgOut[_t] == _b.E0 + (50 - (_b.F[_t]))*(_b.Pinst)*_b.slope_fcr 
#         else:
#             return _b.EstrgOut[_t] == _b.EstrgOut[_t-1] + (50 - (_b.F[_t]))*(_b.Pinst)*_b.slope_fcr
#     b.c_EstrgOut = pyo.Constraint(t, rule = Constraint_EstrgOut)

#     def Constraint_SOC_1(_b, _t,):                                                    # This constraint calculated the current SOC of the battery
#             return _b.EstrgOut[_t] >= _b.SOCmin[_t]*(_b.Einst)
#     b.c_SOC_1 = pyo.Constraint(t, rule = Constraint_SOC_1) 

#     def Constraint_Pout(_b, _t,):                                                   # Based on the SCO and the power demanded, this constraint computes te amoung of energy/power that can be delivered
#         if _t==0:
#             return _b.Pout[_t] == -(50 - (_b.F[_t]))*(_b.Pinst)*_b.slope_fcr
#         else:
#             return _b.Pout[_t] == (_b.EstrgOut[_t] - _b.EstrgOut[_t-1])
#     b.c_Pout = pyo.Constraint(t, rule = Constraint_Pout)   
    
#     def Constraint_SOC_2(_b, _t,):                                                    # This constraint calculated the current SOC of the battery
#             return  _b.EstrgOut[_t] <= _b.SOCmax[_t]*(_b.Einst)
#     b.c_SOC_2 = pyo.Constraint(t, rule = Constraint_SOC_2)
 
    
#     def Constraint_SOC(_b, _t,):                                                    # This constraint calculated the current SOC of the battery
#         if _t==0:
#             return _b.SOC[_t] == (_b.E0 + ((50 - (_b.F[_t]))*(_b.Pinst)*_b.slope_fcr))/ b.Einst
#         else:
#             return _b.SOC[_t] == (_b.EstrgOut[_t-1] + (50 - (_b.F[_t]))*(_b.Pinst)*_b.slope_fcr)/ b.Einst
#     b.c_SOC = pyo.Constraint(t, rule = Constraint_SOC)
    
#     def Constraint_SOC_3(_b, _t,):                                                    # This constraint calculated the current SOC of the battery
#             return  _b.SOC[_t] <= _b.SOCmax[_t]
#     b.c_SOC_3 = pyo.Constraint(t, rule = Constraint_SOC_3)

#     def Constraint_SOC_4(_b, _t,):                                                    # This constraint calculated the current SOC of the battery
#             return  _b.SOC[_t] >= _b.SOCmin[_t]
#     b.c_SOC_4 = pyo.Constraint(t, rule = Constraint_SOC_4)


# """





# """ 
#     def Constraint_Pout(_b, _t,):                                                   # Based on the SCO and the power demanded, this constraint computes te amoung of energy/power that can be delivered
#         if _t==0:
#             return _b.Pout[_t] == -(50 - (_b.F[_t]))*(_b.Pinst)*_b.slope_fcr
#         else:
#             return _b.Pout[_t] == (_b.SOC[_t] - _b.SOC[_t-1])*(_b.Einst)
#     b.c_Pout = pyo.Constraint(t, rule = Constraint_Pout)   
  
#     def Constraint_Pout(_b, _t,):                                                   # Based on the SCO and the power demanded, this constraint computes te amoung of energy/power that can be delivered
#         if _t==0:
#             return _b.Pout[_t] == -(50 - (_b.F[_t]))*(_b.Pinst)*_b.slope_fcr
#         else:
#             return _b.Pout[_t] == (_b.SOC[_t] - _b.SOC[_t-1])*(_b.Einst)
#     b.c_Pout = pyo.Constraint(t, rule = Constraint_Pout)   

    

 

    


    


    
 
# """
    


    
# """
#     def Constraint_P(_b, _t):                                                       #This constraint calculates the amount of power to be delivered
#         if _t==0:
#             return _b.Pdemanded[_t] == (50 - (_b.F[_t]))*(_b.Pinst)*_b.slope_fcr
#         else:
#             return _b.Pdemanded[_t] == (50 - (_b.F[_t]))*(_b.Pinst)*_b.slope_fcr
#     b.Constraint_P = pyo.Constraint(t, rule=Constraint_P) 

#     def Constraint_EstrgIni(_b, _t):                                                # This constraint calculated the power at the beginning of the time step
#         if _t==0:
#             return _b.EstrgIni[_t] == _b.E0 
#         else:
#             return _b.EstrgIni[_t] == _b.EstrgIni[_t-1] + b.Pdemanded[_t-1]
#     b.c_EstrgIni = pyo.Constraint(t, rule = Constraint_EstrgIni) 

    
#     def Constraint_EstrgOut(_b, _t):                                                # This constraint calculated the power at the end of the time step
#         if _t==0:
#             return _b.EstrgOut[_t] == _b.E0 + b.Pdemanded[_t] 
#         else:
#             return _b.EstrgOut[_t] == _b.EstrgOut[_t-1] + b.Pdemanded[_t]
#     b.c_EstrgOut = pyo.Constraint(t, rule = Constraint_EstrgOut)
    
      
#     def Constraint_SOC(_b, _t,):                                                    # This constraint calculated the current SOC of the battery
#         if _t==0:
#             return _b.SOC[_t] == (_b.E0 + b.Pdemanded[_t])/ b.Einst
#         else:
#             return _b.SOC[_t] == _b.EstrgOut[_t]/ b.Einst
#     b.c_SOC = pyo.Constraint(t, rule = Constraint_SOC)

   
#     def Constraint_Pout(_b, _t,):                                                   # Based on the SCO and the power demanded, this constraint computes te amoung of energy/power that can be delivered
#         if _t==0:
#             return _b.Pout[_t] == _b.Pdemanded[_t]
#         else:
#             return _b.Pout[_t] == (_b.SOC[_t] - _b.SOC[_t-1])*(_b.Einst)
#     b.c_Pout = pyo.Constraint(t, rule = Constraint_Pout)

# """
