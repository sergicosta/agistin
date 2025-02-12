# AGISTIN project 
# .\Devices\Batteries.py
"""
Battery pyomo block containing the characteristics of a battery.
"""

import pyomo.environ as pyo
from pyomo.network import Arc, Port


# data: dt, E0, Emax, SOCmin, SOCmax, Pmax
# init_data: E(t), P(t)
    
def Battery(b, t, data, init_data):
    
    """
    Simple Battery.
    
    Modifies its energy state :math:`E(t)` from an initial state :math:`E_0` according to 
    the charge or discharge power :math:`P(t)>0` is :math:`P_{ch}(t)` and :math:`P(t)<0` is :math:`P_{disc}(t)`.
    
    The state of charge `SOC(t)` is computed and taken into account as well.
    
    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: data ``dict``
    :param init_data: init_data ``dict``
    
    data
         - 'dt': time delta :math:`\Delta t`
         - 'E0': Initial energy :math:`E_0`
         - 'Emax': Maximum battery energy :math:`E_{max}`
         - 'SOCmin': Minimum allowed SOC :math:`SOC_{min}` in p.u.
         - 'SOCmax': Maximum allowed SOC :math:`SOC_{max}` in p.u.
         - 'Pmax': Maximum delivered/absorbed power :math:`P_{max}`
         
    init_data
         - 'E': Energy :math:`E(t)` as a ``list``
         - 'P': Power :math:`P(t)` as a ``list``
    
    Pyomo declarations    
        - Parameters: 
            - dt
            - E0
            - Emax
            - SOCmin
            - SOCmax
            - Pmax
            - eff_ch
            - eff_disc
        - Variables: 
            - E (t) bounded :math:`E(t) \in [E_{max}\cdot SOC_{min}, E_{max}\cdot SOC_{max}]`
            - P (t) bounded :math:`P(t) \in [-P_{max}, P_{max}]`
            - Pch (t) bounded :math:`P_{ch}(t) \in [0, P_{max}]`
            - Pdisc (t) bounded :math:`P_{disc}(t) \in [0, P_{max}]`
            - SOC (t) bounded :math:`SOC(t) \in [SOC_{min}, SOC_{max}]`
        - Ports: 
            - port_P @ P (Extensive)
        - Constraints:
            - c_P: :math:`P(t) = P_{ch}(t) - P_{disc}(t)`
            - c_P0: :math:`0 = P_{ch}(t) \cdot P_{disc}(t)`
            - c_SOC: :math:`SOC(t) = E(t) / E_{max}`
            - c_E: 
                - :math:`E(t) = E(t-1) + \Delta t \cdot P(t) \quad` if  :math:`t>0`
                - :math:`E(t) = E_0 + \Delta t \cdot P(t) \quad` otherwise
    """
    
    #b.dt = data['dt']
    
    # Parameters
    b.E0 = pyo.Param(initialize=data['E0'])
    b.Emax = pyo.Param(initialize=data['Emax'])
    b.SOCmax = pyo.Param(initialize=data['SOCmax'])
    b.SOCmin = pyo.Param(initialize=data['SOCmin'])
    b.Pmax = pyo.Param(initialize=data['Pmax'])
    b.Einst = pyo.Param(initialize = data['Einst'])
    b.Pinst = pyo.Param(initialize = data['Pinst'])
    b.rend_ch = pyo.Param(initialize = data['rend_ch'])
    b.rend_disc = pyo.Param(initialize = data['rend_disc'])
    
    # Variables
    b.E  = pyo.Var(t, initialize= init_data['E'], within=pyo.NonNegativeReals)
    b.P = pyo.Var(t, initialize= init_data['P'], bounds=(-data['Pmax'], data['Pmax']), within=pyo.Reals)
    b.Pch = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, within=pyo.NonNegativeReals)
    b.Pdisc = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, within=pyo.NonNegativeReals)
    b.SOC = pyo.Var(t, initialize={k:data['E0']/data['Emax'] for k in range(len(t))}, bounds=(data['SOCmin'], data['SOCmax']), within=pyo.NonNegativeReals)
    b.Pdim = pyo.Var(initialize = 0, bounds =(0,data['Pmax']-data['Pinst']),within = pyo.NonNegativeReals)
    b.Edim = pyo.Var(initialize = 0, bounds = (0,data['Emax']-data['Einst']),within = pyo.NonNegativeReals)
    # Ports
    b.port_P = Port(initialize={'P': (b.P, Port.Extensive)})
    
    # Constraints
    def Constraint_P(_b, _t):
        return _b.P[_t] == _b.Pch[_t] - _b.Pdisc[_t]
    b.c_P = pyo.Constraint(t, rule = Constraint_P)
    
    def Constraint_P0(_b, _t):
        return 0 == _b.Pch[_t] * _b.Pdisc[_t]
    b.c_P0 = pyo.Constraint(t, rule = Constraint_P0)
    
    def Constraint_SOC(_b, _t):
        return _b.SOC[_t] == _b.E[_t] / (_b.Einst + _b.Edim)
    b.c_SOC = pyo.Constraint(t, rule = Constraint_SOC)
      
    def Constraint_consume(_b,_t):
        return _b.Pch[_t] <= (_b.Pinst + _b.Pdim)
    
    b.Consume = pyo.Constraint(t, rule = Constraint_consume)
    
    def ConstraintEnergy_max(_b,_t):
        return _b.E[_t] <= (_b.Einst + _b.Edim)*_b.SOCmax
    
    b.MaxEnergy = pyo.Constraint(t, rule = ConstraintEnergy_max)
    
    def ConstraintEnergymin(_b,_t):
        return _b.E[_t] >= (_b.Einst + _b.Edim)*_b.SOCmin
    b.MinEnergy = pyo.Constraint(t, rule = ConstraintEnergymin)
    

    def Constraint_E(_b, _t): # Auskommentiert M.Valois 080324
        if _t>0:
            return _b.E[_t] == _b.E[_t-1] + _b.dt*(_b.Pch[_t]*_b.rend_ch - _b.Pdisc[_t]*_b.rend_disc)
        else:
            return b.E[_t] == _b.E0 + _b.dt*(_b.Pch[_t]*_b.rend_ch - _b.Pdisc[_t]*_b.rend_disc)
    b.c_E = pyo.Constraint(t, rule = Constraint_E)
    #Dim constraints
    
    def Constraint_prodc(_b,_t):
        return  _b.Pdch[_t] <= (_b.Pinst + _b.Pdim)
    
    b.Prod = pyo.Constraint(t, rule = Constraint_prodc)
    
  
#%% Battery_Ex0

def Battery_Ex0(b, t, data, init_data):
    """
    Simple Battery for testing and example purposes.
    It is used in Example21.
    
    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: data ``dict``
    :param init_data: init_data ``dict``
        
    data:
         - 'E0': Initial energy :math:`E_0`
         - 'Emax': Maximum battery energy :math:`E_{max}`
         - 'SOCmin': Minimum allowed SOC :math:`SOC_{min}` in p.u.
         - 'SOCmax': Maximum allowed SOC :math:`SOC_{max}` in p.u.
         - 'Pmax': Maximum delivered/absorbed power :math:`P_{max}`
         - 'Pinst': Power already installed :math:`P_{inst}`
         - 'Einst': Energy storage already installed :math:`E_{inst}`
             
    init_data:
         - 'E': Energy :math:`E(t)` as a ``list``
         - 'P': Power :math:`P(t)` as a ``list``
         
    Pyomo declaration
        - Parameters: 
            - dt
            - E0
            - Emax
            - SOCmin
            - SOCmax
            - Pmax
            - Pdim
            - Edim
            - eff_ch
            - eff_disc
        - Variables: 
            - E (t) bounded :math:`E(t) \in [E_{max}\cdot SOC_{min}, E_{max}\cdot SOC_{max}]`
            - P (t) bounded :math:`P(t) \in [-P_{max}, P_{max}]`
            - Pch (t) bounded :math:`P_{ch}(t) \in [0, P_{max}]`
            - Pdisc (t) bounded :math:`P_{disc}(t) \in [0, P_{max}]`
            - SOC (t) bounded :math:`SOC(t) \in [SOC_{min}, SOC_{max}]`
            - Edim bounded :math:`E_{dim} \in [0, E_ {max} - E_{inst}]`
            - Pdim bounded :math:`P_{dim} \in [0, P_ {max} - P_{inst}]`
        - Ports: 
            - port_P @ P (Extensive)
        - Constraints:
            - c_P: :math:`P(t) = P_{ch}(t) - P_{disc}(t)`
            - c_P0: :math:`0 = P_{ch}(t) \cdot P_{disc}(t)`
            - c_SOC: :math:`SOC(t) = E(t) /(E_{dim}+E_{inst}`
            - c_ch: :math:`Pch(t) \leq (P{inst} + P{dim})`
            - c_disc: :math:`Pdisc(t) \leq (P{inst} + P{dim})`
            - c_Emax: :math:`E(t) \leq (E{inst} + E{dim})\cdot SOC{max}`
            - c_Emin: :math:`E(t) \leq (E{inst} + E{dim})\cdot SOC{min}`
            - c_E: 
                - :math:`E(t) = E(t-1) + \Delta t \cdot P(t) \quad` if  :math:`t>0`
                - :math:`E(t) = E_0 + \Delta t \cdot P(t) \quad` otherwise

     """       
                

    
    # Parameters
    b.E0 = pyo.Param(initialize=data['E0'])
    b.SOCmax = pyo.Param(initialize=data['SOCmax'])
    b.SOCmin = pyo.Param(initialize=data['SOCmin'])
    b.Pmax = pyo.Param(initialize=data['Pmax'])
    b.Emax = pyo.Param(initialize=data['Emax'])
    b.Einst = pyo.Param(initialize = data['Einst'])
    b.Pinst = pyo.Param(initialize = data['Pinst'])
    b.rend_ch = pyo.Param(initialize = data['rend_ch'])
    b.rend_disc = pyo.Param(initialize = data['rend_disc'])
    
    # Variables
    b.E  = pyo.Var(t, initialize= init_data['E'], within=pyo.NonNegativeReals)
    b.P = pyo.Var(t, initialize= init_data['P'], bounds = (-data['Pmax'],data['Pmax']), within=pyo.Reals)
    b.Pch = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, within=pyo.NonNegativeReals)
    b.Pdisc = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, within=pyo.NonNegativeReals)
    b.SOC = pyo.Var(t, initialize={k:init_data['E'][h]/(data['Emax']) for h,k in enumerate(range(len(t)))}, bounds=(data['SOCmin'], data['SOCmax']), within=pyo.NonNegativeReals)
    b.Pdim = pyo.Var(initialize = 0, bounds =(0,data['Pmax']-data['Pinst']),within = pyo.NonNegativeReals)
    b.Edim = pyo.Var(initialize = 0, bounds = (0,data['Emax']-data['Einst']),within = pyo.NonNegativeReals)

    # Ports
    b.port_P = Port(initialize={'P': (b.P, Port.Extensive)})

    # Constraints
    def Constraint_P(_b, _t):
        return _b.P[_t] == _b.Pch[_t]*_b.rend_ch - _b.Pdisc[_t]*_b.rend_disc
    b.c_P = pyo.Constraint(t, rule = Constraint_P)
    
    def Constraint_P0(_b, _t):
        return 0 == _b.Pch[_t] * _b.Pdisc[_t]
    b.c_P0 = pyo.Constraint(t, rule = Constraint_P0)
    
    def Constraint_SOC(_b, _t):
        return _b.SOC[_t] == _b.E[_t] /(_b.Einst + _b.Edim)
    b.c_SOC = pyo.Constraint(t, rule = Constraint_SOC)
    
    def Constraint_E(_b, _t):
        if _t>0:
            return _b.E[_t] == _b.E[_t-1] +(_b.Pch[_t]*_b.rend_ch - _b.Pdisc[_t]*_b.rend_disc)
        else:
            return b.E[_t] == _b.E0 + (_b.Pch[_t]*_b.rend_ch - _b.Pdisc[_t]*_b.rend_disc)
    b.c_E = pyo.Constraint(t, rule = Constraint_E)

    
    def Constraint_ch(_b,_t):
        return abs(_b.Pch[_t]) <= (_b.Pinst + _b.Pdim)
    
    b.Consume = pyo.Constraint(t, rule = Constraint_ch)
    
    def Constraint_disc(_b,_t):
        return  abs(_b.Pdisc[_t]) <= (_b.Pinst + _b.Pdim)
    
    b.Prod = pyo.Constraint(t, rule = Constraint_disc)
    
    def ConstraintE_max(_b,_t):
        return _b.E[_t] <= (_b.Einst + _b.Edim)*_b.SOCmax
    
    b.MaxEnergy = pyo.Constraint(t, rule = ConstraintE_max)
    
    def ConstraintE_min(_b,_t):
        return _b.E[_t] >= (_b.Einst + _b.Edim)*_b.SOCmin
    b.MinEnergy = pyo.Constraint(t, rule = ConstraintE_min)
  
#__________________________________________________________________________________________________________________________________   
#%% Battery_MV 

def Battery_MV(b, t, data, init_data):
    
    # """
    # This battery model in intended to be used for he optmization examples using frequency data, for example
    # for the provision of frecuency containtment reserve. The model was implemented by Manuel Valois he 19.03.2024
    
    # The current version is missing he deliitaion of a maximun and minimium SOC 
    
    # :param b: pyomo ``Block()`` to be set
    # :param t: pyomo ``Set()`` referring to time
    # :param data: data ``dict``
    # :param init_data: init_data ``dict``
        
    # data:
    #      - 'E0': Initial energy :math:`E_0`
    #      - 'Emax': Maximum battery energy :math:`E_{max}`
    #      - 'SOCmin': Minimum allowed SOC :math:`SOC_{min}` in p.u.
    #      - 'SOCmax': Maximum allowed SOC :math:`SOC_{max}` in p.u.
    #      - 'Pmax': Maximum delivered/absorbed power :math:`P_{max}`
    #      - 'Pinst': Power already installed :math:`P_{inst}`
    #      - 'Einst': Energy storage already installed :math:`E_{inst}`
             
    # init_data:
    #      - 'E': Energy :math:`E(t)` as a ``list``
    #      - 'P': Power :math:`P(t)` as a ``list``
         
    # Pyomo declaration
    #     - Parameters: 
    #         - dt
    #         - E0
    #         - Emax
    #         - Pmax
    #         - Pdim
    #         - Edim
    #         - eff_ch
    #         - eff_disc
    #     - Variables: 
    #         - E (t) bounded :math:`E(t) \in [E_{max}\cdot SOC_{min}, E_{max}\cdot SOC_{max}]`
    #         - P (t) bounded :math:`P(t) \in [-P_{max}, P_{max}]`
    #         - Pch (t) bounded :math:`P_{ch}(t) \in [0, P_{max}]`
    #         - Pdisc (t) bounded :math:`P_{disc}(t) \in [0, P_{max}]`
    #         - SOC (t) bounded :math:`SOC(t) \in [SOC_{min}, SOC_{max}]`
    #         - Edim bounded :math:`E_{dim} \in [0, E_ {max} - E_{inst}]`
    #         - Pdim bounded :math:`P_{dim} \in [0, P_ {max} - P_{inst}]`
    #     - Ports: 
    #         - port_P @ P (Extensive)
    #     - Constraints:
    #         - c_P: :math:`P(t) = P_{ch}(t) - P_{disc}(t)`
    #         - c_P0: :math:`0 = P_{ch}(t) \cdot P_{disc}(t)`
    #         - c_SOC: :math:`SOC(t) = E(t) /(E_{dim}+E_{inst}`
    #         - c_ch: :math:`Pch(t) \leq (P{inst} + P{dim})`
    #         - c_disc: :math:`Pdisc(t) \leq (P{inst} + P{dim})`
    #         - c_Emax: :math:`E(t) \leq (E{inst} + E{dim})\cdot SOC{max}`
    #         - c_Emin: :math:`E(t) \leq (E{inst} + E{dim})\cdot SOC{min}`
    #         - c_E: 
    #             - :math:`E(t) = E(t-1) + \Delta t \cdot P(t) \quad` if  :math:`t>0`
    #             - :math:`E(t) = E_0 + \Delta t \cdot P(t) \quad` otherwise

    #  """     
    
    # Parameters
    b.E0 = pyo.Param(initialize=data['E0'])       # Energy storage at the battery at the beggiing of the simulation (_t=0)
    b.slope_fcr = pyo.Param(initialize=data['slope_fcr']) # statik for the FCR ( 100%  power by a 200mH deviation)
    b.Fup = pyo.Param(t, initialize=data['FCharge'])  # Grid frequency > 50 (time serie)
    b.Fdown = pyo.Param(t, initialize=data['FDisCharge'])  # Grid frequency < 50 (time serie)


    # Variables
    b.Pdim_FCR = pyo.Var(initialize=0, bounds=(0, 4.8)) # Power capacity of the battery for FCR provision , bounds=(0, 480000)
    b.Edim_FCR = pyo.Var(initialize=0)  # Power capacity of the battery for FCR provision
    b.P_FCRCharge = pyo.Var( t, initialize={k:0.0 for k in range(len(t))})  # Power demanded according the current frequency
    b.P_FCRDisCharge = pyo.Var( t, initialize={k:0.0 for k in range(len(t))}, bounds=(-4.80, 0))  # Power demanded according the current frequency
    b.EnerStorg = pyo.Var(t,initialize={k:0.0 for k in range(len(t))}, within=pyo.NonNegativeReals) #  bounds = (249,237500)
    b.Pout = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, within=pyo.Reals)

    # Ports
    b.port_P = Port(initialize={'P': (b.Pout, Port.Extensive)})

    def Constraint_Pout(b, t,):
            return b.Pout[t] == (b.P_FCRCharge[t] + b.P_FCRDisCharge[t])
    b.c_Pout = pyo.Constraint(t, rule = Constraint_Pout)
    def Constraint_P_FCRCharge(b, t):
        return b.P_FCRCharge[t] <= b.Pdim_FCR
    b.c_P_FCRCharge = pyo.Constraint(t, rule=Constraint_P_FCRCharge)

    def Constraint_P_FCRCharge_1(b, t,):
                return b.P_FCRCharge[t] <= (50 - (b.Fup[t])) * (b.Pdim_FCR) * b.slope_fcr
    b.c_P_FCRCharge_1 = pyo.Constraint(t, rule = Constraint_P_FCRCharge_1)

    def Constraint_P_FCRDisCharge(b, t,):
                return b.P_FCRDisCharge[t] >= (50 - (b.Fdown[t])) * (b.Pdim_FCR) * b.slope_fcr
    b.c_P_FCRDisCharge = pyo.Constraint(t, rule = Constraint_P_FCRDisCharge)

    def Constraint_EnerStorg(b, t):
        if t==0:
           return b.EnerStorg[t] == (b.E0) + (b.P_FCRCharge[t]) + (b.P_FCRDisCharge[t])
        else:
           return b.EnerStorg[t]  ==  b.EnerStorg[t-1] + (b.P_FCRCharge[t]) + (b.P_FCRDisCharge[t])                           # This constraint calculated the current SOC of the battery
    b.c_EnerStorg= pyo.Constraint(t, rule = Constraint_EnerStorg)

    #def Constraint_Edim_FCR(b):
    #    return  b.Edim_FCR <= 4*(b.Pdim_FCR)
    #b.c_Edim_FCR = pyo.Constraint (rule = Constraint_Edim_FCR)

    def Constraint_Edim_FCR_1(b, t):
        return   b.EnerStorg[t] <= b.Edim_FCR
    b.c_Edim_FCR_1 = pyo.Constraint(t, rule = Constraint_Edim_FCR_1)



