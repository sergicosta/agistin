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
    
    r"""
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
    r"""
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
    #b.Fdown = pyo.Param(t, initialize=data['FDisCharge'])  # Grid frequency < 50 (time serie)
    b.dt = pyo.Param(initialize=0.000277777)


    # Variables
    b.Pdim_FCR = pyo.Var(initialize=0, bounds=(0, 48000)) # Power capacity of the battery for FCR provision , bounds=(0, 480000)
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
                return b.P_FCRCharge[t] <= ((50 - (b.Fup[t])) * (b.Pdim_FCR) * b.slope_fcr)*b.dt
    b.c_P_FCRCharge_1 = pyo.Constraint(t, rule = Constraint_P_FCRCharge_1)

    def Constraint_P_FCRDisCharge(b, t,):
                return b.P_FCRDisCharge[t] >= ((50 - (b.Fdown[t])) * (b.Pdim_FCR) * b.slope_fcr)*b.dt
    b.c_P_FCRDisCharge = pyo.Constraint(t, rule = Constraint_P_FCRDisCharge)

    def Constraint_EnerStorg(b, t):
        if t==0:
           return b.EnerStorg[t] == (b.E0) + (b.P_FCRCharge[t]) + (b.P_FCRDisCharge[t])
        else:
           return b.EnerStorg[t]  ==  b.EnerStorg[t-1] + (b.P_FCRCharge[t]) + (b.P_FCRDisCharge[t])                           # This constraint calculated the current SOC of the battery
    b.c_EnerStorg= pyo.Constraint(t, rule = Constraint_EnerStorg)

    def Constraint_Edim_FCR(b):
        return  b.Edim_FCR <= 4*(b.Pdim_FCR)
    b.c_Edim_FCR = pyo.Constraint (rule = Constraint_Edim_FCR)

    def Constraint_Edim_FCR_1(b, t):
        return   b.EnerStorg[t] <= b.Edim_FCR
    b.c_Edim_FCR_1 = pyo.Constraint(t, rule = Constraint_Edim_FCR_1)


def Battery_FCR(b, t, data, init_data):

    # Parameters
    b.E0 = pyo.Param(initialize=data['E0'])
    # Build dicts for time-indexed series so Param initialization keys match the time set
    def _series_map(key, default=0.0):
        seq = list(data.get(key, []))
        t_index = list(t)
        if len(seq) != len(t_index):
            if len(seq) == 0:
                seq = [default] * len(t_index)
            else:
                reps = (len(t_index) + len(seq) - 1) // len(seq)
                seq = (seq * reps)[:len(t_index)]
        return {ti: seq[i] for i, ti in enumerate(t_index)}

    b.FCR_Neg = pyo.Param(t, initialize=_series_map('ActFCRNeg', 0.0))
    b.FCR_Pos = pyo.Param(t, initialize=_series_map('ActFCRPos', 0.0))
    b.FCR_reBAP = pyo.Param(t, initialize=_series_map('FCRreBAP', 0.0))
    b.FCR_Remuneration = pyo.Param(t, initialize=_series_map('FCRRemuneration', 0.0))


    # Variables
    b.Pdim_FCR = pyo.Var(initialize=0, bounds=(0, 40000000.8)) # Power capacity of the battery for FCR provision , bounds=(0, 480000)
    b.Edim_FCR = pyo.Var(initialize=0)  # Power capacity of the battery for FCR provision , bounds=(0, 4000000.8)
    b.P_FCRCharge = pyo.Var( t, initialize={k:0.0 for k in range(len(t))}, bounds=(0, 40000000.80))  # Power demanded according the current frequency, it is necesary to add a boud, if not it will tak negative values
    b.P_FCRDisCharge = pyo.Var( t, initialize={k:0.0 for k in range(len(t))}, bounds=(-40000000.80, 0))  # Power demanded according the current frequency
    b.EnerStorg = pyo.Var(t,initialize={k:0.0 for k in range(len(t))}, within=pyo.NonNegativeReals) #  bounds = (249,237500)
    b.Pout = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, within=pyo.Reals)
    b.P_Add_Charge = pyo.Var(t, initialize={k: 0.0 for k in range(len(t))}, within=pyo.NonNegativeReals)


    # Ports
    b.port_P = Port(initialize={'P': (b.Pout, Port.Extensive)})

    def Constraint_Pout(b, t, ):
        return b.Pout[t] == (b.P_FCRCharge[t] + b.P_FCRDisCharge[t])
    b.c_Pout = pyo.Constraint(t, rule=Constraint_Pout)
    def Constraint_P_FCRCharge_1(b, t, ):
        return b.P_FCRCharge[t] <= -1*b.FCR_Neg[t] * (b.Pdim_FCR)  # + b.P_Add_Charge[t] b.FCR_Neg[t] bedeutet, dass eine Überfrequenz vorliegt
    b.c_P_FCRCharge_1 = pyo.Constraint(t, rule=Constraint_P_FCRCharge_1)
    def Constraint_P_FCRDisCharge(b, t, ):
        return b.P_FCRDisCharge[t] >= -b.FCR_Pos[t] * (b.Pdim_FCR) # Discharge mean injecting power so, a negative value
    b.c_P_FCRDisCharge = pyo.Constraint(t, rule=Constraint_P_FCRDisCharge)
    def Constraint_P_FCRCharge(b, t):
        return b.P_FCRCharge[t] <= b.Pdim_FCR
    b.c_P_FCRCharge = pyo.Constraint(t, rule=Constraint_P_FCRCharge)
    def Constraint_EnerStorg(b, t):
        if t == 0:
            return b.EnerStorg[t] == (b.E0) + (b.P_FCRCharge[t])*0.25 + (b.P_FCRDisCharge[t])*-0.25 # It is neccesaty o multiply by 0.25, because the time steps al 15 minutes
        else:
            return b.EnerStorg[t] == b.EnerStorg[t - 1] + (b.P_FCRCharge[t])*0.25 + (b.P_FCRDisCharge[t])*0.25  # This constraint calculated the current SOC of the battery
    b.c_EnerStorg = pyo.Constraint(t, rule=Constraint_EnerStorg)
    def Constraint_Edim_FCR_1(b, t):
        return b.EnerStorg[t] <= b.Edim_FCR
    b.c_Edim_FCR_1 = pyo.Constraint(t, rule=Constraint_Edim_FCR_1)

    def Constraint_Edim_FCR_2(b, t):
        return b.Edim_FCR <= b.Pdim_FCR*3
    b.c_Edim_FCR_2 = pyo.Constraint(t, rule=Constraint_Edim_FCR_2)
    def Constraint_P_AddFCR(b, t):
        return b.P_Add_Charge[t] <= (1 - b.FCR_Neg[t] - b.FCR_Neg[t]) * b.Pdim_FCR
    b.c_P_AddFCR = pyo.Constraint(t, rule=Constraint_P_AddFCR)


#__________________________________________________________________________________________________________________________________   
#%% Battery with State of Health (SOH) Model

# def Battery_SOH(b, t, data, init_data):
#     """
#     Battery with State of Health (SOH) degradation model.
    
#     Implements the SOH equation: SOH(t) = 1 - k · t^n
#     where:
#     - k: degradation coefficient
#     - n: degradation exponent
#     - t: time
    
#     :param b: pyomo Block() to be set
#     :param t: pyomo Set() referring to time
#     :param data: data dict containing SOH parameters
#     :param init_data: init_data dict
    
#     Additional data parameters for SOH:
#          - 'k': degradation coefficient (default: 0.0001)
#          - 'n': degradation exponent (default: 1.0)
#          - 'SOH_min': minimum allowable SOH (default: 0.8)
         
#     Standard data parameters:
#          - 'E0': Initial energy
#          - 'Emax': Maximum battery energy
#          - 'SOCmin': Minimum allowed SOC
#          - 'SOCmax': Maximum allowed SOC
#          - 'Pmax': Maximum delivered/absorbed power
#          - 'Einst': Energy storage already installed
#          - 'Pinst': Power already installed
#          - 'rend_ch': Charging efficiency
#          - 'rend_disc': Discharging efficiency
         
#     init_data:
#          - 'E': Energy E(t) as a list
#          - 'P': Power P(t) as a list
         
#     Pyomo declarations:
#         - Parameters: k, n, SOH_min, E0, Emax, SOCmin, SOCmax, Pmax, Einst, Pinst, rend_ch, rend_disc
#         - Variables: SOH(t), E(t), P(t), Pch(t), Pdisc(t), SOC(t), Pdim, Edim
#         - Ports: port_P @ P (Extensive)
#         - Constraints: SOH equation and all standard battery constraints adjusted for SOH
#     """
    
#     # Parameters for SOH model
#    # b.k = pyo.Param(initialize=data.get('k', 0.0001))  # degradation coefficient
#    # b.n = pyo.Param(initialize=data.get('n', 1.0))     # degradation exponent
#    # b.SOH_min = pyo.Param(initialize=data.get('SOH_min', 0.8))  # minimum allowable SOH
    
#     # Parameters for SOH model
#     b.k = pyo.Param(initialize=data.get('k', 3.37868689199743e-4))  # degradation coefficient
#     b.n = pyo.Param(initialize=data.get('n', 0.5))     # degradation exponent
#     b.SOH_min = pyo.Param(initialize=data.get('SOH_min', 0.8))  # minimum allowable SOH
    
    
#     # Standard battery parameters
#     b.E0 = pyo.Param(initialize=data['E0'])
#     b.Emax = pyo.Param(initialize=data['Emax'])
#     b.SOCmax = pyo.Param(initialize=data['SOCmax'])
#     b.SOCmin = pyo.Param(initialize=data['SOCmin'])
#     b.Pmax = pyo.Param(initialize=data['Pmax'])
#     b.Einst = pyo.Param(initialize=data['Einst'])
#     b.Pinst = pyo.Param(initialize=data['Pinst'])
#     b.rend_ch = pyo.Param(initialize=data['rend_ch'])
#     b.rend_disc = pyo.Param(initialize=data['rend_disc'])
    
#     # Variables
#     b.SOH = pyo.Var(t, initialize={k: 1.0 for k in range(len(t))}, 
#                     bounds=(data.get('SOH_min', 0.8), 1.0), 
#                     within=pyo.NonNegativeReals)
    
#     # Initialize E and P with defaults if init_data is empty
#     init_E = init_data.get('E', {k: data['E0'] for k in range(len(t))})
#     init_P = init_data.get('P', {k: 0.0 for k in range(len(t))})
    
#     b.E = pyo.Var(t, initialize=init_E, within=pyo.NonNegativeReals)
#     b.P = pyo.Var(t, initialize=init_P, 
#                   bounds=(-data['Pmax'], data['Pmax']), within=pyo.Reals)
#     b.Pch = pyo.Var(t, initialize={k: 0.0 for k in range(len(t))}, 
#                     within=pyo.NonNegativeReals)
#     b.Pdisc = pyo.Var(t, initialize={k: 0.0 for k in range(len(t))}, 
#                       within=pyo.NonNegativeReals)
#     b.SOC = pyo.Var(t, initialize={k: data['E0']/data['Emax'] for k in range(len(t))}, 
#                     bounds=(data['SOCmin'], data['SOCmax']), 
#                     within=pyo.NonNegativeReals)
#     b.Pdim = pyo.Var(initialize=0, bounds=(0, data['Pmax']-data['Pinst']), 
#                      within=pyo.NonNegativeReals)
#     b.Edim = pyo.Var(initialize=0, bounds=(0, data['Emax']-data['Einst']), 
#                      within=pyo.NonNegativeReals)
    
#     # Ports
#     b.port_P = Port(initialize={'P': (b.P, Port.Extensive)})
    
#     # SOH Constraint: SOH(t) = 1 - k · t^n
#     def Constraint_SOH(_b, _t):
#         return _b.SOH[_t] == 1 - _b.k * (_t + 1) ** _b.n
#     b.c_SOH = pyo.Constraint(t, rule=Constraint_SOH)
    
#     # Standard battery constraints ( Power P = charging power − discharging power )
#     def Constraint_P(_b, _t):
#         return _b.P[_t] == _b.Pch[_t] - _b.Pdisc[_t]
#     b.c_P = pyo.Constraint(t, rule=Constraint_P)
#     # Ek hi timestep me charge aur discharge dono ek saath na ho.
#     def Constraint_P0(_b, _t):
#         return 0 == _b.Pch[_t] * _b.Pdisc[_t]
#     b.c_P0 = pyo.Constraint(t, rule=Constraint_P0)
    
#     # SOC constraint adjusted for SOH degradation
#     def Constraint_SOC(_b, _t):
#         return _b.SOC[_t] == _b.E[_t] / ((_b.Einst + _b.Edim) * _b.SOH[_t])
#     b.c_SOC = pyo.Constraint(t, rule=Constraint_SOC)
    
#     # Energy balance constraint
#     def Constraint_E(_b, _t):
#         if _t > 0:
#             return _b.E[_t] == _b.E[_t-1] + (_b.Pch[_t]*_b.rend_ch - _b.Pdisc[_t]*_b.rend_disc)
#         else:
#             return _b.E[_t] == _b.E0 + (_b.Pch[_t]*_b.rend_ch - _b.Pdisc[_t]*_b.rend_disc)
#     b.c_E = pyo.Constraint(t, rule=Constraint_E)
    
#     # Power constraints adjusted for SOH
#     def Constraint_ch(_b, _t):
#         return _b.Pch[_t] <= (_b.Pinst + _b.Pdim) * _b.SOH[_t]
#     b.Consume = pyo.Constraint(t, rule=Constraint_ch)
    
#     def Constraint_disc(_b, _t):
#         return _b.Pdisc[_t] <= (_b.Pinst + _b.Pdim) * _b.SOH[_t]
#     b.Prod = pyo.Constraint(t, rule=Constraint_disc)
    
#     # Energy limits adjusted for SOH
#     def ConstraintE_max(_b, _t):
#         return _b.E[_t] <= (_b.Einst + _b.Edim) * _b.SOCmax * _b.SOH[_t]
#     b.MaxEnergy = pyo.Constraint(t, rule=ConstraintE_max)
    
#     def ConstraintE_min(_b, _t):
#         return _b.E[_t] >= (_b.Einst + _b.Edim) * _b.SOCmin * _b.SOH[_t]
#     b.MinEnergy = pyo.Constraint(t, rule=ConstraintE_min)



import pyomo.environ as pyo
from pyomo.network import Port

def Battery_SOH(b, t, data, init_data):
    """
    Battery with State of Health (SOH) degradation model.

    Semi-empirical calendar ageing at ~25°C:
        SOH(t_years) = 1 - k * t_years^n

    Defaults (stationary LFP-friendly):
        n = 0.5  (square-root time law)
        k = 0.03  (~3% capacity loss per year, realistic degradation)
        SOH_min = 0.8 (80% EOL)

    IMPORTANT: This implementation converts the model time index to YEARS using dt_hours.
               So you can keep hourly timesteps in the model and still pass k "per-year".

    Parameters expected in `data`:
        E0, Emax, SOCmin, SOCmax, Pmax, Einst, Pinst, rend_ch, rend_disc
        Optional: k, n, SOH_min, dt_hours

    init_data:
        Optional initial values: dicts for E[t], P[t]

    Pyomo components set on block `b`:
        Params: k, n, SOH_min, dt_hours, E0, Emax, SOCmin, SOCmax, Pmax, Einst, Pinst, rend_ch, rend_disc
        Vars:   SOH[t], E[t], P[t], Pch[t], Pdisc[t], SOC[t], Pdim, Edim
        Port:   port_P @ P (Extensive)
        Cons:   SOH in years; P balance; no-simultaneous charge/discharge; SOC; E-balance; P limits; E limits
    """

    # --- SOH params (per-year) + dt_hours (hours/step) ---
    b.n = pyo.Param(initialize=data.get('n', 0.5))           # √t law exponent
    # default k ~0.03 per year (3%/year). Users can override via data['k']
    b.k = pyo.Param(initialize=data.get('k', 0.03))
    b.SOH_min = pyo.Param(initialize=data.get('SOH_min', 0.8))
    # Accept legacy key 'dt' as hours/step too
    b.dt_hours = pyo.Param(initialize=data.get('dt_hours', data.get('dt', 1.0)))  # hourly steps default

    # --- Standard battery parameters ---
    b.E0 = pyo.Param(initialize=data['E0'])
    b.Emax = pyo.Param(initialize=data['Emax'])
    b.SOCmax = pyo.Param(initialize=data['SOCmax'])
    b.SOCmin = pyo.Param(initialize=data['SOCmin'])
    b.Pmax = pyo.Param(initialize=data['Pmax'])
    b.Einst = pyo.Param(initialize=data['Einst'])
    b.Pinst = pyo.Param(initialize=data['Pinst'])
    b.rend_ch = pyo.Param(initialize=data['rend_ch'])
    b.rend_disc = pyo.Param(initialize=data['rend_disc'])

    # --- Variables ---
    b.SOH = pyo.Var(
        t,
        initialize={k: 1.0 for k in range(len(t))},
        bounds=(pyo.value(b.SOH_min), 1.0),
        within=pyo.NonNegativeReals
    )

    # Initialize SOC at 50% realistically
    initial_SOC = 0.5
    init_E = init_data.get('E', {k: pyo.value(b.Emax) * initial_SOC for k in range(len(t))})

    # Initialize power with small alternating charge/discharge to simulate usage
    init_P = init_data.get('P', {k: 0.1 * ((-1) ** k) for k in range(len(t))})

    b.E = pyo.Var(t, initialize=init_E, within=pyo.NonNegativeReals)
    b.P = pyo.Var(t, initialize=init_P, bounds=(-pyo.value(b.Pmax), pyo.value(b.Pmax)), within=pyo.Reals)
    b.Pch = pyo.Var(t, initialize={k: max(0, init_P[k]) for k in range(len(t))}, within=pyo.NonNegativeReals)
    b.Pdisc = pyo.Var(t, initialize={k: abs(min(0, init_P[k])) for k in range(len(t))}, within=pyo.NonNegativeReals)
    b.SOC = pyo.Var(
        t,
        initialize={k: initial_SOC for k in range(len(t))},
        bounds=(pyo.value(b.SOCmin), pyo.value(b.SOCmax)),
        within=pyo.NonNegativeReals
    )
    b.Pdim = pyo.Var(initialize=0, bounds=(0, pyo.value(b.Pmax) - pyo.value(b.Pinst)), within=pyo.NonNegativeReals)
    b.Edim = pyo.Var(initialize=0, bounds=(0, pyo.value(b.Emax) - pyo.value(b.Einst)), within=pyo.NonNegativeReals)

    # Port
    b.port_P = Port(initialize={'P': (b.P, Port.Extensive)})

    # --- Constraints ---

    def Constraint_SOH(_b, _t):
        t_years = (_t + 1) * _b.dt_hours / 8760.0
        return _b.SOH[_t] == 1 - _b.k * (t_years ** _b.n)
    b.c_SOH = pyo.Constraint(t, rule=Constraint_SOH)

    # Power balance: P = Pch - Pdisc
    def Constraint_P(_b, _t):
        return _b.P[_t] == _b.Pch[_t] - _b.Pdisc[_t]
    b.c_P = pyo.Constraint(t, rule=Constraint_P)

    # No simultaneous charge & discharge (relaxed complementarity)
    def Constraint_P0(_b, _t):
        return 0 == _b.Pch[_t] * _b.Pdisc[_t]
    b.c_P0 = pyo.Constraint(t, rule=Constraint_P0)

    # SOC = E / ((Einst + Edim) * SOH)
    def Constraint_SOC(_b, _t):
        return _b.SOC[_t] == _b.E[_t] / ((_b.Einst + _b.Edim) * _b.SOH[_t])
    b.c_SOC = pyo.Constraint(t, rule=Constraint_SOC)

    # Energy balance (1 step = 1 hour by default; efficiencies applied to Pch/Pdisc)
    def Constraint_E(_b, _t):
        if _t > 0:
            return _b.E[_t] == _b.E[_t-1] + (_b.Pch[_t]*_b.rend_ch - _b.Pdisc[_t]*_b.rend_disc) * _b.dt_hours
        else:
            return _b.E[_t] == _b.E0 + (_b.Pch[_t]*_b.rend_ch - _b.Pdisc[_t]*_b.rend_disc) * _b.dt_hours
    b.c_E = pyo.Constraint(t, rule=Constraint_E)

    # Power limits scaled by SOH
    def Constraint_ch(_b, _t):
        return _b.Pch[_t] <= (_b.Pinst + _b.Pdim) * _b.SOH[_t]
    b.Consume = pyo.Constraint(t, rule=Constraint_ch)

    def Constraint_disc(_b, _t):
        return _b.Pdisc[_t] <= (_b.Pinst + _b.Pdim) * _b.SOH[_t]
    b.Prod = pyo.Constraint(t, rule=Constraint_disc)

    # Energy limits with SOH
    def ConstraintE_max(_b, _t):
        return _b.E[_t] <= (_b.Einst + _b.Edim) * _b.SOCmax * _b.SOH[_t]
    b.MaxEnergy = pyo.Constraint(t, rule=ConstraintE_max)

    def ConstraintE_min(_b, _t):
        return _b.E[_t] >= (_b.Einst + _b.Edim) * _b.SOCmin * _b.SOH[_t]
    b.MinEnergy = pyo.Constraint(t, rule=ConstraintE_min)


#%% Battery_SOH_Najera (combined calendar + cycling ageing)

def Battery_SOH_Najera(b, t, data, init_data):
    """
    Battery with combined calendar + cycling ageing, based on:
        Najera et al., "Semi-empirical ageing model for LFP and NMC
        Li-ion battery chemistries", J. Energy Storage 72 (2023) 108016.

    Calendar ageing (depends on SoC, constant temperature T, elapsed time t in days):
        Ln(Q_cal) = f * exp(g*SoC) * exp(h/T) * t^z

    Cycling ageing (depends on current via C-rate and Ah throughput, constant T):
        Ln(Q_cyc) = (a*T^2 + b*T + c) * exp((d*T + e)*Crate) * AH

    Combined capacity fade and SOH:
        Q_loss = Q_cal + Q_cyc   (in %)
        SOH    = 1 - Q_loss/100

    Since this model has no explicit current/voltage variables, C-rate and
    Ah-throughput are derived from Power under a constant-nominal-voltage
    assumption:
        Crate[t] = (Pch[t] + Pdisc[t]) / Einst
        AH[t]    = AH[t-1] + (Pch[t] + Pdisc[t]) * dt_hours   (cumulative)

    Temperature is held constant (data['T_kelvin'], default 298.15 K = 25 C)
    per instruction, to avoid a full thermal model.

    Default parameters correspond to the Sony US26650FT LFP cell (Table 3
    of the paper), the only LFP cell in the paper with both calendar and
    cycling ageing validated together. Override via `data` for NMC (e.g.
    NMC 20 Ah cell, Table 4) or another cell.

    Parameters expected in `data`:
        E0, Emax, SOCmin, SOCmax, Pmax, Einst, Pinst, rend_ch, rend_disc
        Optional ageing params: a, b, c, d, e, f, g, h, z, T_kelvin, dt_hours

    IMPORTANT (numerical note): a, b, c are tiny numbers that nearly cancel
    at typical operating temperatures. Using only a handful of significant
    figures (as published in the paper's table) can change the cycling
    ageing result substantially. Treat cycling ageing outputs as indicative
    unless higher-precision parameters are obtained from the source thesis.
    """

    # --- Calendar ageing parameters (default: Sony US26650FT, LFP) ---
    b.f_cal = pyo.Param(initialize=data.get('f', 6.4726e8))
    b.g_cal = pyo.Param(initialize=data.get('g', 1.4219))
    b.h_cal = pyo.Param(initialize=data.get('h', -8.2191e3))
    b.z_cal = pyo.Param(initialize=data.get('z', 0.5))

    # --- Cycling ageing parameters (default: Sony US26650FT, LFP) ---
    b.a_cyc = pyo.Param(initialize=data.get('a', 2.9961e-8))
    b.b_cyc = pyo.Param(initialize=data.get('b', -1.7339e-5))
    b.c_cyc = pyo.Param(initialize=data.get('c', 0.0025))
    b.d_cyc = pyo.Param(initialize=data.get('d', -0.0124))
    b.e_cyc = pyo.Param(initialize=data.get('e', 3.8738))

    # --- Constant temperature (kept fixed, per instruction) ---
    b.T_kelvin = pyo.Param(initialize=data.get('T_kelvin', 298.15))
    b.dt_hours = pyo.Param(initialize=data.get('dt_hours', data.get('dt', 1.0)))
    b.SOH_min = pyo.Param(initialize=data.get('SOH_min', 0.8))

    # --- Standard battery parameters ---
    b.E0 = pyo.Param(initialize=data['E0'])
    b.Emax = pyo.Param(initialize=data['Emax'])
    b.SOCmax = pyo.Param(initialize=data['SOCmax'])
    b.SOCmin = pyo.Param(initialize=data['SOCmin'])
    b.Pmax = pyo.Param(initialize=data['Pmax'])
    b.Einst = pyo.Param(initialize=data['Einst'])
    b.Pinst = pyo.Param(initialize=data['Pinst'])
    b.rend_ch = pyo.Param(initialize=data['rend_ch'])
    b.rend_disc = pyo.Param(initialize=data['rend_disc'])

    # --- Variables ---
    initial_SOC = 0.5
    init_E = init_data.get('E', {k: pyo.value(b.Emax) * initial_SOC for k in range(len(t))})
    init_P = init_data.get('P', {k: 0.1 * ((-1) ** k) for k in range(len(t))})

    b.E = pyo.Var(t, initialize=init_E, within=pyo.NonNegativeReals)
    b.P = pyo.Var(t, initialize=init_P, bounds=(-pyo.value(b.Pmax), pyo.value(b.Pmax)), within=pyo.Reals)
    b.Pch = pyo.Var(t, initialize={k: max(0, init_P[k]) for k in range(len(t))}, within=pyo.NonNegativeReals)
    b.Pdisc = pyo.Var(t, initialize={k: abs(min(0, init_P[k])) for k in range(len(t))}, within=pyo.NonNegativeReals)
    b.SOC = pyo.Var(
        t,
        initialize={k: initial_SOC for k in range(len(t))},
        bounds=(pyo.value(b.SOCmin), pyo.value(b.SOCmax)),
        within=pyo.NonNegativeReals
    )
    b.AH = pyo.Var(t, initialize={k: 0.0 for k in range(len(t))}, within=pyo.NonNegativeReals)
    b.Q_cal = pyo.Var(t, initialize={k: 1.0 for k in range(len(t))}, within=pyo.NonNegativeReals)
    b.Q_cyc = pyo.Var(t, initialize={k: 1.0 for k in range(len(t))}, within=pyo.NonNegativeReals)
    b.SOH = pyo.Var(
        t,
        initialize={k: 1.0 for k in range(len(t))},
        bounds=(pyo.value(b.SOH_min), 1.0),
        within=pyo.NonNegativeReals
    )
    b.Pdim = pyo.Var(initialize=0, bounds=(0, pyo.value(b.Pmax) - pyo.value(b.Pinst)), within=pyo.NonNegativeReals)
    b.Edim = pyo.Var(initialize=0, bounds=(0, pyo.value(b.Emax) - pyo.value(b.Einst)), within=pyo.NonNegativeReals)

    # Port
    b.port_P = Port(initialize={'P': (b.P, Port.Extensive)})

    # --- Constraints ---

    # Cumulative Ah throughput (current-based, under constant nominal voltage assumption)
    def Constraint_AH(_b, _t):
        crate_energy = (_b.Pch[_t] + _b.Pdisc[_t]) * _b.dt_hours
        if _t > 0:
            return _b.AH[_t] == _b.AH[_t-1] + crate_energy
        else:
            return _b.AH[_t] == crate_energy
    b.c_AH = pyo.Constraint(t, rule=Constraint_AH)

    # Calendar ageing: Ln(Q_cal) = f * exp(g*SoC) * exp(h/T) * t_days^z
    def Constraint_Qcal(_b, _t):
        t_days = (_t + 1) * _b.dt_hours / 24.0
        return _b.Q_cal[_t] == pyo.exp(
            _b.f_cal * pyo.exp(_b.g_cal * _b.SOC[_t]) * pyo.exp(_b.h_cal / _b.T_kelvin) * (t_days ** _b.z_cal)
        )
    b.c_Qcal = pyo.Constraint(t, rule=Constraint_Qcal)

    # Cycling ageing: Ln(Q_cyc) = (a*T^2 + b*T + c) * exp((d*T+e)*Crate) * AH
    #
    # NUMERICAL SAFEGUARD: (a*T^2 + b*T + c) is a near-cancellation of three
    # tiny terms. With only the ~5 significant figures published in the
    # paper's table, this polynomial flips sign across 0-30C for this cell,
    # even though the paper's own figures show real, increasing capacity
    # fade with cycling in that exact range. Physically, cycling damage must
    # be monotonically non-decreasing with Ah throughput, so the polynomial
    # must act as a positive rate constant; its magnitude (fitted to real
    # test data) is trustworthy, its sign here is a rounding artifact. abs()
    # is used to keep the fitted magnitude while enforcing the correct sign.
    def Constraint_Qcyc(_b, _t):
        crate = (_b.Pch[_t] + _b.Pdisc[_t]) / _b.Einst
        poly_T = abs(_b.a_cyc * _b.T_kelvin**2 + _b.b_cyc * _b.T_kelvin + _b.c_cyc)
        return _b.Q_cyc[_t] == pyo.exp(poly_T * pyo.exp((_b.d_cyc * _b.T_kelvin + _b.e_cyc) * crate) * _b.AH[_t])
    b.c_Qcyc = pyo.Constraint(t, rule=Constraint_Qcyc)

    # Combined capacity fade -> SOH
    def Constraint_SOH(_b, _t):
        return _b.SOH[_t] == 1 - (_b.Q_cal[_t] + _b.Q_cyc[_t]) / 100.0
    b.c_SOH = pyo.Constraint(t, rule=Constraint_SOH)

    # Power balance: P = Pch - Pdisc
    def Constraint_P(_b, _t):
        return _b.P[_t] == _b.Pch[_t] - _b.Pdisc[_t]
    b.c_P = pyo.Constraint(t, rule=Constraint_P)

    # No simultaneous charge & discharge (relaxed complementarity)
    def Constraint_P0(_b, _t):
        return 0 == _b.Pch[_t] * _b.Pdisc[_t]
    b.c_P0 = pyo.Constraint(t, rule=Constraint_P0)

    # SOC = E / ((Einst + Edim) * SOH)
    def Constraint_SOC(_b, _t):
        return _b.SOC[_t] == _b.E[_t] / ((_b.Einst + _b.Edim) * _b.SOH[_t])
    b.c_SOC = pyo.Constraint(t, rule=Constraint_SOC)

    # Energy balance (1 step = 1 hour by default; efficiencies applied to Pch/Pdisc)
    def Constraint_E(_b, _t):
        if _t > 0:
            return _b.E[_t] == _b.E[_t-1] + (_b.Pch[_t]*_b.rend_ch - _b.Pdisc[_t]*_b.rend_disc) * _b.dt_hours
        else:
            return _b.E[_t] == _b.E0 + (_b.Pch[_t]*_b.rend_ch - _b.Pdisc[_t]*_b.rend_disc) * _b.dt_hours
    b.c_E = pyo.Constraint(t, rule=Constraint_E)

    # Power limits scaled by SOH
    def Constraint_ch(_b, _t):
        return _b.Pch[_t] <= (_b.Pinst + _b.Pdim) * _b.SOH[_t]
    b.Consume = pyo.Constraint(t, rule=Constraint_ch)

    def Constraint_disc(_b, _t):
        return _b.Pdisc[_t] <= (_b.Pinst + _b.Pdim) * _b.SOH[_t]
    b.Prod = pyo.Constraint(t, rule=Constraint_disc)

    # Energy limits with SOH
    def ConstraintE_max(_b, _t):
        return _b.E[_t] <= (_b.Einst + _b.Edim) * _b.SOCmax * _b.SOH[_t]
    b.MaxEnergy = pyo.Constraint(t, rule=ConstraintE_max)

    def ConstraintE_min(_b, _t):
        return _b.E[_t] >= (_b.Einst + _b.Edim) * _b.SOCmin * _b.SOH[_t]
    b.MinEnergy = pyo.Constraint(t, rule=ConstraintE_min)
