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
    
    # """
    # Simple Battery.
    
    # Modifies its energy state :math:`E(t)` from an initial state :math:`E_0` according to 
    # the charge or discharge power :math:`P(t)>0` is :math:`P_{ch}(t)` and :math:`P(t)<0` is :math:`P_{disc}(t)`.
    
    # The state of charge `SOC(t)` is computed and taken into account as well.
    
    # :param b: pyomo ``Block()`` to be set
    # :param t: pyomo ``Set()`` referring to time
    # :param data: data ``dict``
    # :param init_data: init_data ``dict``
    
    # data
    #      - 'dt': time delta :math:`\Delta t`
    #      - 'E0': Initial energy :math:`E_0`
    #      - 'Emax': Maximum battery energy :math:`E_{max}`
    #      - 'SOCmin': Minimum allowed SOC :math:`SOC_{min}` in p.u.
    #      - 'SOCmax': Maximum allowed SOC :math:`SOC_{max}` in p.u.
    #      - 'Pmax': Maximum delivered/absorbed power :math:`P_{max}`
         
    # init_data
    #      - 'E': Energy :math:`E(t)` as a ``list``
    #      - 'P': Power :math:`P(t)` as a ``list``
    
    # Pyomo declarations    
    #     - Parameters: 
    #         - dt
    #         - E0
    #         - Emax
    #         - SOCmin
    #         - SOCmax
    #         - Pmax
    #         - eff_ch
    #         - eff_disc
    #     - Variables: 
    #         - E (t) bounded :math:`E(t) \in [E_{max}\cdot SOC_{min}, E_{max}\cdot SOC_{max}]`
    #         - P (t) bounded :math:`P(t) \in [-P_{max}, P_{max}]`
    #         - Pch (t) bounded :math:`P_{ch}(t) \in [0, P_{max}]`
    #         - Pdisc (t) bounded :math:`P_{disc}(t) \in [0, P_{max}]`
    #         - SOC (t) bounded :math:`SOC(t) \in [SOC_{min}, SOC_{max}]`
    #     - Ports: 
    #         - port_P @ P (Extensive)
    #     - Constraints:
    #         - c_P: :math:`P(t) = P_{ch}(t) - P_{disc}(t)`
    #         - c_P0: :math:`0 = P_{ch}(t) \cdot P_{disc}(t)`
    #         - c_SOC: :math:`SOC(t) = E(t) / E_{max}`
    #         - c_E: 
    #             - :math:`E(t) = E(t-1) + \Delta t \cdot P(t) \quad` if  :math:`t>0`
    #             - :math:`E(t) = E_0 + \Delta t \cdot P(t) \quad` otherwise
    # """
    
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
    b.Pdim = pyo.Var(initialize = 0, bounds =(0,data['Pmax']),within = pyo.NonNegativeReals)
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
    # """
    # Simple Battery for testing and example purposes.
    # It is used in Example21.
    
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
    #         - SOCmin
    #         - SOCmax
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

import pyomo.environ as pyo

def Battery_MV(b, t, data):
    
    # Parameters
    b.E0 = pyo.Param(initialize=data['E0'])       # Energy storage at the battery at the beginning of the simulation (_t=0)
    b.Pmax = pyo.Param(initialize=data['Pmax'])   # Battery maximum charging and discharging power  
    b.Emax = pyo.Param(initialize=data['Emax'])   # Battery maximum energy capacity
    b.Einst = pyo.Param(initialize=data['Einst']) # Battery installed rated energy
    b.Pinst = pyo.Param(initialize=data['Pinst']) # Battery installed rated power
    b.slope_fcr = pyo.Param(initialize=data['slope_fcr']) # static for the FCR (100% power by a 200mH deviation)
    b.F = pyo.Param(t, initialize=data['F'])      # Grid frequency (time series)
    
    b.SOCmin = pyo.Param(initialize=0.25*(data['Pinst'])/(data['Einst'])) 
    b.SOCmax = pyo.Param(initialize=((data['Einst']) - 0.25*(data['Pinst']))/(data['Einst'])) # bounds = (0.0,data['SOCmax']),
    
    # Variables
    b.Pdemanded = pyo.Var(t, initialize=0.0, within=pyo.Reals) 
    b.Pcharged = pyo.Var(t, initialize=0.0, within=pyo.NonNegativeReals)
    b.Pdischarged = pyo.Var(t, initialize=0.0, within=pyo.NonNegativeReals)
    b.EstrgOut = pyo.Var(t, initialize=0.0, bounds=(0.0, data['Einst']), within=pyo.NonNegativeReals)
    b.SOC = pyo.Var(t, initialize=0.0, bounds=(data['SOCmin'], data['SOCmax']), within=pyo.NonNegativeReals)
    b.Pout = pyo.Var(t, initialize=0.0, within=pyo.Reals)
    
    # Constraints
    def Constraint_P(b, t):  # This constraint calculates the amount of power to be delivered
        return b.Pdemanded[t] == (50 - (b.F[t])) * (b.Pinst) * b.slope_fcr
    b.Constraint_P = pyo.Constraint(t, rule=Constraint_P) 
    
    def Constraint_Pbalance(b, t):  # This constraint calculates the amount of power to be delivered
        return b.Pdemanded[t] - b.Pcharged[t] + b.Pdischarged[t] == 0.0
    b.Constraint_Pbalance = pyo.Constraint(t, rule=Constraint_Pbalance) 
    
    def Constraint_Pbalance_1(b, t):  # This constraint calculates the amount of power to be delivered
        return b.Pcharged[t] * b.Pdischarged[t] == 0.0
    b.Constraint_Pbalance_1 = pyo.Constraint(t, rule=Constraint_Pbalance_1)

    def Constraint_EstrgOut(b, t): 
        if t == 0:  # This constraint calculates the power at the end of the time step
            return b.EstrgOut[t] == b.E0 + (50 - (b.F[t])) * b.Pinst * b.slope_fcr
        else:
            return b.EstrgOut[t] == b.EstrgOut[t - 1] + (50 - (b.F[t])) * b.Pinst * b.slope_fcr
    b.Constraint_EstrgOut = pyo.Constraint(t, rule=Constraint_EstrgOut)
     
    def Constraint_SOC(b, t):  # This constraint calculates the current SOC of the battery
        return b.SOC[t] == b.EstrgOut[t] / (b.Einst)
    b.c_SOC = pyo.Constraint(t, rule=Constraint_SOC)

    def Constraint_Pout(b, t):  # Based on the SCO and the power demanded, this constraint computes the amount of energy/power that can be delivered
        if t == 0:
            return b.Pout[t] == (50 - (b.F[t])) * (b.Pinst) * b.slope_fcr
        else:
            return b.Pout[t] == (b.SOC[t] - b.SOC[t - 1]) * (b.Einst)
    b.c_Pout = pyo.Constraint(t, rule=Constraint_Pout)  

    # Constraint rule for estrout
    def Constraint_EstrgOut_rule(b, t):
        if t == 0:
            return b.EstrgOut[t] == data['E0'] + (50 - data['F'][t]) * data['Pinst'] * data['slope_fcr']
        else:
            return b.EstrgOut[t] == b.EstrgOut[t - 1] + (50 - data['F'][t]) * data['Pinst'] * data['slope_fcr']
    b.Constraint_EstrgOut_rule = pyo.Constraint(t, rule=Constraint_EstrgOut_rule)

    # Additional Parameters
    tot_time = len(data['F'])
    Price = [0.5 for _ in range(tot_time)]
    b.EtaCharge = pyo.Param(initialize=0.9)  # Charging efficiency
    b.EtaDischarge = pyo.Param(initialize=0.85)  # Discharging efficiency
    b.PMinCharge = pyo.Param(initialize=0)  # Minimum charging power
    b.PMaxCharge = pyo.Param(initialize=100)  # Maximum charging power
    b.PMinDischarge = pyo.Param(initialize=0)  # Minimum discharging power
    b.PMaxDischarge = pyo.Param(initialize=100)  # Maximum discharging power
    
    # Sets
    b.T = pyo.Set(initialize=[t for t in range(tot_time)])
    
    # Variables
    b.QEl = pyo.Var(b.T, within=pyo.NonNegativeReals, initialize=0)
    b.QCharge = pyo.Var(b.T, within=pyo.NonNegativeReals)
    b.QDischarge = pyo.Var(b.T, within=pyo.NonNegativeReals)
    b.Charge = pyo.Var(b.T, within=pyo.Binary)
    b.Discharge = pyo.Var(b.T, within=pyo.Binary)
    b.SOC_2 = pyo.Var(b.T, bounds=(0, 100))
    b.estrout_2 = pyo.Var(b.T, within=pyo.NonNegativeReals, bounds=(0, 24000))

    # Constraints
    b.C1 = pyo.ConstraintList()
    b.C2 = pyo.ConstraintList()
    b.C3 = pyo.ConstraintList()
    b.C4 = pyo.ConstraintList()
    b.C5 = pyo.ConstraintList()
    b.C6 = pyo.ConstraintList()
    b.C7 = pyo.ConstraintList()
    b.C8 = pyo.ConstraintList()
    b.C9 = pyo.ConstraintList()
    b.C10 = pyo.Constraint(b.T, rule=Constraint_EstrgOut_rule)  # New constraint for estrout

    for t in b.T:
        b.C1.add(b.QEl[t] * b.EtaCharge == b.QCharge[t])
        b.C2.add(b.QDischarge[t] * b.EtaDischarge == data['F'][t])
        b.C3.add(b.Charge[t] + b.Discharge[t] <= 1)
        b.C4.add(b.QCharge[t] >= b.PMinCharge * b.Charge[t])
        b.C5.add(b.QCharge[t] <= b.PMaxCharge * b.Charge[t])
        b.C6.add(b.QDischarge[t] >= b.PMinDischarge * b.Discharge[t])
        b.C7.add(b.QDischarge[t] <= b.PMaxDischarge * b.Discharge[t])

        # State of Charge (SOC) constraints
        if t == 0:
            b.C8.add(b.SOC_2[t] == data['E0'] + b.QCharge[t] - b.QDischarge[t])
        else:
            b.C8.add(b.SOC_2[t] == b.SOC_2[t - 1] + b.QCharge[t] - b.QDischarge[t])

        # Final SOC constraint
        if t == tot_time - 1:
            b.C9.add(b.SOC_2[t] == b.SOC_2[0] + sum(b.QCharge[i] - b.QDischarge[i] for i in b.T))

    # Objective
    b.objective = pyo.Objective(expr=sum(b.QEl[t] * Price[t] for t in b.T), sense=pyo.minimize)

