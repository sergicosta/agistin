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
            - E (t) bounded :math:`E(t) \in [0, E_{max}\, SOC_{max}]`
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
    # b.E0 = pyo.Param(initialize=data['E0'])
    b.SOCmax = pyo.Param(initialize=data['SOCmax'])
    b.SOCmin = pyo.Param(initialize=data['SOCmin'])
    b.Pmax = pyo.Param(initialize=data['Pmax'])
    b.Emax = pyo.Param(initialize=data['Emax'])
    b.Pinst = pyo.Param(initialize = data['Pinst'])
    b.Einst = pyo.Param(initialize = data['Einst'])
    b.eff_ch = pyo.Param(initialize = data['eff_ch'])
    b.eff_dc = pyo.Param(initialize = data['eff_dc'])
    
    # Variables
    b.E  = pyo.Var(t, initialize= init_data['E'], bounds=(0, data['Emax']*data['SOCmax']), within=pyo.NonNegativeReals)
    b.P = pyo.Var(t, initialize= init_data['P'], bounds = (-data['Pmax'],data['Pmax']), within=pyo.Reals)
    b.Pch = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, bounds = (0,data['Pmax']), within=pyo.NonNegativeReals)
    b.Pdc = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, bounds = (0,data['Pmax']), within=pyo.NonNegativeReals)
    b.SOC = pyo.Var(t, initialize={k:(init_data['E'][h]/data['Einst'] if data['Einst']>0 else data['SOCmin']) for h,k in enumerate(range(len(t)))}, bounds=(data['SOCmin'], data['SOCmax']), within=pyo.NonNegativeReals)
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
            return b.E[_t] == (_b.SOCmax+_b.SOCmin)/2*(_b.Einst+_b.Edim) + _b.dt*(_b.Pch[_t]*_b.eff_ch - _b.Pdc[_t]*_b.eff_dc)
            # return b.E[_t] == _b.E0 + _b.dt*(_b.Pch[_t]*_b.eff_ch - _b.Pdc[_t]*_b.eff_dc)
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
  
    
def NewBattery(b, t, data, init_data):
    """
    Battery set for sizing.
    
    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: data ``dict``
    :param init_data: init_data ``dict``
        
    data:
         - 'dt': time step :math:`\Delta t`
         - 'SOCmin': Minimum allowed SOC :math:`SOC_{min}` in p.u.
         - 'SOCmax': Maximum allowed SOC :math:`SOC_{max}` in p.u.
         - 'Pmax': Maximum delivered/absorbed power :math:`P_{max}`
         - 'Emax': Maximum battery energy :math:`E_{max}`
         - 'eff_ch': Effiency of charge :math:`\eta_{ch}`
         - 'eff_dc': Effiency of discharge :math:`\eta_{dc}`
             
    init_data:
         - 'E': Energy :math:`E(t)` as a ``list``
         - 'P': Power :math:`P(t)` as a ``list``
         
    Pyomo declaration
        - Parameters: 
            - dt
            - SOCmin
            - SOCmax
            - Pmax
            - Emax
            - eff_ch
            - eff_dc
            - T
        - Variables: 
            - E (t) bounded :math:`E(t) \in [0, E_{max}\, SOC_{max}]`
            - P (t) bounded :math:`P(t) \in [-P_{max}, P_{max}]`
            - Pch (t) bounded :math:`P_{ch}(t) \in [0, P_{max}]`
            - Pdc (t) bounded :math:`P_{dc}(t) \in [0, P_{max}]`
            - SOC (t) bounded :math:`SOC(t) \in [SOC_{min}, SOC_{max}]`
            - Edim bounded :math:`E_{dim} \in [0, E_{max}]`
            - Pdim bounded :math:`P_{dim} \in [0, P_{max}]`
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
                - :math:`E(t) = E(T) + \Delta t \, P(t) \quad` otherwise
     """       
    
    # Parameters
    b.dt = pyo.Param(initialize=data['dt'])
    b.SOCmax = pyo.Param(initialize=data['SOCmax'])
    b.SOCmin = pyo.Param(initialize=data['SOCmin'])
    b.Pmax = pyo.Param(initialize=data['Pmax'])
    b.Emax = pyo.Param(initialize=data['Emax'])
    b.eff_ch = pyo.Param(initialize = data['eff_ch'])
    b.eff_dc = pyo.Param(initialize = data['eff_dc'])
    b.T = pyo.Param(initialize = len(t))
    
    # Variables
    b.E  = pyo.Var(t, initialize= init_data['E'], bounds=(0, data['Emax']*data['SOCmax']), within=pyo.NonNegativeReals)
    b.P = pyo.Var(t, initialize= init_data['P'], bounds = (-data['Pmax'],data['Pmax']), within=pyo.Reals)
    b.Pch = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, bounds = (0,data['Pmax']), within=pyo.NonNegativeReals)
    b.Pdc = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, bounds = (0,data['Pmax']), within=pyo.NonNegativeReals)
    b.Pdim = pyo.Var(initialize = 0, bounds = (0,data['Pmax']),within = pyo.NonNegativeReals)
    b.Edim = pyo.Var(initialize = 1, bounds = (0,data['Emax']),within = pyo.NonNegativeReals)

    # Ports
    b.port_P = Port(initialize={'P': (b.P, Port.Extensive)})

    # Constraints
    def Constraint_P(_b, _t):
        return _b.P[_t] == _b.Pch[_t]*_b.eff_ch - _b.Pdc[_t]*_b.eff_dc
    b.c_P = pyo.Constraint(t, rule = Constraint_P)
    
    def Constraint_P0(_b, _t):
        return 0 == _b.Pch[_t] * _b.Pdc[_t]
    b.c_P0 = pyo.Constraint(t, rule = Constraint_P0)
    
    def Constraint_E(_b, _t):
        if _t>0:
            return _b.E[_t] == _b.E[_t-1] + _b.dt*(_b.Pch[_t]*_b.eff_ch - _b.Pdc[_t]*_b.eff_dc)
        else:
            return b.E[_t] == _b.E[_b.T-1] + _b.dt*(_b.Pch[_t]*_b.eff_ch - _b.Pdc[_t]*_b.eff_dc)
    b.c_E = pyo.Constraint(t, rule = Constraint_E)
    
    def Constraint_ch(_b,_t):
        return _b.Pch[_t] <= _b.Pdim
    b.c_ch = pyo.Constraint(t, rule = Constraint_ch)
    
    def Constraint_disc(_b,_t):
        return  _b.Pdc[_t] <= _b.Pdim
    b.c_dc = pyo.Constraint(t, rule = Constraint_disc)
    
    def ConstraintE_max(_b,_t):
        return _b.E[_t] <= _b.Edim*_b.SOCmax
    b.MaxEnergy = pyo.Constraint(t, rule = ConstraintE_max)
    
    def ConstraintE_min(_b,_t):
        return _b.E[_t] >= _b.Edim*_b.SOCmin
    b.MinEnergy = pyo.Constraint(t, rule = ConstraintE_min)