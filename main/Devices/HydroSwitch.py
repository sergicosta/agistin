#AGISTIN project 
#.\Devices\HydroSwitch.py
"""
HydroSwitch pyomo block contains characteristics of a switch for pipe commutation.
"""

import pyomo.environ as pyo
from pyomo.network import Arc, Port


# data: None
# init_data: None

def HydroSwitch(b, t, data=None, init_data=None):
   
    """
    Hydraulic switch.
    
    Commutes the output :math:`Q_{out}(t)` between 2 inputs of water flow :math:`Q_{in,0}(t)` and :math:`Q_{in,1}(t)` , analogous of an electrical switch
    
    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: ``None``
    :param init_data: ``None``
    
    Pyomo declarations    
        - Parameters:
            - `None`
        - Variables: 
            - Qin0 (t) :math:`\in \mathbb{R}`
            - Qin1 (t) :math:`\in \mathbb{R}`
            - Qout (t) :math:`\in \mathbb{R}`
            - Hin0 (t) :math:`\in \mathbb{R}`
            - Hin1 (t) :math:`\in \mathbb{R}`
            - Hout (t) :math:`\in \mathbb{R}`
            - alpha (t) :math:`\in \{0,1\}`

        - Ports: 
            - port_Qin0 @ Qin0 as ``Extensive``
            - port_Qin1 @ Qin1 as ``Extensive``
            - port_Qout @ Qout as ``Extensive``
            - port_Hout @ Hout as ``Equality``
            - port_Hin0 @ Hin0 as ``Equality``
            - port_Hin1 @ Hin1 as ``Equality``
            
        - Constraints: 
            - c_State: :math:`Q_{in,0}(t) \, Q_{in,1}(t) = 0`
            - c_Qout: :math:`- Q_{out}(t) + Q_{in,0}(t) + Q_{in,1}(t) = 0`
            - c_Hout: :math:`- H_{out}(t) + H_{in1}(t) \, b_{H}(t) + H_{in0} \, (1-b_{H}(t)) = 0`
            - c_Qin0: :math:`Q_{in0}(t) \, H{in1}(t) \, b_{H}(t) = 0`
            - c_Qin1: :math:`Q_{in1}(t) \, H{in0}(t) \, (1-b_{H}(t)) = 0`

    """
    
    # Parameters
    
    # Variables
    b.Qin0 = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, domain=pyo.Reals)
    b.Qin1 = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, domain=pyo.Reals)
    b.Qout = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, domain=pyo.Reals)
    
    b.Hout = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, domain=pyo.Reals)
    b.Hin0 = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, domain=pyo.Reals)
    b.Hin1 = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, domain=pyo.Reals)

    b.alpha = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, domain=pyo.Binary)
    
    # Ports
    b.port_Qin0 = Port(initialize={'Q': (b.Qin0, Port.Extensive)})
    b.port_Qin1 = Port(initialize={'Q': (b.Qin1, Port.Extensive)})
    b.port_Qout = Port(initialize={'Q': (b.Qout, Port.Extensive)})
    
    b.port_Hout = Port(initialize={'H': (b.Hout, Port.Equality)})
    b.port_Hin0 = Port(initialize={'H': (b.Hin0, Port.Equality)})
    b.port_Hin1 = Port(initialize={'H': (b.Hin1, Port.Equality)}) 
    
    # Constraints
    def Constraint_State(_b, _t):
        return _b.Qin0[_t] * _b.Qin1[_t] == 0
    b.c_State = pyo.Constraint(t, rule=Constraint_State)
    
    def Constraint_Pout(_b, _t):
        return 0 == _b.Qin0[_t] + _b.Qin1[_t] - _b.Qout[_t]
    b.c_Qout = pyo.Constraint(t, rule=Constraint_Pout)
    
    
    def Constraint_Hout(_b, _t):
        return 0 == _b.Hin0[_t] * (1-_b.alpha[_t]) + _b.Hin1[_t] * _b.alpha[_t] - _b.Hout[_t]
    b.c_Hout = pyo.Constraint(t, rule=Constraint_Hout)
    
    def Constraint_Qin0(_b, _t):
        return _b.Qin0[_t] * _b.Hin1[_t] * _b.alpha[_t] == 0
    b.c_Qin0 = pyo.Constraint(t, rule=Constraint_Qin0)

    def Constraint_Qin1(_b, _t):
        return _b.Qin1[_t] * _b.Hin0[_t] *  (1-_b.alpha[_t]) == 0
    b.c_Qin1 = pyo.Constraint(t, rule=Constraint_Qin1)


    
    