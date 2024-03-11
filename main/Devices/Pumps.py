# AGISTIN project
# .\Devices\Pumps.py

"""
Pump pyomo block contains characteristics of a pump.
"""


import pyomo.environ as pyo

from pyomo.network import Arc, Port
from pyomo.core import *

# data: A, B, n_n, eff, Qmax, Qnom, Pmax
# init_data: Q(t), H(t), n(t), Pe(t)


def Pump(b, t, data, init_data):
    """
    Frequency controllable hydraulic pump.

    Applies the hydraulic head equation to a flow, with the ability to change its rotational speed, limited by a maximum power:

    .. math:
        H(t) = (n(t)/n_n)^2\cdot A - B \cdot Q(t)^2

    with :math:`A` and :math:`B` the characteristic coefficients that define the behaviour of the pump (i.e its curve).

    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: data ``dict``
    :param init_data: init_data ``dict``

    data
         - 'A': Constant characteristic coefficient of the pump :math:`A`
         - 'B': Quadratic characteristic coefficient of the pump :math:`B`
         - 'n_n': Nominal rotational speed :math:`n_n`
         - 'Qnom': Nominal flow :math:`Q_n`
         - 'eff': Efficiency at the nominal operating point in p.u. :math:`\eta`
         - 'Qmax': Maximum allowed flow :math:`Q_{max}`
         - 'Pmax': Maximum allowed power :math:`P_{max}`

    init_data
         - 'Q': Flow :math:`Q(t)` as a ``list``
         - 'H': Head :math:`H(t)` as a ``list``
         - 'n': Rotational speed :math:`n(t)` as a ``list``
         - 'Pe': Electrical power :math:`P_e(t)` as a ``list``

    Pyomo declarations    
        - Parameters:
            - A
            - B
            - Qnom
            - n_n
            - eff
        - Variables:
            - Qin (t) bounded :math:`Q_{in} \in [-Q_{max}, 0]`
            - Qout (t) bounded :math:`Q_{out} \in [0, Q_{max}]`
            - H (t) bounded :math:`H \ge 0`
            - n (t) bounded :math:`n \ge 0`
            - Ph (t) bounded :math:`P_h \in [0, P_{max}]`
            - Pe (t) bounded :math:`P_e \in [0, P_{max}]`
        - Ports:
            - port_Qin @ Qin with 'Q' as ``Extensive``
            - port_Qout @ Qout with 'Q' as ``Extensive``
            - port_P @ Pe with 'P' as ``Extensive``
            - port_H @ H with 'H' as ``Equality``
        - Constraints:
            - c_Q: :math:`Q_{in}(t) = - Q_{out}(t)`
            - c_H: :math:`H(t) = (n(t)/n_n)^2\cdot A - B \cdot Q_{out}(t)^2`
            - c_Ph: :math:`P_h(t) = 9810\cdot H(t)\cdot Q_{out}`
            - c_Pe: :math:`P_e(t) = P_h(t)/\eta`
    """

    # Parameters
    b.A = pyo.Param(initialize=data['A'])
    b.B = pyo.Param(initialize=data['B'])
    b.Qnom = pyo.Param(initialize=data['Qnom'])
    b.n_n = pyo.Param(initialize=data['n_n'])
    b.eff = pyo.Param(initialize=data['eff'])

    # Variables
    b.Qin = pyo.Var(t, initialize=[-k for k in init_data['Q']], bounds=(-data['Qmax']*data['Qnom'], 0), within=pyo.NonPositiveReals)
    b.Qout = pyo.Var(t, initialize=init_data['Q'], bounds=(0, data['Qmax']*data['Qnom']), within=pyo.NonNegativeReals)
    b.H = pyo.Var(t, initialize=init_data['H'], within=pyo.NonNegativeReals)
    b.n = pyo.Var(t, initialize=init_data['n'], within=pyo.NonNegativeReals)
    b.Ph = pyo.Var(t, initialize=init_data['Pe'], bounds=(0, data['Pmax']), within=pyo.NonNegativeReals)
    b.Pe = pyo.Var(t, initialize=init_data['Pe'], bounds=(0, data['Pmax']), within=pyo.NonNegativeReals)

    # Ports
    b.port_Qin = Port(initialize={'Q': (b.Qin, Port.Extensive)})
    b.port_Qout = Port(initialize={'Q': (b.Qout, Port.Extensive)})
    b.port_P = Port(initialize={'P': (b.Pe, Port.Extensive)})
    b.port_H = Port(initialize={'H': (b.H, Port.Equality)})

    # Constraints
    def Constraint_Q(_b, _t):
        return _b.Qin[_t] == -_b.Qout[_t]
    b.c_Q = pyo.Constraint(t, rule=Constraint_Q)

    def Constraint_H(_b, _t):
        return _b.H[_t] == (_b.n[_t]/_b.n_n)**2*_b.A - _b.B*_b.Qout[_t]**2
    b.c_H = pyo.Constraint(t, rule=Constraint_H)

    def Constraint_Ph(_b, _t):
        return _b.Ph[_t] == 9810*_b.H[_t]*_b.Qout[_t]
    b.c_Ph = pyo.Constraint(t, rule=Constraint_Ph)

    def Constraint_Pe(_b, _t):
        return _b.Pe[_t] == _b.Ph[_t]/_b.eff
    b.c_Pe = pyo.Constraint(t, rule=Constraint_Pe)



def RealPump(b, t, data, init_data):
    """
    In RealPump it's added the working flow limits of the pump.  
    Due to binary variables the problem with RealPump must be solved with "couenne" solver.
    
    It's introduced :math:`Q_{max}` and :math:`Q_{min}` which determines the flow limits in p.u

    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: data ``dict``
    :param init_data: init_data ``dict``

    data
         - 'A': Constant characteristic coefficient of the pump :math:`A`
         - 'B': Quadratic characteristic coefficient of the pump :math:`B`
         - 'n_n': Nominal rotational speed :math:`n_n`
         - 'Qnom': Nominal flow :math:`Q_n`
         - 'eff': Efficiency at the nominal operating point in p.u. :math:`\eta`
         - 'Qmax': Maximum allowed flow :math:`Q_{max}`
         - 'Qmin': Minimum flow :math:`Q_{min}`
         - 'Pmax': Maximum allowed power :math:`P_{max}`

    init_data
         - 'Q': Flow :math:`Q(t)` as a ``list``
         - 'H': Head :math:`H(t)` as a ``list``
         - 'n': Rotational speed :math:`n(t)` as a ``list``
         - 'Pe': Electrical power :math:`P_e(t)` as a ``list``

    Pyomo declarations    
        - Parameters:
            - A
            - B
            - Qnom
            - n_n
            - eff
            - Qmin
            - Qmax
        - Variables:
            - Qin (t) bounded :math:`Q_{in} \in [-Q_{max}, 0]`
            - Qout (t) bounded :math:`Q_{out} \in [0, Q_{max}]`
            - H (t) bounded :math:`H \ge 0`
            - n (t) bounded :math:`n \ge 0`
            - Ph (t) bounded :math:`P_h \in [0, P_{max}]`
            - Pe (t) bounded :math:`P_e \in [0, P_{max}]`
            - PumpOn (t) bounded :math:`PumpOn =  [0,1]`

        - Ports:
            - port_Qin @ Qin with 'Q' as ``Extensive``
            - port_Qout @ Qout with 'Q' as ``Extensive``
            - port_P @ Pe with 'P' as ``Extensive``
            - port_H @ H with 'H' as ``Equality``
        - Constraints:
            - c_Q: :math:`Q_{in}(t) = - Q_{out}(t)`
            - c_H: :math:`H(t) = (n(t)/n_n)^2\cdot A - B \cdot Q_{out}(t)^2`
            - c_Ph: :math:`P_h(t) = 9810\cdot H(t)\cdot Q_{out}`
            - c_Pe: :math:`P_e(t) = P_h(t)/\eta`
            - c_Qmax: :math:`Q_{out}(t) \leq Q_{nom} \cdot Q_{max} \cdot PumpOn(t)`
            - c_Qmin: :math:`Q_{out}(t) \geq Q_{nom} \cdot Q_{min} \cdot PumpOn(t)`
    """

    # Parameters
    b.A = pyo.Param(initialize=data['A'])
    b.B = pyo.Param(initialize=data['B'])
    b.Qnom = pyo.Param(initialize=data['Qnom'])
    b.n_n = pyo.Param(initialize=data['n_n'])
    b.eff = pyo.Param(initialize=data['eff'])
    b.Qmin = pyo.Param(initialize=data['Qmin']*data['Qnom'])
    b.Qmax = pyo.Param(initialize=data['Qmax']*data['Qnom'])

    # Variables
    b.Qin = pyo.Var(t, initialize=[-k for k in init_data['Q']], within=pyo.NonPositiveReals)
    b.Qout = pyo.Var(t, initialize=init_data['Q'], within=pyo.NonNegativeReals)
    b.H = pyo.Var(t, initialize=init_data['H'], within=pyo.NonNegativeReals)
    b.n = pyo.Var(t, initialize=init_data['n'], within=pyo.NonNegativeReals)
    b.Ph = pyo.Var(t, initialize=init_data['Pe'], bounds=(0, data['Pmax']), within=pyo.NonNegativeReals)
    b.Pe = pyo.Var(t, initialize=init_data['Pe'], bounds=(0, data['Pmax']), within=pyo.NonNegativeReals)
    b.PumpOn = pyo.Var(t, initialize=1, within=pyo.Binary)
    # b.PumpOn = pyo.Param(t, initialize=0, within=pyo.Binary)

    # Ports
    b.port_Qin = Port(initialize={'Q': (b.Qin, Port.Extensive)})
    b.port_Qout = Port(initialize={'Q': (b.Qout, Port.Extensive)})
    b.port_P = Port(initialize={'P': (b.Pe, Port.Extensive)})
    b.port_H = Port(initialize={'H': (b.H, Port.Equality)})

    # Constraints
    def Constraint_Q(_b, _t):
        return _b.Qin[_t] == -_b.Qout[_t]
    b.c_Q = pyo.Constraint(t, rule=Constraint_Q)

    def Constraint_H(_b, _t):
        return _b.H[_t] == (_b.n[_t]/_b.n_n)**2*_b.A - _b.B*_b.Qout[_t]**2
    b.c_H = pyo.Constraint(t, rule=Constraint_H)

    def Constraint_Ph(_b, _t):
        return _b.Ph[_t] == 9810*_b.H[_t]*_b.Qout[_t]
    b.c_Ph = pyo.Constraint(t, rule=Constraint_Ph)

    def Constraint_Pe(_b, _t):
        return _b.Pe[_t] == _b.Ph[_t]/_b.eff
    b.c_Pe = pyo.Constraint(t, rule=Constraint_Pe)

    def Constraint_Qmax(_b, _t):
        return _b.Qout[_t] <= _b.Qmax * _b.PumpOn[_t]
    b.c_Qmax = pyo.Constraint(t, rule=Constraint_Qmax)
    
    def Constraint_Qmin(_b, _t):
        return _b.Qout[_t] >= _b.Qmin * _b.PumpOn[_t]
    b.c_Qmin = pyo.Constraint(t, rule=Constraint_Qmin)


def RealPumpControlled(b, t, data, init_data):
    """   
    In RealPumpControl, three working regimes are added depending on the height (z) between the surfaces of the reservoirs.
    Due to binary variables the problem with RealPumpControlled must be solved with "couenne" solver.
        
    data
         - 'A': Constant characteristic coefficient of the pump :math:`A`
         - 'B': Quadratic characteristic coefficient of the pump :math:`B`
         - 'n_n': Nominal rotational speed :math:`n_n`
         - 'Qnom': Nominal flow :math:`Q_n`
         - 'Qmin': Minimum flow :math:`Q_{min}`
         - 'Qbep': Final BEP flow :math:`Q_{bep}`
         - 'Qpmax': Final Pmax flow :math:`Q_{pmax}`
         - 'eff': Efficiency at the nominal operating point in p.u. :math:`\eta`
         - 'Pmax': Maximum allowed power :math:`P_{max}`
         - 'zmin': Minimum heigh that the pump can operate :math:`z_{min}`

    init_data
         - 'Q': Flow :math:`Q(t)` as a ``list``
         - 'H': Head :math:`H(t)` as a ``list``
         - 'n': Rotational speed :math:`n(t)` as a ``list``
         - 'Pe': Electrical power :math:`P_e(t)` as a ``list``

    Pyomo declarations    
        - Parameters:
            - A
            - B
            - Qnom
            - Qmin
            - Qbep
            - Qpmax
            - n_n
            - eff
            - zmin

            
        - Variables:
            - Qin (t) bounded :math:`Q_{in} \in [-Q_{max}, 0]`
            - Qout (t) bounded :math:`Q_{out} \in [0, Q_{max}]`
            - H (t) bounded :math:`H \ge 0`
            - n (t) bounded :math:`n \ge 0`
            - Ph (t) bounded :math:`P_h \in [0, P_{max}]`
            - Pe (t) bounded :math:`P_e \in [0, P_{max}]`
            - PumpOn (t) bounded :math:`PumpOn =  [0,1]`
            - aux (t) bounded :math:`aux = [0,1]`
            - Qcontrol (t) bounded :math:`Q_{control} \in [Q_{min},{Q_{max}}]`
            
        - Ports:
            - port_Qin @ Qin with 'Q' as ``Extensive``
            - port_Qout @ Qout with 'Q' as ``Extensive``
            - port_P @ Pe with 'P' as ``Extensive``
            - port_H @ H with 'H' as ``Equality``
            
        - Constraints:
            - c_Q: :math:`Q_{in}(t) = - Q_{out}(t)`
            - c_H: :math:`H(t) = (n(t)/n_n)^2\cdot A - B \cdot Q_{out}(t)^2`
            - c_Ph: :math:`P_h(t) = 9810\cdot H(t)\cdot Q_{out}`
            - c_Pe: :math:`P_e(t) = P_h(t)/\eta`
            - c_Qcontrol: :math:`Qout(t) = Qcontrol(t)\cdot PumpOn(t) \cdot aux(t)`
            - c_Hmin1: :math:`H(t) \leq z_{min}  \cdot aux(t)`
            - c_Hmin2: :math:`z_{min} \leq H(t) \cdot (1 - aux(t))`
            - c_Binary: :math:`PumpOn(t) + aux(t) = 1`
            - c_Piecewise:
            .. math::
                f(x)=\begin{cases}
                        Q_{min} & \text{if} z{1} \leq z \leq z{2} \\
                        Q_{bep} & \text{if} z{2} \leq z \leq z{3} \\
                        Q_{pmax} & \text{if} z{3} \leq z \leq z{4} 
                \end{cases}

    """

    # Parameters
    b.A = pyo.Param(initialize=data['A'])
    b.B = pyo.Param(initialize=data['B'])
    b.Qnom = pyo.Param(initialize=data['Qnom'])
    b.n_n = pyo.Param(initialize=data['n_n'])
    b.eff = pyo.Param(initialize=data['eff'])
    b.Qmin = pyo.Param(initialize=data['Qmin'])
    b.Qbep = pyo.Param(initialize=data['Qbep'])
    b.Qpmax = pyo.Param(initialize=data['Qpmax'])
    b.zmin = pyo.Param(initialize=data['zmin'])

    # Variables
    b.Qin = pyo.Var(
        t, initialize=[-k for k in init_data['Q']], within=pyo.NonPositiveReals)
    b.Qout = pyo.Var(t, initialize=init_data['Q'], within=pyo.NonNegativeReals)
    b.H = pyo.Var(t, initialize=init_data['H'], bounds=(0, 35), within=pyo.NonNegativeReals)
    b.n = pyo.Var(t, initialize=init_data['n'], within=pyo.NonNegativeReals)
    b.Ph = pyo.Var(t, initialize=init_data['Pe'], bounds=(0, data['Pmax']), within=pyo.NonNegativeReals)
    b.Pe = pyo.Var(t, initialize=init_data['Pe'], bounds=(0, data['Pmax']), within=pyo.NonNegativeReals)
    b.PumpOn = pyo.Var(t, initialize=1, within=pyo.Binary)
    b.Qcontrol = pyo.Var(t, initialize=init_data['Q'], within=pyo.NonNegativeReals)
    b.aux = pyo.Var(t,initialize=1, within=pyo.Binary)
    
    # Ports
    b.port_Qin = Port(initialize={'Q': (b.Qin, Port.Extensive)})
    b.port_Qout = Port(initialize={'Q': (b.Qout, Port.Extensive)})
    b.port_P = Port(initialize={'P': (b.Pe, Port.Extensive)})
    b.port_H = Port(initialize={'H': (b.H, Port.Equality)})

    # PieceWise function
    #           /  Qmin      if z1<=z<=z2
    #    Q(z) = | Q(BEP)      if z2<=z<=z3
    #           \  Q(Pmax)      if z3<=z<=z4

    Q_min = b.Qmin * b.Qnom
    Q_bep = b.Qbep * b.Qnom
    Q_pmax = b.Qpmax * b.Qnom

    Domain_PTS = [0,20,20,30,30,35]  # Z1,Z2,Z3,z4
    Range_PTS = [Q_min,Q_min,Q_min,Q_bep,Q_bep,Q_pmax]

    b.con = Piecewise(
        t,
        b.Qcontrol,  # variable
        b.H,  # range and domain variables
        pw_pts=Domain_PTS,    # Domain points
        pw_constr_type='EQ',  # 'EQ' - Q variable is equal to the piecewise function
        f_rule=Range_PTS,     # Range points
        pw_repn='DCC',        # + 'DCC' - Disaggregated convex combination model.
        unbounded_domain_var=True)  # Indicates the type of piecewise representation to use


    # Constraints
    def Constraint_Q(_b, _t):
        return _b.Qin[_t] == -_b.Qout[_t]
    b.c_Q = pyo.Constraint(t, rule=Constraint_Q)

    def Constraint_H(_b, _t):
        return _b.H[_t] == (_b.n[_t]/_b.n_n)**2*_b.A - _b.B*_b.Qout[_t]**2
    b.c_H = pyo.Constraint(t, rule=Constraint_H)

    def Constraint_Ph(_b, _t):
        return _b.Ph[_t] == 9810*_b.H[_t]*_b.Qout[_t]
    b.c_Ph = pyo.Constraint(t, rule=Constraint_Ph)

    def Constraint_Pe(_b, _t):
        return _b.Pe[_t] == _b.Ph[_t]/_b.eff
    b.c_Pe = pyo.Constraint(t, rule=Constraint_Pe)

    def Constraint_Control(_b, _t):
        return _b.Qout[_t] == _b.Qcontrol[_t] * _b.PumpOn[_t] * _b.aux[_t]
    b.c_Qcontrol = pyo.Constraint(t, rule=Constraint_Control)

    def Constraint_Hmin1(_b,_t):
        return _b.H[_t] >= _b.zmin * _b.aux[_t]
    b.c_hmin  =pyo.Constraint(t, rule=Constraint_Hmin1)

    def Constraint_Hmin2(_b,_t):
        return _b.zmin >= _b.H[_t] * (1 - _b.aux[_t])
    b.c_hmin2 = pyo.Constraint(t,rule = Constraint_Hmin2)
    
    def Constraint_Binary(_b,_t):
        return _b.PumpOn[_t] + _b.aux[_t] >= 1
    b.c_bins = pyo.Constraint(t, rule=Constraint_Binary)
    
    
def ReversiblePump(b, t, data, init_data):
    """
    Frequency controllable hydraulic pump.

    It's introduced :math:`Q_{max}` and :math:`Q_{min}` which determines the flow limits
    in addition to the reversibility of the pump.


    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: data ``dict``
    :param init_data: init_data ``dict``

    data
         - 'A': Constant characteristic coefficient of the pump :math:`A`
         - 'B': Quadratic characteristic coefficient of the pump :math:`B`
         - 'n_n': Nominal rotational speed :math:`n_n`
         - 'Qnom': Nominal flow :math:`Q_n`
         - 'eff': Efficiency at the nominal operating point in p.u. :math:`\eta`
         - 'Qmax': Maximum allowed flow :math:`Q_{max}`
         - 'Qmin': 
         - 'Pmax': Maximum allowed power :math:`P_{max}`

    init_data
         - 'Q': Flow :math:`Q(t)` as a ``list``
         - 'H': Head :math:`H(t)` as a ``list``
         - 'n': Rotational speed :math:`n(t)` as a ``list``
         - 'Pe': Electrical power :math:`P_e(t)` as a ``list``

    Pyomo declarations    
        - Parameters:
            - A
            - B
            - Qnom
            - n_n
            - eff
        - Variables:
            - Qin (t) bounded :math:`Q_{in} \in [-Q_{max}, 0]`
            - Qout (t) bounded :math:`Q_{out} \in [0, Q_{max}]`
            - H (t) bounded :math:`H \ge 0`
            - n (t) bounded :math:`n \ge 0`
            - Ph (t) bounded :math:`P_h \in [0, P_{max}]`
            - Pe (t) bounded :math:`P_e \in [0, P_{max}]`
        - Ports:
            - port_Qin @ Qin with 'Q' as ``Extensive``
            - port_Qout @ Qout with 'Q' as ``Extensive``
            - port_P @ Pe with 'P' as ``Extensive``
            - port_H @ H with 'H' as ``Equality``
        - Constraints:
            - c_Q: :math:`Q_{in}(t) = - Q_{out}(t)`
            - c_H: :math:`H(t) = (n(t)/n_n)^2\cdot A - B \cdot Q_{out}(t)^2`
            - c_Ph: :math:`P_h(t) = 9810\cdot H(t)\cdot Q_{out}`
            - c_Pe: :math:`P_e(t) = P_h(t)/\eta`

    """

    # Parameters
    b.A = pyo.Param(initialize=data['A'])
    b.B = pyo.Param(initialize=data['B'])
    b.Qnom = pyo.Param(initialize=data['Qnom'])
    b.n_n = pyo.Param(initialize=data['n_n'])
    b.eff = pyo.Param(initialize=data['eff'])
    b.eff_t = pyo.Param(initialize=data['eff_t'])
    b.S = pyo.Param(initialize=data['S'])
    b.K = pyo.Var(t,initialize = 1, bounds = (1e-6,1), within=pyo.NonNegativeReals)


    # Variables
    b.Qin = pyo.Var(t, initialize=[-k for k in init_data['Q']], bounds=(-data['Qmax']*data['Qnom'], data['Qmax']*data['Qnom']), within=pyo.Reals)
    b.Qout = pyo.Var(t, initialize=init_data['Q'], bounds=(-data['Qmax']*data['Qnom'], data['Qmax']*data['Qnom']), within=pyo.Reals)
    b.Qoutp = pyo.Var(t, initialize=init_data['Q'], bounds=(0, data['Qmax']*data['Qnom']), within=pyo.NonNegativeReals)
    b.Qoutt = pyo.Var(t, initialize=init_data['Q'], bounds=(0, data['Qmax']*data['Qnom']), within=pyo.NonNegativeReals)
    b.H = pyo.Var(t, initialize=init_data['H'], within=pyo.NonNegativeReals)
    b.n = pyo.Var(t, initialize=init_data['n'], within=pyo.NonNegativeReals)
    b.Ph = pyo.Var(t, initialize=init_data['Pe'], bounds=(-data['Pmax'], data['Pmax']), within=pyo.Reals)
    b.Pe = pyo.Var(t, initialize=init_data['Pe'], bounds=(-data['Pmax'], data['Pmax']), within=pyo.Reals)
    b.ModePump = pyo.Var(t, initialize=1, within=pyo.Binary)
    # b.alpha = pyo.Var(t, initialize=0, within=pyo.NonNegativeReals)

    # Ports
    b.port_Qin = Port(initialize={'Q': (b.Qin, Port.Extensive)})
    b.port_Qout = Port(initialize={'Q': (b.Qout, Port.Extensive)})
    b.port_P = Port(initialize={'P': (b.Pe, Port.Extensive)})
    b.port_H = Port(initialize={'H': (b.H, Port.Equality)})

    # Constraints
    # def Constraint_aux1(_b, _t):
    #     return _b.alpha[_t] + _b.ModePump[_t] == 1
    # b.c_aux1 = pyo.Constraint(t, rule=Constraint_aux1)
    # def Constraint_aux2(_b, _t):
    #     return _b.alpha[_t] * _b.ModePump[_t] == 0
    # b.c_aux2 = pyo.Constraint(t, rule=Constraint_aux2)
        
    
    def Constraint_Q(_b, _t):
        return _b.Qin[_t] == -_b.Qout[_t]
    b.c_Q = pyo.Constraint(t, rule=Constraint_Q)

    def Constraint_H(_b, _t):
        return _b.H[_t] == ((_b.n[_t]/_b.n_n)**2*_b.A - _b.B*_b.Qoutp[_t]**2)*_b.ModePump[_t] + _b.Qoutt[_t]**2/(2*9.81*_b.S**2*_b.K[_t])*(1-_b.ModePump[_t])
    b.c_H = pyo.Constraint(t, rule=Constraint_H)

    
    # def Constraint_Qoutt(_b, _t):
    #     return _b.Qoutt[_t] == (2*9.81*_b.H0[_t])**0.5*_b.S
    # b.c_Qoutt = pyo.Constraint(t, rule=Constraint_Qoutt)
    
    def Constraint_Qout(_b, _t):
        return _b.Qout[_t] == +_b.Qoutp[_t]*_b.ModePump[_t] - _b.Qoutt[_t]*(1-_b.ModePump[_t])
    b.c_Qout = pyo.Constraint(t, rule=Constraint_Qout)

    def Constraint_Ph(_b, _t):
        return _b.Ph[_t] == 9810*_b.H[_t]*_b.Qout[_t]
    b.c_Ph = pyo.Constraint(t, rule=Constraint_Ph)

    def Constraint_Pe(_b, _t):
        return _b.Pe[_t] == _b.Ph[_t] *(_b.ModePump[_t]/_b.eff + (1-_b.ModePump[_t])*_b.eff_t)
    b.c_Pe = pyo.Constraint(t, rule=Constraint_Pe)


def ReversibleRealPump(b, t, data, init_data):
    '''
    Frequency controllable hydraulic pump with water flow limits.

    Applies the hydraulic head equation to a flow, with the ability to change its rotational speed, limited by a maximum power:
    A

    .. math:
        H(t) = (n(t)/n_n)^2\cdot A - B \cdot Q(t)^2

    with :math:`A` and :math:`B` the characteristic coefficients that define the behaviour of the pump (i.e its curve).

    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: data ``dict``
    :param init_data: init_data ``dict``

    data
         - 'A': Constant characteristic coefficient of the pump :math:`A`
         - 'B': Quadratic characteristic coefficient of the pump :math:`B`
         - 'n_n': Nominal rotational speed :math:`n_n`
         - 'Qnom': Nominal flow :math:`Q_n`
         - 'eff': Efficiency at the nominal operating point in p.u. :math:`\eta`
         - 'Qmax': Maximum allowed flow in p.u. :math:`Q_{max}`
         - 'Qmin': Minimum flow in p.u. :math:`Q_{min}`
         - 'Pmax': Maximum allowed power :math:`P_{max}`

    init_data
         - 'Q': Flow :math:`Q(t)` as a ``list``
         - 'H': Head :math:`H(t)` as a ``list``
         - 'n': Rotational speed :math:`n(t)` as a ``list``
         - 'Pe': Electrical power :math:`P_e(t)` as a ``list``

    Pyomo declarations    
        - Parameters:
            - A
            - B
            - Qnom
            - n_n
            - eff
            - Qmax
            - Qmin
        - Variables:
            - Qin (t) bounded :math:`Q_{in} \in [-Q_{max}, 0]`
            - Qout (t) bounded :math:`Q_{out} \in [0, Q_{max}]`
            - Qoutp (t)
            - Qoutt (t) bounded :math;`Q_{outt} \in [1e-6,Q_{max}]`
            - H (t) bounded :math:`H \ge 0`
            - n (t) bounded :math:`n \ge 0`
            - Ph (t) bounded :math:`P_h \in [0, P_{max}]`
            - Pe (t) bounded :math:`P_e \in [0, P_{max}]`
            - ModePump (t) bounded :math:`ModePump =  [0,1]`
            - ModeTurbine (t) bounded :math:`ModeTurbine = [0,1]`
            - K (t) bounded :math:`K = [0,1]`
            
        - Ports:
            - port_Qin @ Qin with 'Q' as ``Extensive``
            - port_Qout @ Qout with 'Q' as ``Extensive``
            - port_P @ Pe with 'P' as ``Extensive``
            - port_H @ H with 'H' as ``Equality``
        - Constraints:
            - c_Q: :math:`Q_{in}(t) = - Q_{out}(t)`
            - c_H: :math:`H(t) = (n(t)/n_n)^2\cdot A - B \cdot Q_{outp}(t)^2 + Q_{outt}^2/(2 \cdot g \cdot S**2)*ModeTurbine(t)`
            - c_Ph: :math:`P_h(t) = 9810\cdot H(t)\cdot Q_{out}`
            - c_Pe: :math:`P_e(t) = P_h(t) \cdot (ModePump(t)/\eta_{Pump} + ModeTurbine(t) \cdot \eta_{Turbine})`
            - c_Qmax: :math:`Q_{out}(t) \leq Q_{max} \cdot ModePump(t)`
            - c_Qmin: :math:`Q_{out}(t) \geq Q_{min} \cdot ModePump(t)`
            - c_WorkingBin :math:`ModeTurbine(t) + ModePump(t) = 1`
            
    '''

    
    # Parameters
    b.A = pyo.Param(initialize=data['A'])
    b.B = pyo.Param(initialize=data['B'])
    b.Qnom = pyo.Param(initialize=data['Qnom'])
    b.n_n = pyo.Param(initialize=data['n_n'])
    b.eff = pyo.Param(initialize=data['eff'])
    b.eff_t = pyo.Param(initialize=data['eff_t'])
    b.S = pyo.Param(initialize=data['S'])
    b.Qmin = pyo.Param(initialize=data['Qmin']*data['Qnom'])
    b.Qmax = pyo.Param(initialize=data['Qmax']*data['Qnom'])


    # Variables
    b.Qin = pyo.Var(t, initialize=[-k for k in init_data['Q']], bounds=(-data['Qmax']*data['Qnom'], data['Qmax']*data['Qnom']), within=pyo.Reals)
    b.Qout = pyo.Var(t, initialize=init_data['Q'], bounds=(-data['Qmax']*data['Qnom'], data['Qmax']*data['Qnom']), within=pyo.Reals)
    b.Qoutp = pyo.Var(t, initialize=init_data['Q'], bounds=(0, data['Qmax']*data['Qnom']), within=pyo.NonNegativeReals)
    b.Qoutt = pyo.Var(t, initialize=init_data['Q'], bounds=(0, data['Qmax']*data['Qnom']), within=pyo.NonNegativeReals)
    b.H = pyo.Var(t, initialize=init_data['H'], within=pyo.NonNegativeReals)
    b.n = pyo.Var(t, initialize=init_data['n'], within=pyo.NonNegativeReals)
    b.Ph = pyo.Var(t, initialize=init_data['Pe'], bounds=(-data['Pmax'], data['Pmax']), within=pyo.Reals)
    b.Pe = pyo.Var(t, initialize=init_data['Pe'], bounds=(-data['Pmax'], data['Pmax']), within=pyo.Reals)
    # b.ModePump = pyo.Var(t, initialize=1, within=pyo.Binary)
    # b.ModeTurbine = pyo.Var(t,initialize=0, within=pyo.Binary)
    b.a = pyo.Var(t,initialize = init_data['H'][0], within=pyo.NonNegativeReals)
    b.K = pyo.Var(t,initialize = 1, bounds = (0.01,1), within=pyo.NonNegativeReals)
    
    b.ON = pyo.Var(t,initialize=1, within=pyo.Binary)
    b.PumpTurbine = pyo.Var(t,initialize=1, within=pyo.Binary)


    # Ports
    b.port_Qin = Port(initialize={'Q': (b.Qin, Port.Extensive)})
    b.port_Qout = Port(initialize={'Q': (b.Qout, Port.Extensive)})
    b.port_P = Port(initialize={'P': (b.Pe, Port.Extensive)})
    b.port_H = Port(initialize={'H': (b.H, Port.Equality)})

    
    def Constraint_Q(_b, _t):
        return _b.Qin[_t] == -_b.Qout[_t]
    b.c_Q = pyo.Constraint(t, rule=Constraint_Q)

    def Constraint_H(_b, _t):
        # return _b.H[_t] == ((_b.n[_t]/_b.n_n)**2*_b.A - _b.B*_b.Qoutp[_t]**2)*_b.ModePump[_t] + _b.Qoutt[_t]**2/(2*9.81*_b.S**2*_b.K[_t])*_b.ModeTurbine[_t] + (1-(_b.ModePump[_t] +_b.ModeTurbine[_t]))*_b.a[_t]
        return _b.H[_t] == _b.ON[_t]*( ((_b.n[_t]/_b.n_n)**2*_b.A - _b.B*_b.Qoutp[_t]**2)*_b.PumpTurbine[_t] + _b.Qoutt[_t]**2/(2*9.81*_b.S**2*_b.K[_t])*(1-_b.PumpTurbine[_t]) ) + (1-_b.ON[_t])*_b.a[_t]
    b.c_H = pyo.Constraint(t, rule=Constraint_H)
    
    def Constraint_Qout(_b, _t):
        # return _b.Qout[_t] == +_b.Qoutp[_t]*_b.ModePump[_t] - _b.Qoutt[_t]*_b.ModeTurbine[_t]
        return _b.Qout[_t] == _b.ON[_t]*(_b.Qoutp[_t]*_b.PumpTurbine[_t] - _b.Qoutt[_t]*(1-_b.PumpTurbine[_t]))
    b.c_Qout = pyo.Constraint(t, rule=Constraint_Qout)

    def Constraint_Ph(_b, _t):
        return _b.Ph[_t] == 9810*_b.H[_t]*_b.Qout[_t]
    b.c_Ph = pyo.Constraint(t, rule=Constraint_Ph)

    def Constraint_Pe(_b, _t):
        # return _b.Pe[_t] == _b.Ph[_t] *(_b.ModePump[_t]/_b.eff + (_b.ModeTurbine[_t])*_b.eff_t)
        return _b.Pe[_t] == _b.Ph[_t] *(_b.PumpTurbine[_t]/_b.eff + (1-_b.PumpTurbine[_t])*_b.eff_t)
    b.c_Pe = pyo.Constraint(t, rule=Constraint_Pe)
    
    def Constraint_Qmaxp(_b, _t):
        return _b.Qoutp[_t] <= (_b.Qmax *_b.ON[_t]*_b.PumpTurbine[_t])
    b.c_Qmaxp = pyo.Constraint(t, rule=Constraint_Qmaxp)
    
    def Constraint_Qminp(_b, _t):
        return _b.Qoutp[_t] >= (_b.Qmin *_b.ON[_t]*_b.PumpTurbine[_t])
    b.c_Qminp = pyo.Constraint(t, rule=Constraint_Qminp)
    
    # def Constraint_WorkingBin(_b, _t):
    #     return _b.ModePump[_t] + _b.ModeTurbine[_t] <= 1
    # b.c_workingbin = pyo.Constraint(t, rule = Constraint_WorkingBin)
   
