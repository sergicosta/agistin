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

    .. math::
        H(t) = (n(t)/n_n)^2\, A - B \, Q(t)^2

    with :math:`A` and :math:`B` the characteristic coefficients that define the behaviour of the pump (i.e its curve).

    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: data ``dict``
    :param init_data: init_data ``dict``

    data
         - 'A': Constant characteristic coefficient of the pump :math:`A`
         - 'B': Quadratic characteristic coefficient of the pump :math:`B`
         - 'Qnom': Nominal flow :math:`Q_n`
         - 'eff': Efficiency at the nominal operating point in p.u. :math:`\eta`
         - 'Qmax': Maximum allowed flow (in p.u. of Qnom) :math:`Q_{max}`
         - 'Pmax': Maximum allowed power :math:`P_{max}`

    init_data
         - 'Q': Flow :math:`Q(t)` as a ``list``
         - 'H': Head :math:`H(t)` as a ``list``
         - 'n': Rotational speed (in p.u.) :math:`n(t)` as a ``list``
         - 'Pe': Electrical power :math:`P_e(t)` as a ``list``

    Pyomo declarations    
        - Parameters:
            - A
            - B
            - Qnom
            - eff
        - Variables:
            - Qin (t) bounded :math:`Q_{in} \in [-Q_{max}\,Q_{nom}, 0]`
            - Qout (t) bounded :math:`Q_{out} \in [0, Q_{max}\,Q_{nom}]`
            - H (t) bounded :math:`H \ge 0`
            - npu2 (t) bounded :math:`n \ge 0`
            - Ph (t) bounded :math:`P_h \in [0, P_{max}]`
            - Pe (t) bounded :math:`P_e \in [0, P_{max}]`
        - Ports:
            - port_Qin @ Qin with 'Q' as ``Extensive``
            - port_Qout @ Qout with 'Q' as ``Extensive``
            - port_P @ Pe with 'P' as ``Extensive``
            - port_H @ H with 'H' as ``Equality``
        - Constraints:
            - c_Q: :math:`Q_{in}(t) = - Q_{out}(t)`
            - c_H: :math:`H(t) = (n(t)/n_n)^2\, A - B \, Q_{out}(t)^2`
            - c_Ph: :math:`P_h(t) = 9810\, H(t)\, Q_{out}`
            - c_Pe: :math:`P_e(t) = P_h(t)/\eta`
    """

    # Parameters
    b.A = pyo.Param(initialize=data['A'])
    b.B = pyo.Param(initialize=data['B'])
    b.Qnom = pyo.Param(initialize=data['Qnom'])
    b.eff = pyo.Param(initialize=data['eff'])

    # Variables
    b.Qin = pyo.Var(t, initialize=[-k for k in init_data['Q']], bounds=(-data['Qmax']*data['Qnom'], 0), within=pyo.NonPositiveReals)
    b.Qout = pyo.Var(t, initialize=init_data['Q'], bounds=(0, data['Qmax']*data['Qnom']), within=pyo.NonNegativeReals)
    b.H = pyo.Var(t, initialize=init_data['H'], within=pyo.NonNegativeReals)
    b.Ph = pyo.Var(t, initialize=[k*data['eff'] for k in init_data['Pe']], bounds=(0, data['Pmax']*data['eff']), within=pyo.NonNegativeReals)
    b.Pe = pyo.Var(t, initialize=init_data['Pe'], bounds=(0, data['Pmax']), within=pyo.NonNegativeReals)
    b.npu2 = pyo.Var(t, initialize=init_data['n'], bounds=(0,data['nmax']), within=pyo.NonNegativeReals)
    
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
        return _b.H[_t] == (_b.npu2[_t])*_b.A - _b.B*_b.Qout[_t]**2
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
    Be wary of binary variables when choosing a solver.
    
    It's introduced :math:`Q_{max}` and :math:`Q_{min}` which determines the working limits.

    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: data ``dict``
    :param init_data: init_data ``dict``

    data
         - 'A': Constant characteristic coefficient of the pump :math:`A`
         - 'B': Quadratic characteristic coefficient of the pump :math:`B`
         - 'Qnom': Nominal flow :math:`Q_n`
         - 'eff': Efficiency at the nominal operating point in p.u. :math:`\eta`
         - 'Qmax': Maximum allowed flow (in p.u. of Qnom) :math:`Q_{max}`
         - 'Qmin': Minimum working flow (in p.u. of Qnom) :math:`Q_{min}`
         - 'Pmax': Maximum allowed power :math:`P_{max}`

    init_data
         - 'Q': Flow :math:`Q(t)` as a ``list``
         - 'H': Head :math:`H(t)` as a ``list``
         - 'n': Rotational speed (in p.u.) :math:`n(t)` as a ``list``
         - 'Pe': Electrical power :math:`P_e(t)` as a ``list``

    Pyomo declarations    
        - Parameters:
            - A
            - B
            - Qnom
            - eff
            - Qmin
            - Qmax
        - Variables:
            - Qin (t) bounded :math:`Q_{in} \in [-Q_{max}\,Q_{n}, 0]`
            - Qout (t) bounded :math:`Q_{out} \in [0, Q_{max}\,Q_{n}]`
            - H (t) bounded :math:`H \ge 0`
            - npu2 (t) bounded :math:`n \ge 0`
            - Ph (t) bounded :math:`P_h \in [0, P_{max}]`
            - Pe (t) bounded :math:`P_e \in [0, P_{max}]`
            - PumpOn (t) :math:`b_{ON}(t) \in \{0,1\}`

        - Ports:
            - port_Qin @ Qin with 'Q' as ``Extensive``
            - port_Qout @ Qout with 'Q' as ``Extensive``
            - port_P @ Pe with 'P' as ``Extensive``
            - port_H @ H with 'H' as ``Equality``
        - Constraints:
            - c_Q: :math:`Q_{in}(t) = - Q_{out}(t)`
            - c_H: :math:`H(t) = (n(t)/n_n)^2\, A - B \, Q_{out}(t)^2`
            - c_Ph: :math:`P_h(t) = 9810\, H(t)\, Q_{out}(t)`
            - c_Pe: :math:`P_e(t) = P_h(t)/\eta`
            - c_Qmax: :math:`Q_{out}(t) \leq Q_{n} \, Q_{max} \, b_{ON}(t)`
            - c_Qmin: :math:`Q_{out}(t) \geq Q_{n} \, Q_{min} \, b_{ON}(t)`
    """

    # Parameters
    b.A = pyo.Param(initialize=data['A'])
    b.B = pyo.Param(initialize=data['B'])
    b.Qnom = pyo.Param(initialize=data['Qnom'])
    b.eff = pyo.Param(initialize=data['eff'])
    b.Qmin = pyo.Param(initialize=data['Qmin']*data['Qnom'])
    b.Qmax = pyo.Param(initialize=data['Qmax']*data['Qnom'])
    # b.npmax = pyo.Param(initialize=data['nmax'])

    # Variables
    b.Qin = pyo.Var(t, initialize=[-k for k in init_data['Q']], bounds=(-data['Qmax']*data['Qnom'], 0), within=pyo.NonPositiveReals)
    b.Qout = pyo.Var(t, initialize=init_data['Q'], bounds=(0, data['Qmax']*data['Qnom']), within=pyo.NonNegativeReals)
    b.H = pyo.Var(t, initialize=init_data['H'],  bounds=(0, data['A']), within=pyo.NonNegativeReals)
    b.Ph = pyo.Var(t, initialize=[k*data['eff'] for k in init_data['Pe']], bounds=(0, data['Pmax']*data['eff']), within=pyo.NonNegativeReals)
    b.Pe = pyo.Var(t, initialize=init_data['Pe'], bounds=(0, data['Pmax']), within=pyo.NonNegativeReals)
    b.PumpOn = pyo.Var(t, initialize=1, within=pyo.Binary)
    # b.PumpOn = pyo.Var(t, initialize=1, bounds=(0,1), within=pyo.NonNegativeReals)
    b.npu2 = pyo.Var(t, initialize=init_data['n'], bounds=(0,data['nmax']), within=pyo.NonNegativeReals)


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
        return _b.H[_t] == _b.npu2[_t]*_b.A - _b.B*_b.Qout[_t]**2
    b.c_H = pyo.Constraint(t, rule=Constraint_H)
    
    # def Constraint_H(_b, _t):
    #     return _b.H[_t] <= _b.npmax**2*_b.A - _b.B*_b.Qout[_t]**2
    # b.c_H = pyo.Constraint(t, rule=Constraint_H)


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
    
    # def Constraint_onoff(_b, _t):
    #     return _b.PumpOn[_t]*(1-_b.PumpOn[_t]) == 0
    # b.c_onoff = pyo.Constraint(t, rule=Constraint_onoff)


def RealPumpControlled(b, t, data, init_data):
    """   
    In RealPumpControlled, three working regimes are added depending on the height (z) between the surfaces of the reservoirs.
        
    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: data ``dict``
    :param init_data: init_data ``dict``
    
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
            - c_Piecewise: :math:`Q = \begin{cases} Q_{min} & \text{if} z{1} \le z \le z{2}\\ Q_{bep} & \text{if} z{2} \le z \le z{3}\\ Q_{pmax} & \text{if} z{3} \le z \le z{4} \end{cases}` 
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
    Frequency controllable reversible hydraulic pump.
    Introduces the reversibility of the pump's flow.


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
         - 'Qmin': Minimum working flow (in p.u. of Qnom) :math:`Q_{min}`
         - 'Pmax': Maximum allowed power :math:`P_{max}`
         - 'eff_t': Efficiency at turbine mode :math:`\eta_{t}`

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
            - eff_t
        - Variables:
            - Qin (t) bounded :math:`Q_{in} \in [-Q_{max}\,Q_{nom}, Q_{max}\,Q_{nom}]`
            - Qout (t) bounded :math:`Q_{out} \in [-Q_{max}\,Q_{nom}, Q_{max}\,Q_{nom}]`
            - Qoutp (t) bounded :math:`Q_{out,p} \in [0, Q_{max}\,Q_{nom}]`
            - Qoutt (t) bounded :math:`Q_{out,t} \in [0, Q_{max}\,Q_{nom}]`
            - H (t) bounded :math:`H \ge 0`
            - n (t) bounded :math:`n \ge 0`
            - Ph (t) bounded :math:`P_h \in [-P_{max}, P_{max}]`
            - Pe (t) bounded :math:`P_e \in [-P_{max}, P_{max}]`
            - ModePump (t) :math:`b_{Mode}(t) \in \{0,1\}`
            - K (t) bounded :math:`K(t) \in [0,1]`
        - Ports:
            - port_Qin @ Qin with 'Q' as ``Extensive``
            - port_Qout @ Qout with 'Q' as ``Extensive``
            - port_P @ Pe with 'P' as ``Extensive``
            - port_H @ H with 'H' as ``Equality``
        - Constraints:
            - c_Q: :math:`Q_{in}(t) = - Q_{out}(t)`
            - c_H: :math:`H(t) = [(n(t)/n_n)^2\, A - B \, Q_{out,p}(t)^2]\,b_{Mode} + Q_{out,t}^2/(2\cdot9.81\, S^2 \, K(t))\,(1-b_{Mode})`
            - c_Qout: :math:`Q_{out}(t) = Q_{out,p}\,b_{Mode} - Q_{out,t}(1-b_{Mode})`
            - c_Ph: :math:`P_h(t) = 9810\cdot H(t)\cdot Q_{out}`
            - c_Pe: :math:`P_e(t) = P_h(t)(b_{Mode}/\eta + (1-b_{Mode})\eta_t)`

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
    Frequency controllable reversible hydraulic pump with water flow limits.

    Applies the hydraulic head equation to a flow, with the ability to change its rotational speed, limited by a maximum power:

    .. math::
        H(t) = (n(t)/n_n)^2\cdot A - B \cdot Q(t)^2

    with :math:`A` and :math:`B` the characteristic coefficients that define the behaviour of the pump (i.e its curve).

    Since it is a reversible pump it may operate as a turbine. This is defined with a binary variable :math:`b_{Mode}` (1 for pump, 0 for turbine).
    
    .. math::
        H(t) = ( (n(t)/n_n)^2\, A - B \, Q_{out,p}(t)^2 )\, b_{Mode}(t) + (1 - b_{Mode}(t))\, a_h(t)
        
    with :math:`a_h` a free variable which will take the appropiate head value when operating as a turbine.

    The code makes use of two auxiliary variables, :math:`a_q` and :math:`a_a`, to reduce the order such that:
        
    .. math::
        a_q(t) = Q_{out,p}(t)\, b_{Mode}(t)
        
    .. math::
        a_a(t) = a_h(t)\, b_{Mode}(t)
        
    .. math::
        H(t) = (n(t)/n_n)^2\, A - B\, Q_{out,p}(t)\,a_q(t) + (a_h(t) - a_a(t)) 


    It also introduces :math:`Q_{max}` and :math:`Q_{min}` which determines the flow limits when working as a pump:
    
    .. math::
       Q_{min}\, Q_n\, b_{Mode} \leq Q_{out,p}(t) \leq Q_{max}\, Q_n\, b_{Mode}
       
    and the upper limit of running as a turbine:
        
    .. math::
        Q_{out,t}(t) \leq S\,\sqrt{2\cdot 9.81\,H(t)}\,(1-b_{Mode}(t))
        
    The way both flows for pump and turbine modes, :math:`Q_{out,p}` and :math:`Q_{out,t}` respectively, 
        

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
         - 'eff_t': Efficiency at turbine mode :math:`\eta_{t}`

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
            - eff_t
            - S
            - Qmin
            - Qmax
            
        - Variables:
            - Qin (t) bounded :math:`Q_{in} \in [-Q_{max}\,Q_n, Q_{max}\,Q_n]`
            - Qout (t) bounded :math:`Q_{out} \in [-Q_{max}\,Q_n, Q_{max}\,Q_n]`
            - Qoutp (t) bounded :math:`Q_{out,p} \in [0, Q_{max}\,Q_n]`
            - Qoutt (t) bounded :math:`Q_{out,t} \in [0, Q_{max}\,Q_n]`
            - H (t) bounded :math:`H \ge 0`
            - n (t) bounded :math:`n \ge 0`
            - Ph (t) bounded :math:`P_h \in [-P_{max}, P_{max}]`
            - Pe (t) bounded :math:`P_e \in [-P_{max}, P_{max}]`
            - ModePump (t) :math:`b_{Mode}(t) \in \{0,1\}`
            - a (t) :math:`a \ge 0`
            - aux (t) :math:`a_q \ge 0`
            - auxa (t) :math:`a_a \ge 0`
            
        - Ports:
            - port_Qin @ Qin with 'Q' as ``Extensive``
            - port_Qout @ Qout with 'Q' as ``Extensive``
            - port_P @ Pe with 'P' as ``Extensive``
            - port_H @ H with 'H' as ``Equality``
            
        - Constraints:
            - c_Q: :math:`Q_{in}(t) = - Q_{out}(t)`
            - c_H: :math:`H(t) = (n(t)/n_n)^2\, A - B\, Q_{out,p}(t)\,a_q(t) + (a_h(t) - a_a(t))`
            - c_aux1: :math:`a_q(t) \leq Q_{max}\, b_{Mode}(t)`
            - c_aux2: :math:`a_q(t) \leq Q_{out,p}(t)`
            - c_aux3: :math:`a_q(t) \geq Q_{out,p}(t) - Q_{max}\,(1 - b_{Mode}(t))`
            - c_auxa1: :math:`a_a(t) \leq A\, b_{Mode}(t)`
            - c_auxa2: :math:`a_a(t) \leq a_h(t)`
            - c_auxa3: :math:`a_a(t) \geq a_h(t) - A\,(1 - b_{Mode}(t))`
            - c_Qout: :math:`Q_{out}(t) = Q_{out,p}(t) - Q_{out,t}(t)`
            - c_Ph: :math:`P_h(t) = 9810\, H(t)\, Q_{out}(t)`
            - c_Pe: :math:`P_e(t) = P_h(t) (b_{Mode}(t)/\eta + (1-b_{Mode}(t))\, \eta_{t})`
            - c_Qmaxp: :math:`Q_{out,p}(t) \leq Q_{max} \, b_{Mode}(t)`
            - c_Qminp: :math:`Q_{out,p}(t) \geq Q_{min} \, b_{Mode}(t)`
            - c_Qmaxt: :math:`Q_{out,t}(t) \leq S\,\sqrt{2\cdot 9.81\,H(t)}\,(1-b_{Mode}(t))`
    '''
    _b.Qoutt[_t] <= (_b.H[_t]*2*9.81)**0.5*_b.S*(1-_b.ModePump[_t])
    
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
    b.a = pyo.Var(t,initialize = 0, within=pyo.NonNegativeReals)
    b.ModePump = pyo.Var(t,initialize=1, within=pyo.Binary)
    
    b.aux = pyo.Var(t, initialize=init_data['Q'], within=pyo.NonNegativeReals)
    b.auxa = pyo.Var(t, initialize=init_data['Q'], within=pyo.NonNegativeReals)

    # Ports
    b.port_Qin = Port(initialize={'Q': (b.Qin, Port.Extensive)})
    b.port_Qout = Port(initialize={'Q': (b.Qout, Port.Extensive)})
    b.port_P = Port(initialize={'P': (b.Pe, Port.Extensive)})
    b.port_H = Port(initialize={'H': (b.H, Port.Equality)})

    
    def Constraint_Q(_b, _t):
        return _b.Qin[_t] == -_b.Qout[_t]
    b.c_Q = pyo.Constraint(t, rule=Constraint_Q)

    def Constraint_H(_b, _t):
        # return _b.H[_t] ==((_b.n[_t]/_b.n_n)**2*_b.A - _b.B*_b.Qoutp[_t]**2)*_b.ModePump[_t] + (1-_b.ModePump[_t])*_b.a[_t]
        # return _b.H[_t] ==(_b.n[_t]/_b.n_n)**2*_b.A*_b.ModePump[_t] - _b.B*_b.Qoutp[_t]*_b.aux[_t] + (1-_b.ModePump[_t])*_b.a[_t]
        # return _b.H[_t] ==(_b.n[_t]/_b.n_n)**2*_b.A - _b.B*_b.Qoutp[_t]*_b.aux[_t] + (1-_b.ModePump[_t])*_b.a[_t]
        return _b.H[_t] ==(_b.n[_t]/_b.n_n)**2*_b.A - _b.B*_b.Qoutp[_t]*_b.aux[_t] + (_b.a[_t]-_b.auxa[_t])
    b.c_H = pyo.Constraint(t, rule=Constraint_H)
    
    # From AIMMS Ch 7.7: Elimination of products of variables
    def Constraint_aux1(_b,_t):
        return _b.aux[_t] <= _b.Qmax*_b.ModePump[_t]
    b.c_aux1 = pyo.Constraint(t, rule=Constraint_aux1)
    def Constraint_aux2(_b,_t):
        return _b.aux[_t] <= _b.Qoutp[_t]
    b.c_aux2 = pyo.Constraint(t, rule=Constraint_aux2)
    def Constraint_aux3(_b,_t):
        return _b.aux[_t] >= _b.Qoutp[_t] - _b.Qmax*(1-_b.ModePump[_t])
    b.c_aux3 = pyo.Constraint(t, rule=Constraint_aux3)
    
    # From AIMMS Ch 7.7: Elimination of products of variables
    def Constraint_auxa1(_b,_t):
        return _b.auxa[_t] <= _b.A*_b.ModePump[_t]
    b.c_auxa1 = pyo.Constraint(t, rule=Constraint_auxa1)
    def Constraint_auxa2(_b,_t):
        return _b.auxa[_t] <= _b.a[_t]
    b.c_auxa2 = pyo.Constraint(t, rule=Constraint_auxa2)
    def Constraint_auxa3(_b,_t):
        return _b.auxa[_t] >= _b.a[_t] - _b.A*(1-_b.ModePump[_t])
    b.c_auxa3 = pyo.Constraint(t, rule=Constraint_auxa3)
    
    def Constraint_Qout(_b, _t):
        return _b.Qout[_t] == _b.Qoutp[_t] - _b.Qoutt[_t]
    b.c_Qout = pyo.Constraint(t, rule=Constraint_Qout)

    def Constraint_Ph(_b, _t):
        return _b.Ph[_t] == 9810*_b.H[_t]*_b.Qout[_t]
    b.c_Ph = pyo.Constraint(t, rule=Constraint_Ph)

    def Constraint_Pe(_b, _t):
        return _b.Pe[_t] == _b.Ph[_t] *(_b.ModePump[_t]/_b.eff + (1-_b.ModePump[_t])*_b.eff_t)
    b.c_Pe = pyo.Constraint(t, rule=Constraint_Pe)
    
    # From AIMMS Ch 7.1: Variable taking discontinuous values
    def Constraint_Qmaxp(_b, _t): 
        return _b.Qoutp[_t] <= (_b.Qmax*_b.ModePump[_t])
    b.c_Qmaxp = pyo.Constraint(t, rule=Constraint_Qmaxp)
    def Constraint_Qminp(_b, _t):
        return _b.Qoutp[_t] >= (_b.Qmin*_b.ModePump[_t])
    b.c_Qminp = pyo.Constraint(t, rule=Constraint_Qminp)
    
    def Constraint_Qmaxt(_b, _t):
        return _b.Qoutt[_t] <= (_b.H[_t]*2*9.81)**0.5*_b.S*(1-_b.ModePump[_t])
    b.c_Qmaxt = pyo.Constraint(t, rule=Constraint_Qmaxt)
    

def DiscretePump(b, t, data, init_data):
    """
    DiscretePump is the association of "N" pumps in parallel. 
    Each individual pump have two state:
        1. ON consuming the nominal power. 
        2. OFF.
    The model allows for the activation of an integer number of pumps ranging from 0 to "N".
    The total power consumed by the system is directly proportional to the number of pumps that are ON.
    
    It's introduced :math:`P_{n}` wich is the individual nominal power of the pump and :math:`N_{pumps}` the maximum number of pumps that can be associated.

    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: data ``dict``
    :param init_data: init_data ``dict``

    data
         - 'Npumps': Maximum number of pumps to consider :math:`Npumps`
         - 'eff': Efficiency at the nominal operating point in p.u. :math:`\eta`
         - 'Pn': Nominal power of individual pump :math:`P_{n}`

    init_data
         - 'Q': Flow :math:`Q(t)` as a ``list``
         - 'H': Head :math:`H(t)` as a ``list``
         - 'Pe': Electrical power :math:`P_e(t)` as a ``list``

    Pyomo declarations 
        - Parameters:
            - eff
            - Pn
            - Npumps
        - Variables:
            - Qin (t) bounded :math:`Q_{in} \leq 0`
            - Qout (t) bounded :math:`Q_{out} \ge 0`
            - H (t) bounded :math:`H \ge 0`
            - Ph (t) bounded :math:`P_h \in [0, P_{max}]`
            - Pe (t) bounded :math:`P_e \in [0, P_{max}]`
            - N_on (t) :math:`b_{ON}(t) \in [0,N_{pumps}]`

        - Ports:
            - port_Qin @ Qin with 'Q' as ``Extensive``
            - port_Qout @ Qout with 'Q' as ``Extensive``
            - port_P @ Pe with 'P' as ``Extensive``
            - port_H @ H with 'H' as ``Equality``
        - Constraints:
            - c_Q: :math:`Q_{in}(t) = - Q_{out}(t)`
            - c_Ph: :math:`P_h(t) = 9810\, H(t)\, Q_{out}(t)`
            - c_Pe: :math:`P_e(t) = P_n \, N_{on}(t)`
            - c_eff: :math:`P_e(t) = P_h / \eta`
    """

    # Parameters
    b.eff = pyo.Param(initialize=data['eff'])
    b.Pn = pyo.Param( initialize=data['Pn'])
    b.Npumps = pyo.Param(initialize=data['Npumps'])

    # Variables
    b.Qin = pyo.Var(t, initialize=[-k for k in init_data['Q']], within=pyo.NonPositiveReals)
    b.Qout = pyo.Var(t, initialize=init_data['Q'], within=pyo.NonNegativeReals)
    b.H = pyo.Var(t, initialize=init_data['H'], within=pyo.NonNegativeReals)
    b.Ph = pyo.Var(t, initialize=init_data['Pe'],within=pyo.NonNegativeReals)
    b.Pe = pyo.Var(t, initialize=init_data['Pe'], within=pyo.NonNegativeReals)
    b.N_on = pyo.Var(t, initialize=0, within=pyo.Integers, bounds=(0, data['Npumps']))

    # Ports
    b.port_Qin = Port(initialize={'Q': (b.Qin, Port.Extensive)})
    b.port_Qout = Port(initialize={'Q': (b.Qout, Port.Extensive)})
    b.port_P = Port(initialize={'P': (b.Pe, Port.Extensive)})
    b.port_H = Port(initialize={'H': (b.H, Port.Equality)})


    #Constraints
    def Constraint_Q(_b, _t):
        return _b.Qin[_t] == -_b.Qout[_t]
    b.c_Q = pyo.Constraint(t, rule=Constraint_Q)
    
    def Constraint_Pe(_b, _t):
        return _b.Pe[_t] == _b.Pn * _b.N_on[_t]
    b.c_Pe = pyo.Constraint(t, rule = Constraint_Pe)
    
    def Constraint_Ph(_b, _t):
        return _b.Ph[_t] == 9810*_b.H[_t]*_b.Qout[_t]
    b.c_Ph = pyo.Constraint(t, rule=Constraint_Ph)
   
    def Constraint_eff(_b, _t):
        return _b.Pe[_t] == _b.Ph[_t]/_b.eff
    b.c_eff = pyo.Constraint(t, rule=Constraint_eff)
    
    
    
def RealPumpS(b, t, data, init_data):
    """
    In RealPump it's added the working flow limits of the pump.  
    Be wary of binary variables when choosing a solver.
    
    It's introduced :math:`Q_{max}` and :math:`Q_{min}` which determines the working limits.

    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: data ``dict``
    :param init_data: init_data ``dict``

    data
         - 'A': Constant characteristic coefficient of the pump :math:`A`
         - 'B': Quadratic characteristic coefficient of the pump :math:`B`
         - 'Qnom': Nominal flow :math:`Q_n`
         - 'eff': Efficiency at the nominal operating point in p.u. :math:`\eta`
         - 'Qmax': Maximum allowed flow (in p.u. of Qnom) :math:`Q_{max}`
         - 'Qmin': Minimum working flow (in p.u. of Qnom) :math:`Q_{min}`
         - 'Pmax': Maximum allowed power :math:`P_{max}`

    init_data
         - 'Q': Flow :math:`Q(t)` as a ``list``
         - 'H': Head :math:`H(t)` as a ``list``
         - 'n': Rotational speed (in p.u.) :math:`n(t)` as a ``list``
         - 'Pe': Electrical power :math:`P_e(t)` as a ``list``

    Pyomo declarations    
        - Parameters:
            - A
            - B
            - Qnom
            - eff
            - Qmin
            - Qmax
        - Variables:
            - Qin (t) bounded :math:`Q_{in} \in [-Q_{max}\,Q_{n}, 0]`
            - Qout (t) bounded :math:`Q_{out} \in [0, Q_{max}\,Q_{n}]`
            - H (t) bounded :math:`H \ge 0`
            - npu2 (t) bounded :math:`n \ge 0`
            - Ph (t) bounded :math:`P_h \in [0, P_{max}]`
            - Pe (t) bounded :math:`P_e \in [0, P_{max}]`
            - PumpOn (t) :math:`b_{ON}(t) \in \{0,1\}`

        - Ports:
            - port_Qin @ Qin with 'Q' as ``Extensive``
            - port_Qout @ Qout with 'Q' as ``Extensive``
            - port_P @ Pe with 'P' as ``Extensive``
            - port_H @ H with 'H' as ``Equality``
        - Constraints:
            - c_Q: :math:`Q_{in}(t) = - Q_{out}(t)`
            - c_H: :math:`H(t) = (n(t)/n_n)^2\, A - B \, Q_{out}(t)^2`
            - c_Ph: :math:`P_h(t) = 9810\, H(t)\, Q_{out}(t)`
            - c_Pe: :math:`P_e(t) = P_h(t)/\eta`
            - c_Qmax: :math:`Q_{out}(t) \leq Q_{n} \, Q_{max} \, b_{ON}(t)`
            - c_Qmin: :math:`Q_{out}(t) \geq Q_{n} \, Q_{min} \, b_{ON}(t)`
    """

    # Parameters
    b.A = pyo.Param(initialize=data['A'])
    b.B = pyo.Param(initialize=data['B'])
    b.Qnom = pyo.Param(initialize=data['Qnom'])
    b.eff = pyo.Param(initialize=data['eff'])
    b.Qmin = pyo.Param(initialize=data['Qmin']*data['Qnom'])
    b.Qmax = pyo.Param(initialize=data['Qmax']*data['Qnom'])

    # Variables
    b.Qin = pyo.Var(t, initialize=[-k for k in init_data['Q']], bounds=(-data['Qmax']*data['Qnom'], 0), within=pyo.NonPositiveReals)
    b.Qout = pyo.Var(t, initialize=init_data['Q'], bounds=(0, data['Qmax']*data['Qnom']), within=pyo.NonNegativeReals)
    b.H = pyo.Var(t, initialize=init_data['H'], within=pyo.NonNegativeReals)
    b.Ph = pyo.Var(t, initialize=[k*data['eff'] for k in init_data['Pe']], bounds=(0, data['Pmax']*data['eff']), within=pyo.NonNegativeReals)
    b.Pe = pyo.Var(t, initialize=init_data['Pe'], bounds=(0, data['Pmax']), within=pyo.NonNegativeReals)
    # b.PumpOn = pyo.Var(t, initialize=1, within=pyo.Binary)
    b.PumpOn = pyo.Var(t, initialize=1, bounds=(0,1), within=pyo.NonNegativeReals)
    b.npu2 = pyo.Var(t, initialize=init_data['n'], bounds=(0,1), within=pyo.NonNegativeReals)
    b.e = pyo.Var(t, initialize=0, within=pyo.NonNegativeReals)    

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
        return _b.H[_t] == _b.npu2[_t]*_b.A - _b.B*_b.Qout[_t]**2
    b.c_H = pyo.Constraint(t, rule=Constraint_H)

    def Constraint_Ph(_b, _t):
        return _b.Ph[_t] == 9810*_b.H[_t]*_b.Qout[_t]
    b.c_Ph = pyo.Constraint(t, rule=Constraint_Ph)

    def Constraint_Pe(_b, _t):
        return _b.Pe[_t] == _b.Ph[_t]/_b.eff
    b.c_Pe = pyo.Constraint(t, rule=Constraint_Pe)

    def Constraint_onoff(_b, _t):
        return _b.PumpOn[_t]*(1-_b.PumpOn[_t]) <= 1e-3
    b.c_onoff = pyo.Constraint(t, rule=Constraint_onoff)
    
    # def Constraint_onoff(_b, _t):
    #     return _b.PumpOn[_t]+(1-_b.PumpOn[_t]) == 1
    # b.c_onoff = pyo.Constraint(t, rule=Constraint_onoff)



    # def sigmoid(x,k,x0):
    #     return 1/(1+2.7182818284**(-k*(x-x0)))
    
    # def normal(x,mu,sig):
    #     return 2.7182818284**(-(x-mu)**2/(2*sig**2))
    
    # def Constraint_e(_b,_t):
    #     return _b.e[_t] == sigmoid(_b.Qout[_t],k=8000,x0=0.001) - sigmoid(_b.Qout[_t],k=8000,x0=_b.Qmin-0.001) + normal(_b.Qout[_t], mu=_b.Qmin/2, sig=_b.Qmin/2/4)
    # b.c_e = pyo.Constraint(t, rule=Constraint_e)

        
    
