# AGISTIN project 
# .\Devices\Pipes.py

"""
Pipe pyomo block contains characteristics of a pipe.
"""


import pyomo.environ as pyo
from pyomo.network import Arc, Port
from pyomo.core import Piecewise
   
    
def Pipe(b, t, data, init_data):

    """
    Pipe that transports water and has implicit energy losses.
    
    Acts as a flow transportation device which also applies an energy loss to the fluid as:
    
    .. math::
        H(t) = H_0(t) \pm K \, Q(t)^2
    
    where :math:`H_0(t)` is the static height and is computed as the difference of heights between both pipe extremes.
    
    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: data ``dict``
    :param init_data: init_data ``dict``
        
    data
         - 'K': Linear pressure loss coefficient of the pipe :math:`K`
         - 'Qmax': Maximum allowed flow :math:`Q_{max}`
         - 'zlow_bounds': zmin and zmax of the lower reservoir as a ``tuple`` (optional)
         - 'zhigh_bounds': zmin and zmax of the upper reservoir as a ``tuple`` (optional)
         - 'H_approx': Approximation for height between reservoirs :math:`\hat{H}` (optional)
         
    init_data
         - 'Q': Flow :math:`Q(t)` as a ``list`` or pandas ``Series``
         - 'H': Head :math:`H(t)` as a ``list`` or pandas ``Series``
         - 'zlow': Lower height :math:`z_{low}(t)` as a ``list`` or pandas ``Series``
         - 'zhigh': Higher height :math:`z_{high}(t)` as a ``list`` or pandas ``Series``
    
    Pyomo declarations    
        - Parameters: 
            - K
            - Qmax
            
        - Variables: 
            - Q (t) bounded :math:`\in [-Q_{max}, Q_{max}]`
            - H (t) bounded :math:`H \ge 0`
            - zlow (t) bounded :math:`z_{low} \ge 0`
            - zhigh (t) bounded :math:`z_{high} \ge 0`
            - Qp(t) bounded :math:`Q_p \in [0, Q_{max}]`
            - Qn(t) bounded :math:`Q_n \in [0, Q_{max}]`
    
        - Ports: 
            - port_Q @ Q as ``Extensive``
            - port_H @ H as ``Equality``
            - port_zlow @ zlow as ``Equality``
            - port_zhigh @ zhigh as ``Equality``
            
        - Constraints: 
            - c_H: :math:`H(t) = z_{high}(t)-z_{low}(t) + K (Q_p(t)^2 - Q_n(t)^2)`
            - c_Q: :math:`Q(t)= Qp(t) - Qn(t)`
            - c_QpQn0: :math:`0 = Qp(t)\,Qn(t)
            
    """
    
    # Parameters
    b.K = pyo.Param(initialize=data['K'])
    b.Qmax = pyo.Param(initialize=data['Qmax'])
    
    
    zlow_bounds = (0, None)
    zhigh_bounds = (0, None)
    if 'zlow_bounds' in data:
        zlow_bounds = data['zlow_bounds']
    if 'zhigh_bounds' in data:
        zhigh_bounds = data['zhigh_bounds']
    
    # Variables
    b.Q = pyo.Var(t, initialize=init_data['Q'], bounds=(-data['Qmax'], data['Qmax']), within=pyo.Reals)
    b.H = pyo.Var(t, initialize=init_data['H'], within=pyo.NonNegativeReals) 
    b.zlow = pyo.Var(t, initialize=init_data['zlow'], bounds=zlow_bounds, within=pyo.NonNegativeReals) 
    b.zhigh = pyo.Var(t, initialize=init_data['zhigh'], bounds=zhigh_bounds, within=pyo.NonNegativeReals) 
    # b.signQ = pyo.Var(t, initialize=1, within=pyo.Binary)
    b.Qp = pyo.Var(t, initialize=init_data['Q'], bounds=(0, data['Qmax']),within=pyo.NonNegativeReals)
    b.Qn = pyo.Var(t, initialize=0, bounds=(0, data['Qmax']),within=pyo.NonNegativeReals)
    
    # Ports
    b.port_Q = Port(initialize={'Q': (b.Q, Port.Extensive)})
    b.port_H = Port(initialize={'H': (b.H, Port.Equality)})
    b.port_zlow = Port(initialize={'z': (b.zlow, Port.Equality)})
    b.port_zhigh = Port(initialize={'z': (b.zhigh, Port.Equality)})
    
    # Constraints
    if('H_approx' in data):
        def Constraint_H(_b, _t):
            return _b.H[_t] == data['H_approx'] + _b.K*(_b.Qp[_t]**2 -_b.Qn[_t]**2)
    else:
        def Constraint_H(_b, _t):
            return _b.H[_t] ==  _b.zhigh[_t] - _b.zlow[_t] + _b.K*(_b.Qp[_t]**2 -_b.Qn[_t]**2)
    b.c_H = pyo.Constraint(t, rule = Constraint_H)
    
    def Constraint_Q(_b, _t):
        return _b.Q[_t] == _b.Qp[_t] - _b.Qn[_t]
    b.c_Q = pyo.Constraint(t, rule = Constraint_Q)
    
    def Constraint_QpQn0(_b,_t):
        return _b.Qp[_t] * _b.Qn[_t] == 0
    b.c_QpQn0 = pyo.Constraint(t, rule = Constraint_QpQn0)

    # def Constraint_sign(_b,_t):
    #     # return _b.signQ[_t] == _b.Q[_t]/((_b.Q[_t] )**2+1e-6)**0.5
    #     # return _b.signQ[_t] == 2 * ( 1/(1+2.718281828**(-5*_b.Q[_t])) - 0.5)
    #     return _b.Q[_t]*(2*_b.signQ[_t]-1) >= 0
    # b.c_sign = pyo.Constraint(t, rule = Constraint_sign)


def PipeValve(b, t, data, init_data):

    """
    Pipe that transports water and has implicit energy losses.
    
    Acts as a flow transportation device which also applies an energy loss to the fluid and considers the aperture :math:`a(t)` of a valve as:
    
    .. math::
        H(t) = H_0(t) \pm (K + C_v/a(t) )\, Q(t)^2
    
    where :math:`H_0(t)` is the static height and is computed as the difference of heights between both pipe extremes.
    
    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: data ``dict``
    :param init_data: init_data ``dict``
        
    data
         - 'K': Linear pressure loss coefficient of the pipe :math:`K`
         - 'Cv': Linear pressure loss coefficient of the valve :math:`C_{v}`
         - 'Qmax': Maximum allowed flow :math:`Q_{max}`
         
    init_data
         - 'Q': Flow :math:`Q(t)` as a ``list`` or pandas ``Series``
         - 'H': Head :math:`H(t)` as a ``list`` or pandas ``Series``
         - 'zlow': Lower height :math:`z_{low}(t)` as a ``list`` or pandas ``Series``
         - 'zhigh': Higher height :math:`z_{high}(t)` as a ``list`` or pandas ``Series``
         - 'alpha': Valve aperture :math:`a(t)` as a ``list`` or pandas ``Series``
    
    Pyomo declarations    
        - Parameters: 
            - K
            - Cv
            - Qmax
            
        - Variables: 
            - Q (t) bounded :math:`\in [-Q_{max}, Q_{max}]`
            - H (t) bounded :math:`H \ge 0`
            - zlow (t) bounded :math:`z_{low} \ge 0`
            - zhigh (t) bounded :math:`z_{high} \ge 0`
            - Qp (t) bounded :math:`\in [0, Q_{max}]`
            - Qn (t) bounded :math:`\in [0, Q_{max}]`
    
        - Ports: 
            - port_Q @ Q as ``Extensive``
            - port_H @ H as ``Equality``
            - port_zlow @ zlow as ``Equality``
            - port_zhigh @ zhigh as ``Equality``
            
        - Constraints: 
            - c_H: :math:`H(t) = z_{high}(t)-z_{low}(t) + (K+C_v/a(t)) (Q_p(t)^2 - Q_n(t)^2)`
            - c_Q: :math:`Q(t)= Qp(t) - Qn(t)`
            - c_QpQn0: :math:`0 = Qp(t)\,Qn(t)`          
    """
    
    # Parameters
    b.K = pyo.Param(initialize=data['K'])
    b.Cv = pyo.Param(initialize=data['Cv'])
    b.Qmax = pyo.Param(initialize=data['Qmax'])

    # Variables
    b.Q = pyo.Var(t, initialize=init_data['Q'], bounds=(-data['Qmax'], data['Qmax']), within=pyo.Reals)
    b.H = pyo.Var(t, initialize=init_data['H'], within=pyo.NonNegativeReals) 
    b.zlow = pyo.Var(t, initialize=init_data['zlow'], within=pyo.NonNegativeReals) 
    b.zhigh = pyo.Var(t, initialize=init_data['zhigh'], within=pyo.NonNegativeReals)
    # b.signQ = pyo.Var(t, initialize=1, bounds=(-1,1), within=pyo.Reals)
    # b.signQ = pyo.Var(t, initialize=1, within=pyo.Binary)
    b.alpha = pyo.Var(t, initialize=init_data['alpha'], bounds=(1e-6, 1), within=pyo.NonNegativeReals)
    b.Qp = pyo.Var(t, initialize=init_data['Q'], bounds=(0, data['Qmax']),within=pyo.NonNegativeReals)
    b.Qn = pyo.Var(t, initialize=0, bounds=(0, data['Qmax']),within=pyo.NonNegativeReals)
    
    
    # Ports
    b.port_Q = Port(initialize={'Q': (b.Q, Port.Extensive)})
    b.port_H = Port(initialize={'H': (b.H, Port.Equality)})
    b.port_zlow = Port(initialize={'z': (b.zlow, Port.Equality)})
    b.port_zhigh = Port(initialize={'z': (b.zhigh, Port.Equality)})
    
    # Constraints
    # def Constraint_H(_b, _t):
    #     return _b.H[_t] == _b.H0[_t] + _b.K*_b.Q[_t]**2*(2*_b.signQ[_t]-1)
    # b.c_H = pyo.Constraint(t, rule = Constraint_H)

    # def Constraint_sign(_b,_t):
    #     # return _b.signQ[_t] == _b.Q[_t]/((_b.Q[_t] )**2+1e-6)**0.5
    #     # return _b.signQ[_t] == 2 * ( 1/(1+2.718281828**(-5*_b.Q[_t])) - 0.5)
    #     return _b.Q[_t]*(2*_b.signQ[_t]-1) >= 0
    # b.c_sign = pyo.Constraint(t, rule = Constraint_sign)
    
    def Constraint_QpQn0(_b,_t):
        return _b.Qp[_t] * _b.Qn[_t] == 0
    b.c_QpQn0 = pyo.Constraint(t, rule = Constraint_QpQn0)
    
    def Constraint_Q(_b,_t):
        return _b.Q[_t] == b.Qp[_t] - b.Qn[_t]
    b.c_Q = pyo.Constraint(t, rule = Constraint_Q)
    
    def Constraint_valve(_b,_t):
        return _b.H[_t] == _b.zhigh[_t]-_b.zlow[_t] +(_b.K+_b.Cv/(_b.alpha[_t]))*_b.Qp[_t]**2 - (_b.K+_b.Cv/(_b.alpha[_t]))*_b.Qn[_t]**2
    b.c_H = pyo.Constraint(t, rule = Constraint_valve)


def Pipe_Ex0(b, t, data, init_data):
    
    """
    Simple Pipe for testing and example purposes.
    It is utilised in Example0.
    
    Acts as a flow transportation device which also applies an energy loss to the fluid as:
    
    .. math::
        H(t) = H_0 \pm K \cdot Q(t)^2
    
    where :math:`H_0` is the static height, which is considered constant as defined by the user.
    
    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: data ``dict``
    :param init_data: init_data ``dict``
        
    data
         - 'H0': Static height :math:`H_0`
         - 'K': Linear pressure loss coefficient of the pipe :math:`K`
         - 'Qmax': Maximum allowed flow :math:`Q_{max}`
         
    init_data
         - 'Q': Flow :math:`Q(t)` as a ``list`` or pandas ``Series``
         - 'H': Head :math:`H(t)` as a ``list`` or pandas ``Series``
    
    Pyomo declarations    
        - Parameters: 
            - H0 
            - K
            - Qmax
            
        - Variables: 
            - Q (t) bounded :math:`\in [-Q_{max}, Q_{max}]`
            - H (t) bounded :math:`H \ge 0`
            - Qp (t) bounded :math:`\in [0, Q_{max}]`
            - Qn (t) bounded :math:`\in [0, Q_{max}]`
            
        - Ports: 
            - port_Q @ Q as ``Extensive``
            - port_H @ H as ``Equality``
            
        - Constraints: 
            - c_H: :math:`H(t) = H_0 + K(Q_p(t)^2 - Q_n(t)^2)`
            - c_Q: :math:`Q(t)= Qp(t) - Qn(t)`
            - c_QpQn0: :math:`0 = Qp(t)\,Qn(t)`
            
    """
    
    # Parameters
    b.H0 = pyo.Param(initialize=data['H0'])
    b.K = pyo.Param(initialize=data['K'])
    b.Qmax = pyo.Param(initialize=data['Qmax'])
    
    # Variables
    b.Q = pyo.Var(t, initialize=init_data['Q'], bounds=(-data['Qmax'], data['Qmax']), within=pyo.Reals)
    b.H = pyo.Var(t, initialize=init_data['H'], within=pyo.NonNegativeReals)
    b.Qp = pyo.Var(t, initialize=init_data['Q'], bounds=(0, data['Qmax']),within=pyo.NonNegativeReals)
    b.Qn = pyo.Var(t, initialize=0, bounds=(0, data['Qmax']),within=pyo.NonNegativeReals)
    # b.signQ = pyo.Var(t, initialize=1, bounds=(-1,1), within=pyo.Reals)
    # b.signQ = pyo.Var(t, initialize=1, within=pyo.Binary)
    
    # Ports
    b.port_Q = Port(initialize={'Q': (b.Q, Port.Extensive)})
    b.port_H = Port(initialize={'H': (b.H, Port.Equality)})
    
    # Constraints
    def Constraint_H(_b, _t):
        #return _b.H[_t] == _b.H0 + _b.K*_b.Q[_t]**2*(2*_b.signQ[_t]-1)#_b.signQ[_t]
        return _b.H[_t] == _b.H0 + _b.K*(_b.Qp[_t]**2 -_b.Qn[_t]**2)    
    b.c_H = pyo.Constraint(t, rule = Constraint_H)
    
    def Constraint_QpQn0(_b,_t):
        return _b.Qp[_t] * _b.Qn[_t] == 0
    b.c_QpQn0 = pyo.Constraint(t, rule = Constraint_QpQn0)
    
    def Constraint_Q(_b,_t):
        return _b.Q[_t] == b.Qp[_t] - b.Qn[_t]
    b.c_Q = pyo.Constraint(t, rule = Constraint_Q)
    
#    def Constraint_sign(_b,_t):
#        # return _b.signQ[_t] == _b.Q[_t]/((_b.Q[_t] )**2+1e-6)**0.5
#        # return _b.signQ[_t] == 2 * ( 1/(1+pyo.exp(-5*_b.Q[_t])) - 0.5)
#        return _b.Q[_t]*(2*_b.signQ[_t]-1) >= 0
#    b.c_sign = pyo.Constraint(t, rule = Constraint_sign)
    