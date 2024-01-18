# AGISTIN project 
# .\Devices\Pipes.py

"""
Pipe pyomo block contains characteristics of a pipe.
"""


import pyomo.environ as pyo
from pyomo.network import Arc, Port
from pyomo.core import Piecewise


# data: K, Qmax
# init_data: H0(t), Q(t), H(t)

def Pipe(b, t, data, init_data):

    """
    Simple Pipe for testing and example purposes.
    It is utilised in Example0.
    
    Acts as a flow transportation device which also applies an energy loss to the fluid as:
    
    .. math::
        H(t) = H_0(t) + K \cdot Q(t)^2
    
    where :math:`H_0(t)` is the static height and is computed as the difference of heights between both pipe extremes.
    
    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: data ``dict``
    :param init_data: init_data ``dict``
        
    data
         - 'K': Linear pressure loss coefficient of the pipe :math:`K`
         - 'Qmax': Maximum allowed flow :math:`Q_{max}`
         
    init_data
         - 'H0': Static height :math:`H_0` as a ``list``
         - 'Q': Flow :math:`Q(t)` as a ``list``
         - 'H': Head :math:`H(t)` as a ``list``
    
    Pyomo declarations    
        - Parameters: 
            - K
        - Variables: 
            - Q (t)
            - H (t)
            - zlow (t)
            - zhigh (t)
            - H0 (t)
            - signQ (t)
        - Ports: 
            - port_Q @ Q as ``Extensive``
            - port_H @ H as ``Equality``
            - port_zlow @ zlow as ``Equality``
            - port_zhigh @ zhigh as ``Equality``
        - Constraints: 
            - c_H: :math:`H(t) = H_0(t) + K \cdot Q(t)^2`
            - c_H0: :math:`H_0(t) = z_{high}(t) - z_{low}(t)`
            - c_sign: :math:`signQ(t) = Q / (\abs(Q))`
    """
    
    # Parameters
    b.K = pyo.Param(initialize=data['K'])
    
    # Variables
    b.Q = pyo.Var(t, initialize=init_data['Q'], bounds=(-data['Qmax'], data['Qmax']), within=pyo.Reals)
    b.H = pyo.Var(t, initialize=init_data['H'], within=pyo.NonNegativeReals) 
    b.zlow = pyo.Var(t, initialize=init_data['zlow'], within=pyo.NonNegativeReals) 
    b.zhigh = pyo.Var(t, initialize=init_data['zhigh'], within=pyo.NonNegativeReals) 
    b.H0 = pyo.Var(t, initialize=init_data['H0'], within=pyo.NonNegativeReals)
    b.signQ = pyo.Var(t, initialize=1, bounds=(-1,1), within=pyo.Reals)
    
    # Ports
    b.port_Q = Port(initialize={'Q': (b.Q, Port.Extensive)})
    b.port_H = Port(initialize={'H': (b.H, Port.Equality)})
    b.port_zlow = Port(initialize={'z': (b.zlow, Port.Equality)})
    b.port_zhigh = Port(initialize={'z': (b.zhigh, Port.Equality)})
    
    # Constraints
    def Constraint_H(_b, _t):
        return _b.H[_t] == _b.H0[_t] + _b.K*_b.Q[_t]**2
    b.c_H = pyo.Constraint(t, rule = Constraint_H)
    
    def Constraint_H0(_b, _t):
        return _b.H0[_t] == _b.zhigh[_t] - _b.zlow[_t]
    b.c_H0 = pyo.Constraint(t, rule = Constraint_H0)

    def Constraint_sign(_b,_t):
        return _b.signQ[_t] == _b.Q[_t]/((_b.Q[_t] )**2+1e-6)**0.5
        # return _b.signQ[_t] == 2 * ( 1/(1+2.718281828**(-5*_b.Q[_t])) - 0.5)
    b.c_sign = pyo.Constraint(t, rule = Constraint_sign)


# data: H0, K, Qmax
# init_data: Q(t), H(t)

def Pipe_Ex0(b, t, data, init_data):
    
    """
    Simple Pipe for testing and example purposes.
    It is utilised in Example0.
    
    Acts as a flow transportation device which also applies an energy loss to the fluid as:
    
    .. math::
        H(t) = H_0 + K \cdot Q(t)^2
    
    where :math:`H_0` is the static height and is constant and defined by the user.
    
    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: data ``dict``
    :param init_data: init_data ``dict``
        
    data
         - 'H0': Static height :math:`H_0`
         - 'K': Linear pressure loss coefficient of the pipe :math:`K`
         - 'Qmax': Maximum allowed flow :math:`Q_{max}`
         
    init_data
         - 'Q': Flow :math:`Q(t)` as a ``list``
         - 'H': Head :math:`H(t)` as a ``list``
    
    Pyomo declarations    
        - Parameters: 
            - H0 
            - K
        - Variables: 
            - Q (t)
            - H (t)
            - signQ (t)
        - Ports: 
            - port_Q @ Q as ``Extensive``
            - port_H @ H as ``Equality``
        - Constraints: 
            - c_H: :math:`H(t) = H_0 \pm K \cdot Q(t)^2`
            - c_sign: :math:`signQ(t) = Q / (\abs(Q))`
    """
    
    # Parameters
    b.H0 = pyo.Param(initialize=data['H0'])
    b.K = pyo.Param(initialize=data['K'])
    
    # Variables
    b.Q = pyo.Var(t, initialize=init_data['Q'], bounds=(-data['Qmax'], data['Qmax']), within=pyo.Reals)
    b.H = pyo.Var(t, initialize=init_data['H'], within=pyo.NonNegativeReals)
    b.signQ = pyo.Var(t, initialize=1, bounds=(-1,1), within=pyo.Reals)
    
    # Ports
    b.port_Q = Port(initialize={'Q': (b.Q, Port.Extensive)})
    b.port_H = Port(initialize={'H': (b.H, Port.Equality)})
    
    # Constraints
    def Constraint_H(_b, _t):
        return _b.H[_t] == _b.H0 + _b.K*_b.Q[_t]**2*_b.signQ[_t]
    b.c_H = pyo.Constraint(t, rule = Constraint_H)
    
    def Constraint_sign(_b,_t):
        return _b.signQ[_t] == _b.Q[_t]/((_b.Q[_t] )**2+1e-6)**0.5
        # return _b.signQ[_t] == 2 * ( 1/(1+pyo.exp(-5*_b.Q[_t])) - 0.5)
    b.c_sign = pyo.Constraint(t, rule = Constraint_sign)
    
    # Q_PTS = [-data['Qmax'],-1e-6,1e-6,data['Qmax']]
    # Sign_PTS = [-1,-1,1,1]
    # b.c_sign = Piecewise(
    #     t,
    #     b.signQ,  # variable
    #     b.Q,  # range and domain variables
    #     pw_pts=Q_PTS,    # Domain points
    #     pw_constr_type='EQ',  # 'EQ' - Q variable is equal to the piecewise function
    #     f_rule=Sign_PTS,     # Range points
    #     pw_repn='SOS2',        # + 'DCC' - Disaggregated convex combination model.
    #     unbounded_domain_var=True)  # Indicates the type of piecewise representation to use
    