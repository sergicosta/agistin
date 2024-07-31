# AGISTIN project 
# .\Devices\Reservoirs.py
"""
Reservoir pyomo block containing the characteristics of a reservoir.
"""

import pyomo.environ as pyo
from pyomo.network import Arc, Port


# data: dt, W0, Wmin, Wmax, zmin, zmax
# init_data: Q(t), W(t)

def Reservoir(b, t, data, init_data):
    
    """
    Simple Reservoir.
    
    Modifies its volume state :math:`W(t)` and height :math:`z(t)` from an initial state :math:`W_0` according to 
    the aggregated input flows :math:`Q(t)`
    
    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: data ``dict``
    :param init_data: init_data ``dict``
    
    data
         - 'dt': Time delta :math:`\Delta t`
         - 'W0': Initial volume :math:`W_0`
         - 'Wmin': Minimum allowed volume :math:`W_{min}`
         - 'Wmax': Maximum allowed volume :math:`W_{max}`
         - 'zmin': Height at minimum volume :math:`z_{min}`
         - 'zmax': Height at maximum volume :math:`z_{max}`
    
    init_data
         - 'Q': Flow :math:`Q(t)` as a ``list``
         - 'W': Volume :math:`W(t)` as a ``list``
    
    Pyomo declarations    
        - Parameters: 
            - W0
            - Wmin
            - Wmax
            - zmin
            - zmax
        - Variables: 
            - Q (t) :math:`\in \mathbb{R}`
            - W (t) bounded :math:`W \in [W_{min},W_{max}]`
            - z (t) bounded :math:`z \in [z_{min},z_{max}]`
        - Ports: 
            - port_Q @ Q as ``Extensive``
            - port_z @ z as ``Equality``
        - Constraints:
            - c_W: 
                - :math:`W(t) = W(t-1) + \Delta t \, Q(t) \quad` if  :math:`t>0`
                - :math:`W(t) = W_0 + \Delta t \, Q(t) \quad` otherwise
            - c_z: :math:`z(t) = (W(t) - W_{min})/(W_{max} - W_{min})\,(z_{max}-z_{min}) + z_{min}`
    """
    
    def z(_b, w):
        return (w-_b.Wmin)/(_b.Wmax-_b.Wmin)*(_b.zmax-_b.zmin) + _b.zmin
    
    b.dt = data['dt']
    
    # Parameters
    b.W0 = pyo.Param(initialize=data['W0'])
    b.Wmin = pyo.Param(initialize=data['Wmin'])
    b.Wmax = pyo.Param(initialize=data['Wmax'])
    b.zmin = pyo.Param(initialize=data['zmin'])
    b.zmax = pyo.Param(initialize=data['zmax'])
    b.WTmin = pyo.Param(initialize=data['WT_min'])
    b.WTmax = pyo.Param(initialize=data['WT_max'])
    
    # Variables
    b.Q = pyo.Var(t, initialize=init_data['Q'], within=pyo.Reals)
    b.W = pyo.Var(t, initialize=init_data['W'], bounds=(data['Wmin'], data['Wmax']), within=pyo.NonNegativeReals)
    b.z = pyo.Var(t, initialize={k: z(b,init_data['W'][k]) for k in range(len(t))}, bounds=(data['zmin'], data['zmax']), within=pyo.NonNegativeReals) 
    
    if ('WT_min' in data) and ('WT_max' in data):
        b.W[len(t)-1].bounds=(data['WT_min'], data['WT_max'])
    
    # Ports
    b.port_Q = Port(initialize={'Q': (b.Q, Port.Extensive)})
    b.port_z = Port(initialize={'z': (b.z, Port.Equality)})

    # Constraints
    def Constraint_W(_b, _t):
        if _t>0:
            return _b.W[_t] == _b.W[_t-1] + _b.dt*(_b.Q[_t])
        else:
            return _b.W[_t] == _b.W0 + _b.dt*(_b.Q[_t])
    b.c_W = pyo.Constraint(t, rule = Constraint_W)
    
    def Constraint_z(_b, _t):
        return _b.z[_t] == z(_b, _b.W[_t])
    b.c_z = pyo.Constraint(t, rule=Constraint_z)
    
    

# data: W0, Wmin, Wmax
# init_data: Q(t), W(t)
    
def Reservoir_Ex0(b, t, data, init_data):
    
    """
    Simple Reservoir for testing and example purposes.
    It is utilised in Example0.
    
    Modifies its volume state :math:`W(t)` from an initial state :math:`W_0` according to 
    the aggregated input flows :math:`Q(t)`
    
    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: data ``dict``
    :param init_data: init_data ``dict``
    
    data
         - 'W0': Initial volume :math:`W_0`
         - 'Wmin': Minimum allowed volume :math:`W_{min}`
         - 'Wmax': Maximum allowed volume :math:`W_{max}`
         
    init_data
         - 'Q': Flow :math:`Q(t)` as a ``list``
         - 'W': Volume :math:`W(t)` as a ``list``
    
    Pyomo declarations    
        - Parameters: 
            - W0
        - Variables: 
            - Q (t) :math:`\in \mathbb{R}`
            - W (t) bounded :math:`W \in [W_{min},W_{max}]`
        - Ports: 
            - port_Q @ Q (Extensive)
        - Constraints:
            - c_W: 
                - :math:`W(t) = W(t-1) + Q(t) \quad` if  :math:`t>0`
                - :math:`W(t) = W_0 + Q(t) \quad` otherwise
    """
    
    # Parameters
    b.W0 = pyo.Param(initialize=data['W0'])
    
    # Variables
    b.Q = pyo.Var(t, initialize=init_data['Q'], within=pyo.Reals)
    b.W = pyo.Var(t, initialize=init_data['W'], bounds=(data['Wmin'], data['Wmax']), within=pyo.NonNegativeReals)

    # Ports
    b.port_Q = Port(initialize={'Q': (b.Q, Port.Extensive)})

    # Constraints
    def Constraint_W(_b, _t):
        if _t>0:
            return _b.W[_t] == _b.W[_t-1] + (_b.Q[_t])
        else:
            return _b.W[_t] == _b.W0 + (_b.Q[_t])
    b.c_W = pyo.Constraint(t, rule = Constraint_W)