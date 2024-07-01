# AGISTIN project 
# .\Devices\MainGrid.py

"""
MainGrid pyomo block contains characteristics of the point of connection.
"""


import pyomo.environ as pyo
from pyomo.network import Arc, Port


# data: Pmax
# init_data: None

def Grid(b, t, data, init_data=None):

    """
    Point of connection to the grid.
    
    Delivers or consumes the required power up to a set maximum.
    It also defines whether the power is consumed from the grid :math:`P_{buy}` or delivered to :math:`P_{sell}`.

    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: data ``dict``
    :param init_data: ``None``
        
    data
         - 'Pmax': Maximum allowed power :math:`P(t) \in (-P_{max}, P_{max})`
    
    Pyomo declarations    
        - Parameters: 
            - Pmax
        - Variables: 
            - P (t)
            - Psell (t)
            - Pbuy (t)
        - Ports: 
            - port_P @ P as ``Extensive``
        - Constraints: 
            - c_P: :math:`P(t) = P_{sell}(t) -  P_{buy}(t)`
            - c_P0: :math:`0 = P_{sell}(t) \cdot  P_{buy}(t)`
    """
    
    # Parameters
    b.Pmax = pyo.Param(initialize=data['Pmax'])

    
    # Variables
    b.P = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, bounds=(-b.Pmax, b.Pmax), domain=pyo.Reals)
    b.Psell = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, bounds=(0, b.Pmax), domain=pyo.NonNegativeReals)
    b.Pbuy = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, bounds=(0, b.Pmax), domain=pyo.NonNegativeReals)
    
    # Ports
    b.port_P = Port(initialize={'P': (b.P, Port.Extensive)})
    
    # Constraints
    def Constraint_P(_b, _t):
        return _b.P[_t] == _b.Psell[_t] - _b.Pbuy[_t]
    b.c_P = pyo.Constraint(t, rule=Constraint_P)
    
    def Constraint_P0(_b, _t):
        return 0 == _b.Psell[_t]*_b.Pbuy[_t]
    b.c_P0 = pyo.Constraint(t, rule=Constraint_P0)
    

def FlexibleGrid(b, t, data, init_data=None):

    """
    Point of connection to the grid.
    
    Delivers or consumes the required power up to a set maximum.
    It also defines whether the power is consumed from the grid :math:`P_{buy}` or delivered to :math:`P_{sell}`.
    This version

    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: data ``dict``
    :param init_data: ``None``
        
    data
         - 'Pmax': Maximum allowed power :math:`P(t) \in (-P_{max}, P_{max})`
    
    Pyomo declarations    
        - Parameters: 
            - Pmax
        - Variables: 
            - P (t)
            - Psell (t)
            - Pbuy (t)
        - Ports: 
            - port_P @ P as ``Extensive``
        - Constraints: 
            - c_P: :math:`P(t) = P_{sell}(t) -  P_{buy}(t)`
            - c_P0: :math:`0 = P_{sell}(t) \cdot  P_{buy}(t)`
    """
    
    # Parameters
    b.Pmax = pyo.Param(initialize=data['Pmax'])
    b.Pdiffmin = pyo.Param(t,initialize=data['Pdiffmin']) 
    b.Pdiffmax = pyo.Param(t, initialize=data['Pdiffmax'])
    
    # Variables
    b.P = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, bounds=(-b.Pmax, b.Pmax), domain=pyo.Reals)
    b.Psell = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, bounds=(0, b.Pmax), domain=pyo.NonNegativeReals)
    b.Pbuy = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, bounds=(0, b.Pmax), domain=pyo.NonNegativeReals)
    # Ports
    b.port_P = Port(initialize={'P': (b.P, Port.Extensive)})
    
    # Constraints
    def Constraint_P(_b, _t):
        return _b.P[_t] == _b.Psell[_t] - _b.Pbuy[_t]
    b.c_P = pyo.Constraint(t, rule=Constraint_P)
    
    def Constraint_P0(_b, _t):
        return 0 == _b.Psell[_t]*_b.Pbuy[_t]
    b.c_P0 = pyo.Constraint(t, rule=Constraint_P0)
    
    def Constraint_dPmax(_b, _t):
        return _b.Pbuy[_t] <= _b.Pdiffmax[_t]
    b.c_dPmax = pyo.Constraint(t, rule = Constraint_dPmax)
    
    def Constraint_dPmin(_b, _t):
        return _b.Pbuy[_t] >=_b.Pdiffmin[_t]
    b.c_dPmin = pyo.Constraint(t, rule = Constraint_dPmin)
    
#_____________________________________________________________________________________________________________________    
def Grid_task_3_2 (b, t, data, init_data=None):

        """
        Point of connection to the grid.
        
        Delivers or consumes the required power up to a set maximum.
        It also defines whether the power is consumed from the grid :math:`P_{buy}` or delivered to :math:`P_{sell}`.

        :param b: pyomo ``Block()`` to be set
        :param t: pyomo ``Set()`` referring to time
        :param data: data ``dict``
        :param init_data: ``None``
            
        data
             - 'Pmax': Maximum allowed power :math:`P(t) \in (-P_{max}, P_{max})`
        
        Pyomo declarations    
            - Parameters: 
                - Pmax
            - Variables: 
                - P (t)
                - Psell (t)
                - Pbuy (t)
            - Ports: 
                - port_P @ P as ``Extensive``
            - Constraints: 
                - c_P: :math:`P(t) = P_{sell}(t) -  P_{buy}(t)`
                - c_P0: :math:`0 = P_{sell}(t) \cdot  P_{buy}(t)`
        """
        
        # Parameters
        b.Pmax = pyo.Param(initialize=data['Pmax'])
        b.slope_fcr = pyo.Param(initialize=data['slope_fcr'])  # statik for the FCR ( 100%  power by a 200mH deviation)

        
        # Variables
        b.P = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, bounds=(-b.Pmax, b.Pmax), domain=pyo.Reals)
        b.Psell = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, bounds=(0, b.Pmax), domain=pyo.NonNegativeReals)
        b.Pbuy = pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, bounds=(0, b.Pmax), domain=pyo.NonNegativeReals)
        #b.PCostGrid= pyo.Var(t, initialize={k:0.0 for k in range(len(t))}, domain=pyo.NonNegativeReals) # cost of electricity from grid
        b.CostPbuy = pyo.Var(t, initialize={k: 0.0 for k in range(len(t))}, domain=pyo.Reals)  # cost of electricity from grid
        costElectricityFromGrid[t] = (m.Grid.Pbuy[t] * m.cost_MainGrid[t])
        # Ports
        b.port_P = Port(initialize={'P': (b.P, Port.Extensive)})
        
        # Constraints
        def Constraint_P(_b, _t):
            return _b.P[_t] == _b.Psell[_t] - _b.Pbuy[_t]
        b.c_P = pyo.Constraint(t, rule=Constraint_P)
        
        def Constraint_P0(_b, _t):
            return 0 == _b.Psell[_t]*_b.Pbuy[_t]
        b.c_P0 = pyo.Constraint(t, rule=Constraint_P0)
    