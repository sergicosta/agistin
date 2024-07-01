

"""
Inverter pyomo block contains characteristics of an inverter.
"""


import pyomo.environ as pyo

from pyomo.network import Arc, Port
from pyomo.core import *


def Inverter(b, t, data, init_data=None):

    """
    :param b: pyomo ``Block()`` to be set
    :param t: pyomo ``Set()`` referring to time
    :param data: data ``dict``
    :param init_data: init_data ``dict``

    Pyomo declarations
    - Parameters:
        - P_max
    - Variables:
        - P_in
        - P_out
    - Ports:
        - port_Pin @ Pin with 'P' as ``Extensive``
        - port_Pout @ Pout with 'P' as ``Extensive``
    - Constraints:
        -
    """
    # Create a Concrete Model

    # Parameters
    b.Pmax = pyo.Param(initialize=data['Pmax'])

    # Variables

    b.Pin  = pyo.Var(t, within=pyo.Reals, bounds=(-data['Pmax'], data['Pmax']))
    b.Pout = pyo.Var(t, within=pyo.Reals, bounds=(-data['Pmax'], data['Pmax']))
    b.P = pyo.Var(t, within=pyo.Reals, bounds=(-data['Pmax'], data['Pmax']))
    b.Pcondloss = pyo.Var(t, within=pyo.Reals)
    b.Pswitchloss = pyo.Var(t, within=pyo.Reals)
    b.P_losses = pyo.Var(t, within=pyo.Reals)


    # Ports
    b.port_Pin = Port(initialize={'P': (b.Pin, Port.Extensive)})
    b.port_Pout = Port(initialize={'P': (b.Pout, Port.Extensive)})
    #b.port_P = Port(initialize={'P': (b.P, Port.Extensive)})
    # Stützpunkte und Pcondloss-Werte für piecewise linear Funktion
    #xdata = # Stützpunkte für Pin
    #ydata = # Pcondloss-Werte entsprechend den Stützpunkte



    #Variables for Piecewise Function
    b.Y_cond = pyo.Var(t, bounds=(0, 5e3))
    b.Y_switch = pyo.Var(t, bounds=(0, 5e3))

    # Piecewise Linear Function for modelling conduction losses
    b.cond_loss = Piecewise(t, b.Y_cond, b.Pin,
                          pw_pts=data['x_cond_loss'],
                          pw_constr_type='EQ',
                          f_rule=data['y_cond_loss'],
                          pw_repn='SOS2')

    # Piecewise Linear Function for modelling switching losses
    b.switch_loss = Piecewise(t, b.Y_switch, b.Pin,
                          pw_pts=data['x_switch_loss'],
                          pw_constr_type='EQ',
                          f_rule=data['y_switch_loss'],
                          pw_repn='SOS2')

    # Constraints
    #def Constraint_P_balance(_b, _t):
    #    return _b.Pin[_t] + _b.Pout[_t] + abs(_b.P_losses[_t]) == 0

    #b.c_Pin = pyo.Constraint(t, rule=Constraint_P_balance)

    def Constraint_P_losses(_b, _t):
        return _b.P_losses[_t] == _b.Y_cond[_t] + _b.Y_switch[_t]

    b.c_P_loss = pyo.Constraint(t, rule=Constraint_P_losses)


    def Constraint_P(_b, _t):
        return _b.Pout[_t] + _b.Pin[_t] + _b.Y_cond[_t] + _b.Y_switch[_t] == 0

    b.c_P = pyo.Constraint(t, rule=Constraint_P)



    #def Constraint_P(_b, _t):
    #    return _b.P[_t]  ==  _b.P_losses[_t] #(_b.Y_cond[_t] + _b.Y_switch[_t]) == 0

    #b.c_P = pyo.Constraint(t, rule=Constraint_P)



