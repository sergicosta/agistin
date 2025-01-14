Devices
===============

| The following are the devices currently implemented in the tool.


| Devices are defined as python functions which apply Pyomo ``Param``, ``Var``, ``Port`` and ``Constraint`` to a ``Block``.
| These functions are structured as follows:

.. testcode::

	def Name(b, [t], data, init_data):
		
		# Parameters. Values set by `data`
		
		# Variables. Values initialized by `init_data`
		
		# Ports
		
		# Constraints

| Here is an example of the definition of a *PQ load* device:

.. testcode::

	def PQ_Load(b, data, init_data):
		
		# Parameters
		b.P = pyo.Param(initialize=data['P'])
		b.Q = pyo.Param(initialize=data['Q'])
		
		# Variables
		b.S = pyo.Var(initialize=init_data['S'], within=pyo.Reals)
		
		# Ports
		b.port_P = Port(initialize={'P': (b.P, Port.Extensive)})
		b.port_Q = Port(initialize={'Q': (b.Q, Port.Extensive)})
		b.port_S = Port(initialize={'S': (b.S, Port.Extensive)})
		
		# Constraints
		def Constraint_S(_b):
			return _b.P**2 + _b.Q**2 == _b.S**2
		b.c_S = pyo.Constraint(rule=Constraint_S)


They are meant to be called after defining a ``Block`` object as follows:

.. testcode::

	some_load = pyo.Block()
	data = {'P':10e6, 'Q':2e6}
	init_data = {'S':0}
	PQ_Load(some_load, data, init_data)

.. note::
   It is required to import the pyomo libraries and the corresponding devices' files:
   
	.. testcode::
   
		# Import pyomo
		import pyomo.environ as pyo
		from pyomo.network import *

		# Import devices
		from Devices.Loads import PQ_Load

One may find the need to add **time dependent** variables and relations. In such case a ``Set`` is defined:

.. testcode::

	time_list = [1, 2, 3, 4, 5] # 5 time steps
	t = pyo.Set(initialize=time_list)

and a ``t`` argument is added to the function as well as to its call

.. testcode::

	def PQ_Load(b, t, data, init_data):
		
		# Parameters
		b.P = pyo.Param(t, initialize=data['P'])
		b.Q = pyo.Param(t, initialize=data['Q'])
		
		# Variables
		b.S = pyo.Var(t, initialize=init_data['S'], within=pyo.Reals)
		
		# Ports
		b.port_P = Port(initialize={'P': (b.P, Port.Extensive)})
		b.port_Q = Port(initialize={'Q': (b.Q, Port.Extensive)})
		b.port_S = Port(initialize={'S': (b.S, Port.Extensive)})
		
		# Constraints
		def Constraint_S(_b, _t):
			return _b.P[_t]**2 + _b.Q[_t]**2 == _b.S[_t]**2
		b.c_S = pyo.Constraint(t, rule=Constraint_S)


	some_load = pyo.Block()
	data = {'P':[10e6,10e6,0,0,5e6] , 'Q':[2e6,1e6,0,0,0.5e6]}
	init_data = {'S':[0, 0, 0, 0, 0]}
	PQ_Load(some_load, t, data, init_data)



   
Hydro devices
-------------

This lists the devices related to hydro

Source
```````````````````

.. image:: img/Block_Source.svg
   :scale: 100 %
   :alt: Source block

.. automodule:: Devices.Sources
   :members:
   :undoc-members:
   :show-inheritance:


.. _HydroSwitch:

Hydro Switch
```````````````````

.. image:: img/Block_HydroSwitch.svg
   :scale: 100 %
   :alt: Hydro switch block
   
.. automodule:: Devices.HydroSwitch
   :members:
   :undoc-members:
   :show-inheritance:
   
.. seealso::
	`Switch`_
   
New Pump
```````````````````

.. image:: img/Block_NewPump.svg
   :scale: 100 %
   :alt: New pump block
   
.. automodule:: Devices.NewPumps
   :members:
   :undoc-members:
   :show-inheritance:


Pipes
```````````````````
   
.. image:: img/Block_Pipe_full.svg
   :scale: 100 %
   :alt: Pipe block
   
.. automodule:: Devices.Pipes
   :members:
   :undoc-members:
   :show-inheritance:


Pumps
```````````````````

.. image:: img/Block_Pump.svg
   :scale: 100 %
   :alt: Pump block
   
.. automodule:: Devices.Pumps
   :members:
   :undoc-members:
   :show-inheritance:
   

Reservoirs
```````````````````
   
.. image:: img/Block_Reservoir_full.svg
   :scale: 100 %
   :alt: Reservoir block
   
.. automodule:: Devices.Reservoirs
   :members:
   :undoc-members:
   :show-inheritance:
   

Turbines
```````````````````
   
.. image:: img/Block_Turbine.svg
   :scale: 100 %
   :alt: Turbine block
   
.. automodule:: Devices.Turbines
   :members:
   :undoc-members:
   :show-inheritance:
   

Electrical devices
------------------


EB (Pumping Station)
`````````````````````

.. image:: img/Block_EB.svg
   :scale: 100 %
   :alt: EB block
   
.. automodule:: Devices.EB
   :members:
   :undoc-members:
   :show-inheritance:


Main Grid
```````````````````

.. image:: img/Block_Grid.svg
   :scale: 100 %
   :alt: Grid block
   
.. automodule:: Devices.MainGrid
   :members:
   :undoc-members:
   :show-inheritance:


Solar PV
```````````````````

.. image:: img/Block_Solar.svg
   :scale: 100 %
   :alt: Solar PV block
   
.. automodule:: Devices.SolarPV
   :members:
   :undoc-members:
   :show-inheritance:


.. _Switch:

Switch
```````````````````

.. image:: img/Block_Switch.svg
   :scale: 100 %
   :alt: Switch block
   
.. automodule:: Devices.Switch
   :members:
   :undoc-members:
   :show-inheritance:


.. _Battery:

Battery
```````````````````

.. image:: img/Block_Battery.svg
   :scale: 100 %
   :alt: Battery block
   
.. automodule:: Devices.Batteries
   :members:
   :undoc-members:
   :show-inheritance:
   