To evaluate the CEDER system, three cases were conducted for both Winter and Summer conditions. In the folder, you can find a CV file containing all solutions for the 24-hour simulation, as well as two plots representing the electrical balance, the hydraulic actions, and the levels of three reservoirs during the simulation.

The descriptions of the cases are explained below


Case 1 (C1):

	- No sizing or existing battery is considered.
	- Considered a 4 pumps in a bank. Individuals pumps work at 100% power or doesn't work.
	- 2 Banks of pump for pipe 1 and pipe 2. Constraint is added in order to prevent Bank1 and Bank2 to work in the same time. This equals a Hydroswitch with better simulation time.
	- Turbine of 40 kW.
	- Use of a VarSource to connect the R1 and R2.
	- No sizing of PV is considered. Only 16 kW already installed.
	- All data is obtained from the CEDER database for 24h of the summer and winter
	- Summer and Winter study.

Case 2 (C2):

	- Able to size a maximum of 66 kW of PV in total, while 16kW are already installed (56 kW new).	
	-  Able to size from 0 90kW and 400kWh of battery
	-  2 Simulation for winter  and summer.


Case 3 (C3):
	- Able to size a maximum of 66 kW of PV in total, while 16kW are already installed (56 kW new). 
	- Considering a 90 kW and 400 kWh  battery already installed in the system.
	- Able to size an extra battery up to 90 kW and 400 kWh
	- 2 simulations of winter and summer
