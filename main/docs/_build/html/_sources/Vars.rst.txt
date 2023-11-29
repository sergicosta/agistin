Variables and units
===============

.. csv-table:: Variables and units
	:header: "Magnitude", "Variable", "Units", "Comments"
	
	"**Hydraulic**"
	"Height", ":math:`H or z`", ":math:`m`"
	"Flow", ":math:`Q`", ":math:`m^3/s`", "*⚠ Interfering with Reactive power. Change to F ?*"
	"Control Flow", ":math:`Q_{control}`",":math:`m^3/s`", "Height depending flow"
	"Volume", ":math:`W`", ":math:`m^3`"
	"Rotational speed", ":math:`n`", ":math:`rpm`"
	"Linear pressure loss coefficient", ":math:`K`", ":math:`s/m^2`"
	"Pump efficiency", ":math:`\eta_{i}`", ":math:`pu`"
	"Pump On", ":math:`PumpOn`", "", "Status of the pump binary variable"
	"Auxiliar", ":math:`aux`", "", "Binary auxiliar variable"
	
	"**Electrical**"
	"Active power", ":math:`P`", ":math:`W`", "Defined positive if consumed by the device"
	"Reactive power", ":math:`Q`", ":math:`VAR`", "*TBD: Q(inductive) same sign as in P?*"
	"Current", ":math:`I`", ":math:`A`",
	"Voltage", ":math:`V`", ":math:`V`",
	"State of Health", ":math:`SOC`", ":math:`pu`"
	"Battery efficiency", ":math:`\eta_{ch/dch}`", ":math:`pu`", ":math:`\eta_{dch} = 2-\eta_{ch}`"
	
	"**Global**"
	"Time", ":math:`t`", ":math:`s`"
	"Energy", ":math:`E`", ":math:`Wh`"
	"Forecast", ":math:`f(t)`", ":math:`pu`", "In p.u. of peak power, referenced at :math:`1000 W/m^2`"

