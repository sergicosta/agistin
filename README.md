# AGISTIN Task 4.6 Optimization Tool

[<img src="https://www.agistin.eu/wp-content/uploads/2022/10/AGISTIN_logo_1.png" height="128px" align="center" alt="AGISTIN logo">](https://www.agistin.eu)

Welcome to the AGISTIN Task 4.6 Optimization Tool. This tool is a part of the [AGISTIN project](https://www.agistin.eu) on advanced storage integration methods.

The Task 4.6 Optimization Tool is aimed to solve **energy storage optimization** problems in order to transform irrigation canals into flexible energy storage systems:
* **Sizing**, defining key changes in topology and new devices installation, such as power converter design, new installed PV, turbomachinery (pumps and turbines), batteries... 
* **Operation** of the system in different scenarios and objectives in optimal conditions

The tool runs in [Python](https://www.python.org/) and uses the [Pyomo](http://www.pyomo.org/) open-source optimization library

[<img src="https://www.python.org/static/img/python-logo.png" height="50px" align="left" alt="Python logo">](https://www.python.org/) [<img src="https://pyomo.readthedocs.io/en/stable/_images/PyomoNewBlue3.png" height="36px" align="left" alt="Pyomo logo">](http://www.pyomo.org/)

<br>
<br>
<br>

*AGISTIN is supported by the European Union’s Horizon Europe programme under agreement 101096197.*

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributors](#contributors)
- [License](#license)

## Installation

In order to run the tool you require:
* **Python $\geq$ 3.11**: Download from the [official site](https://www.python.org/downloads/)
* **Pyomo $\geq$ 6.8**: Install the library using ``pip`` or ``conda``, following the instructions at the [official site](https://pyomo.readthedocs.io/en/stable/installation.html)
	* Using pip:
		```
		$ pip install pyomo
		```
	* Using conda:
		```
		$ conda install -c conda-forge pyomo
		```
* **Solver(s)**: Find the [supported Pyomo solvers](https://pyomo.readthedocs.io/en/stable/solving_pyomo_models.html#supported-solvers) at the official site or by running:
	```
	$ pyomo help --solvers
	```
	You may use [Interior Point Optimizer (Ipopt)](https://coin-or.github.io/Ipopt/), which latest realease can be downloaded from the [official github](https://github.com/coin-or/Ipopt)

## Usage

Find usage examples at ``./AGISTIN tool usage examples.pdf`` ([here](./AGISTIN%20tool%20usage%20examples.pdf))

You may find the full documentation at ``./main/docs/_build/html/index.html`` [here](./main/docs/_build/html/index.html)

## Contributors
In alphabetical order:

[<img src="https://citcea.upc.edu/ca/shared/logos/citcea.png" height="48px" align="center" alt="CITCEA-UPC logo">](https://citcea.upc.edu/ca)
* Carla Cinto Campmany (carla.cinto@upc.edu)
* Sergi Costa Dilmé (sergi.costa.dilme@upc.edu)
* Pau García Motilla (pau.garcia.motilla@upc.edu)
* Paula Muñoz Peña (paula.munoz.pena@upc.edu)
* Juan Carlos Olives Camps (juan.carlos.olives@upc.edu)
* Eduardo Prieto Araujo (eduardo.prieto-araujo@upc.edu)

[<img src="https://www.epri.com/static/media/epri-logo-2021-white.324099d1.svg" height="32px" align="center" alt="EPRI logo">](https://www.epri.com/)
* Daniel Pombo (dpombo@epri.com)

## Citation

If used in a scientific publication, we appreciate a citation on the following paper:

S. Costa-Dilmé, J. C. Olives-Camps, P. Muñoz-Peña, P. García-Motilla, O. Gomis-Bellmunt, and E. Prieto-Araujo, “Multi-physics operation and sizing optimisation in pyomo: Application to large irrigation systems,”
in 2024 Open Source Modelling and Simulation of Energy Systems (OSMSES), pp. 1–6, 2024

```
@INPROCEEDINGS{AGISTIN_opt_tool,
  author={Costa-Dilmé, Sergi and Olives-Camps, J. Carlos and Muñoz-Peña, Paula and García-Motilla, Pau and Gomis-Bellmunt, Oriol and Prieto-Araujo, Eduardo},
  booktitle={2024 Open Source Modelling and Simulation of Energy Systems (OSMSES)}, 
  title={Multi-physics operation and sizing optimisation in Pyomo: Application to large irrigation systems}, 
  year={2024},
  volume={},
  number={},
  pages={1-6},
  keywords={Irrigation;Uncertainty;Object oriented modeling;Libraries;Demand response;Power systems;Optimization;Open source;Multi-physics simulation;Power system analysis;Optimisation tool;Python;Object-oriented;Irrigation systems},
  doi={10.1109/OSMSES62085.2024.10668997}
}
```

## License
<p xmlns:cc="http://creativecommons.org/ns#" >This work is licensed under <a href="http://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1"></a></p> 
