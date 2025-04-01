[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

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
- [Development](#development)

## Installation

In order to run the tool you require:
* **Python $\geq$ 3.11**: Download from the [official site](https://www.python.org/downloads/)
* **Pyomo $\geq$ 6.8**: Install the library using ``pip`` or ``conda``, following the instructions at the [official site](https://pyomo.readthedocs.io/en/stable/getting_started/installation.html)
	* Using pip:
		```
		$ pip install pyomo
		```
	* Using conda:
		```
		$ conda install -c conda-forge pyomo
		```
* **Solver(s)**: Find the [supported Pyomo solvers](https://pyomo.readthedocs.io/en/stable/explanation/solvers/index.html) at the official site or by running:
	```
	$ pyomo help --solvers
	```
  	Although many solvers and algorithms do exist and may be suitable for your particular case, we obtained good results with those listed below.

	For Non-Linear Problems (NLP):
	* COIN-OR [Interior Point Optimizer (Ipopt)](https://coin-or.github.io/Ipopt/), (see the [official github](https://github.com/coin-or/Ipopt))

	For Mixed Integer Non-Linear Problems (MINLP):
	* COIN-OR [Bonmin algorithms](https://coin-or.github.io/Bonmin/), (see the [official github](https://github.com/coin-or/Bonmin))
 	* COIN-OR [Couenne](https://www.coin-or.org/Couenne/), (see the [official github](https://github.com/coin-or/Couenne)).

   	To install COIN-OR solvers you may follow their [official github](https://github.com/coin-or) instructions, download the binaries, use the [coinbrew](https://coin-or.github.io/coinbrew/) script or find your Linux distribution package in [repology](https://repology.org/project/coin-or-bonmin/versions).

	

## Usage

Find usage examples at ``./AGISTIN tool usage examples.pdf`` ([here](./AGISTIN%20tool%20usage%20examples.pdf))

You may find the full documentation at ``./main/docs/_build/html/index.html`` [here](./main/docs/_build/html/index.html)

## Development

[<img src="https://citcea.upc.edu/ca/shared/logos/citcea.png" height="48px" align="center" alt="CITCEA-UPC logo">](https://citcea.upc.edu/ca)
* Sergi Costa Dilmé* (sergi.costa.dilme@upc.edu)
* Juan Carlos Olives Camps* (juan.carlos.olives@upc.edu)
* Pau García Motilla (pau.garcia.motilla@upc.edu)
* Carla Cinto Campmany (carla.cinto@upc.edu)
* Paula Muñoz Peña (paula.munoz.pena@upc.edu)
* Eduardo Prieto Araujo* (eduardo.prieto-araujo@upc.edu)

(*: corresponding authors)

### Other contributions
* Daniel Pombo (dpombo@epri.com): Utilities.clear_clc() function and insights 

## Citation

If used in a scientific publication, we appreciate you cite the following paper:

### Complementary citations

S. Costa-Dilmé, J. C. Olives-Camps, P. Muñoz-Peña, P. García-Motilla, O. Gomis-Bellmunt, and E. Prieto-Araujo, “Redesign of Large-Scale Irrigation Systems for Flexible Energy Storage,”
in 2024 IEEE PES Innovative Smart Grid Technologies Europe (ISGT EUROPE), pp. 1–6, 2024 [[Link](https://www.doi.org/10.1109/isgteurope62998.2024.10863693)]

```
@INPROCEEDINGS{AGISTIN_ISGT2024,
  author    = {Costa-Dilmé, Sergi and Olives-Camps, J. Carlos and Muñoz-Peña, Paula and García-Motilla, Pau and Gomis-Bellmunt, Oriol and Prieto-Araujo, Eduardo},
  booktitle = {2024 IEEE PES Innovative Smart Grid Technologies Europe (ISGT EUROPE)},
  title     = {Redesign of Large-Scale Irrigation Systems for Flexible Energy Storage},
  year      = {2024},
  month     = oct,
  pages     = {1--5},
  publisher = {IEEE},
  doi       = {10.1109/isgteurope62998.2024.10863693},
}
```

S. Costa-Dilmé, J. C. Olives-Camps, P. Muñoz-Peña, P. García-Motilla, O. Gomis-Bellmunt, and E. Prieto-Araujo, “Multi-physics operation and sizing optimisation in pyomo: Application to large irrigation systems,”
in 2024 Open Source Modelling and Simulation of Energy Systems (OSMSES), pp. 1–6, 2024 [[Link](https://www.doi.org/10.1109/OSMSES62085.2024.10668997)]

```
@INPROCEEDINGS{AGISTIN_opt_tool,
  author={Costa-Dilmé, Sergi and Olives-Camps, J. Carlos and Muñoz-Peña, Paula and García-Motilla, Pau and Gomis-Bellmunt, Oriol and Prieto-Araujo, Eduardo},
  booktitle={2024 Open Source Modelling and Simulation of Energy Systems (OSMSES)}, 
  title={Multi-physics operation and sizing optimisation in Pyomo: Application to large irrigation systems}, 
  year={2024},
  pages={1-6},
  keywords={Irrigation;Uncertainty;Object oriented modeling;Libraries;Demand response;Power systems;Optimization;Open source;Multi-physics simulation;Power system analysis;Optimisation tool;Python;Object-oriented;Irrigation systems},
  doi={10.1109/OSMSES62085.2024.10668997}
}
```
