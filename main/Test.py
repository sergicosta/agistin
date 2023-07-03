# -*- coding: utf-8 -*-

import pyomo.environ as pyo
from Devices.Reservoirs import Reservoirs
from Devices.Pumps import Pumps
from Devices.Sources import Sources
from Functionalities.Construct import Builder

l_t = list(range(5))

model = pyo.AbstractModel()

pumps_set = Pumps()
res_set = Reservoirs()
src_set = Sources()

pumps_set.add('Bomba1', A=106.3,B=(2e-5)*3600**2,rpm_nom=1450, init_Q=0,init_H=0, l_t=l_t) 
pumps_set.add('Bomba2', A=106.3,B=(2e-5)*3600**2,rpm_nom=1450, init_Q=0,init_H=0, l_t=l_t) 

res_set.add('Bassa1', [], ['Pipe1'], 142869*0.5, 142869, 142869*0.5, 328, 335.50, 142869*0.5, [0,0,0,0,0], l_t=l_t)
res_set.add('Bassa3', ['Pipe1'], [], 85268*0.5, 85268, 85268*0.5, 414, 423.50, 85268*0.5, [0,0,0,0,0], l_t=l_t)

src_set.add('Src1', [1,1,2,2,3])

Builder(model, l_t, res_set, pumps_set)
