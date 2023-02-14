import numpy as np
from IntensityofSunlight import PlanckL
from TemperatureofDebris import TofDebris

lambda1 = 4.

T = TofDebris(t)
I_total = PlanckL(lambda1, T)
Q_total = I_total * (4 * np.pi * radius1 * radius1 * 1.E-12) * (rho * 1.E15 * 1.E5)
Q_perphoton = Q_total / nphotons  # 单光束出射功率