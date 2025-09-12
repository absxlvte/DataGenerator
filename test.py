import math
from func import *
import numpy as np
import matplotlib.pyplot as plt
import math
import os


massiv, val, time = createGlucoza(1000)

plt.figure()
plt.plot(time,val)
plt.show()