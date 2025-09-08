from func import *
import numpy as np
import matplotlib.pyplot as plt
import math

createNitrate('Beet',100,10,10)

#a0 = [0,1,40,1,0,-34,118,-99,0,2,21,2,0,0,0];
#d0 = [0,27,59,91,131,141,163,185,195,275,307,339,357,390,440];
#a = [x/max(a0) for x in a0]
#d = d0
#d = [math.ceil(x*750/d0[-1]) for x in d0]
#y, t = create_dynamix(np.array(d), np.array(a),0,750,1000)
#t,y = create_HeartRate(80, 750*10,0.1,3,1000)
#plt.figure()
#plt.plot(t,y,'b')
#plt.plot(d,a,'r*--')
#plt.show()