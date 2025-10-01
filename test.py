from func import *
import numpy as np
import random
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
def testing(a: float=0.7,points: int=1000):
    k = -0.0704
    b = 451.6080
    f = np.linspace(10, 100,10)
    c = []
    y = np.array([])
    for h in range(55,10-1,-1):
        y = (700/(f+10)+h)*np.cos(np.deg2rad(random.randrange(-5,5+1)))
        y = np.round(y+a*np.random.rand(len(y)))
        s = trapezoid(y,f)
        c.append(k*s+b)
    ts = [0, 60, 160, 210, 410, 530, 660, 760, 830, 930, 960, 1060]
    ys = [120, 120, 250, 50, 140, 120, 120, 200, 60, 130, 120, 120]
    t = np.linspace(0,1060,points)
    y_interp = pchip_interpolate(ts,ys,t)
    yref = y_interp
    quantized_signal = np.zeros_like(yref)
    for i in range(len(c) - 1):
        mask = (yref >= c[i]) & (yref < c[i + 1])
        quantized_signal = np.where(mask, c[i], quantized_signal)
    mask = yref >= c[-1]
    quantized_signal = np.where(mask,c[-1],quantized_signal)
    values = []
    for i in range(len(quantized_signal)):
        for j in range(len(c)):
            if quantized_signal[i]==c[j]:
                values.append(56-j)
    array = np.zeros((len(values), 10))
    buf = np.array([])
    for i in range(len(values)):
        buf = (700/(f+10)+values[i])*np.cos(np.deg2rad(random.randrange(-5,5+1)))
        buf = np.round(buf+a*np.random.rand(len(y)))
        array[i, :] = buf
    print(np.shape(array))

    #calculating result
    final_c = []
    for i in range(np.size(array,0)):
        buf = trapezoid(array[i,:],f)
        final_c.append(k*buf+b)
    #print(f"{len(t)}  {len(final_c)}")
    plt.figure()
    plt.plot(t,final_c)
    plt.show()
    pass


testing()