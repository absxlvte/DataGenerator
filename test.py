from tests.tests_ppg import heart_rates

from func import *
import numpy as np
import random
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.signal import find_peaks

import numpy as np
import matplotlib.pyplot as plt

import neurokit2 as nk


heart_rate = 80

# Генерация ЭКГ с заданной ЧСС
ecg_signal = nk.ecg_simulate(
    duration=5,
    heart_rate=heart_rate,
    sampling_rate=1000,
    noise=0.1  # Уровень шума
)
ppg_signal = nk.ppg_simulate(
    duration=10,
    sampling_rate=1000,
    heart_rate=heart_rate
)



print(f"ecg: {len(ecg_signal)}")
# Визуализация
#nk.signal_plot(ppg_signal,sampling_rate=1000)

plt.show()
'''
k = 10
values, time = create_dynamix(np.array([0, 15, 30, 45, 60]), np.array([1, 0.2, 0.8, 0, 0])*k, 0, 60, 60)

def getFreq(t):
    idx = np.abs(time - t).argmin()
    return values[idx]

def geigerSensor(freq_func, time=60, sampling_rate=1000):
    n_samples = int(time*sampling_rate)
    time_array = np.linspace(0,time,n_samples)
    freqs = np.array([freq_func(t) for t in time_array])
    freqs = np.clip(freqs,0,sampling_rate/2)
    dt = 1.0 / sampling_rate  # длительность одного шага
    probability = freqs * dt  # эквивалентно freqs/sampling_rate
    binaryArray = np.random.random(n_samples)<probability
    return time_array,binaryArray.astype(int)

t, v = geigerSensor(getFreq)
print(f'число точек: {len(v)}')
plt.figure()
plt.plot(t,v, time, values/k)
plt.show()'''
'''
def is_non_decreasing(arr):
    return np.all(np.diff(arr) > 0)

def create_HeartRate(bpm,minV,maxV,baseV = 1.0,noise=0.07):
    #max bpm 250
    t,val = [],[]
    a0 = [0, 1, 40, 1, 0, -34, 118, -99, 0, 2, 21, 2, 0, 0, 0]
    d0 = [0,27/3,60/3,90/3,132/3,141/3,162/3,186/3,195/3,276/3,306/3,339/3,357/3,390/3,420/3]
    a = [x / max(a0) for x in a0]
    delay = (60000 - bpm*140)/bpm
    for i in range(bpm):
        t.extend([x + i * (140+delay) for x in d0])
        #t.append((i+1)*(140+delay))
        val.extend(a)
        #val.append(0)
    #t, val = increase_sampling(t, val, factor=20, method='linear')
    val += np.ones_like(val)*baseV
    val += np.random.rand(len(val))*noise
    for value in val:
        if value>maxV:
            value=maxV
        if value<minV:
            value=minV
    #print(f"t: {is_non_decreasing(t)};")
    return t,val

def increase_sampling(x, y, factor=10, method='cubic'):
    """
    Увеличивает дискретизацию массива
    Parameters:
    x, y - исходные массивы
    factor - во сколько раз увеличить количество точек
    method - метод интерполяции ('linear', 'cubic', 'spline')
    """
    x_new = np.linspace(min(x), max(x), len(x) * factor)
    if method == 'linear':
        y_new = np.interp(x_new, x, y)
    elif method == 'cubic':
        f = interpolate.interp1d(x, y, kind='cubic')
        y_new = f(x_new)
    elif method == 'spline':
        tck = interpolate.splrep(x, y, s=0)
        y_new = interpolate.splev(x_new, tck, der=0)
    return x_new, y_new



t,y = create_HeartRate(50,0.1,3)

ex = {}
for point in range(len(y)):
    if y[point]>1.5:
        ex.update({t[point]:y[point]})

print(len(ex.keys()))

plt.figure()
plt.plot(t,y, ex.keys(),ex.values(),'x')
plt.xlim(0,10000)
plt.show()'''

#converting \n\r -> None
'''qwe = ""
with open("repl.txt",'r') as file:
    qwe = file.read()
qwe = qwe.replace('\n', '').replace('\r', '').replace(' ','')
print(qwe)'''