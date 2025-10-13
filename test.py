from random import randint

from func import *
import random
from scipy.integrate import trapezoid
from scipy import interpolate
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk

time = np.array([0,5,10,15,25,40,45,50,55,60,70,90,95,100])
value = np.array([0,7,4,5,0,0,2,2,3,1,1,5,5,0])
points = 100
time_step = 1
d = 0.1
vliq = 1.400
noise = 0.01
t_liq = d/vliq
delta_t, time = create_dynamix(time,value,0,points*time_step,points)
delta_t = scale_signal(delta_t,t_liq,1)
T_send = time + t_liq * np.random.uniform(1.0, 1.5, size=len(time)) + noise * np.random.normal(size=len(time)) * np.max(np.abs(time))
T_recv = T_send + delta_t
state = [t>1.1*t_liq for t in delta_t]
T_send, T_recv, state, delt_v, time_new = bubbleCreate()

plt.figure()
plt.plot(state)
plt.plot(delt_v)
plt.show()
#======================================

"""with open("C:/Users/Zotin/Desktop/work/_Current tasks/Generator/GeneratedVal/120_70.txt", "r") as file:
    data = file.read()
Vref = 6
N = 12
k1 = 30
k2 = 25
data = [int(item) for item in data.split('\n') if item.strip()]
data = [item*Vref/(2**N) for item in data]
t = np.arange(len(data))
peaks, properties = find_peaks(data, distance=20)
peaks_dict = {t[index]:data[index] for index in peaks}
verh,niz =  4.45, 3.37
syst = k1 * verh
diast = k2 * niz
print(f"систолическое- {syst}; диастолическое- {diast}")
"""

"""A = 110
B = 25
N = 16
V_ref = 5
with open("C:/Users/Zotin/Desktop/work/_Current tasks/Generator/GeneratedVal/Датчик насыщения крови кислородом.txt", "r") as file:
    data = file.readlines()
z660, z940 = [], []
v660, v940 = [], []
ac660, dc660 = 0, 0
ac940, dc940 = 0, 0
amp660, amp940 = [], []
R = 0
for raw in data:
    z660.append(int(raw.split()[0]))
    z940.append(int(raw.split()[1]))
v660 = [x * V_ref / (2**N) for x in z660]
v940 = [x * V_ref / (2**N) for x in z940]
dc660 = sum(v660)/len(v660)
dc940 = sum(v940)/len(v940)
ac940 = [x-dc940 for x in v940]
ac660 = [x-dc660 for x in v660]
amp940 = (max(ac940)-min(ac940))/2
amp660 = (max(ac660)-min(ac660))/2
R = (amp660/dc660)/(amp940/dc940)
Sp = A - B * R
print(Sp)"""

#converting \n\r -> None
'''qwe = ""
with open("repl.txt",'r') as file:
    qwe = file.read()
qwe = qwe.replace('\n', '').replace('\r', '').replace(' ','')
print(qwe)'''