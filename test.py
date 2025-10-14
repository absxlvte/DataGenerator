from random import randint

from func import *
import random
from scipy.integrate import trapezoid
from scipy import interpolate
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk























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