from random import randint

from func import *
import random
from scipy.integrate import trapezoid
from scipy import interpolate
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk


def z_in_v(Z,N,V_ref,bip=False):
    if not bip:
        return (Z*V_ref)/(2**N)
    else:
        Zmsb = int(bin(Z)[2:].zfill(N)[0])
        Zother = int(bin(Z)[2:].zfill(N)[1:],2)
        return ((-1)**Zmsb*Zother*V_ref/2)/(2**(N-1))

def v_in_z(V,N,V_ref,bip=False):
    if not bip:
        return max(0,min(round((2**N*V)/V_ref),2**N-1))
    else:
        if V == 0:
            Z = 0
        else:
            Zmsb = 0 if V>= 0 else 1
            Zother = round((abs(V)*2**(N-1))/(V_ref/2)-1)
            Z = int(str(Zmsb)+str(bin(Zother)[2:].zfill(N)[1:]),2)
        return Z


















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