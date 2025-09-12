import math
from func import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

# Инициализация
np.random.seed(42)  # для воспроизводимости результатов
f = np.linspace(10, 100, 10)

# Создание функций y1-y10
y1 = (700/(f + 10) + 10) * np.cos(np.deg2rad(np.random.randint(-5, 5, 1)[0]))
y2 = (700/(f + 10) + 15) * np.cos(np.deg2rad(np.random.randint(-5, 5, 1)[0]))
y3 = (700/(f + 10) + 20) * np.cos(np.deg2rad(np.random.randint(-5, 5, 1)[0]))
y4 = (700/(f + 10) + 25) * np.cos(np.deg2rad(np.random.randint(-5, 5, 1)[0]))
y5 = (700/(f + 10) + 30) * np.cos(np.deg2rad(np.random.randint(-5, 5, 1)[0]))
y6 = (700/(f + 10) + 35) * np.cos(np.deg2rad(np.random.randint(-5, 5, 1)[0]))
y7 = (700/(f + 10) + 40) * np.cos(np.deg2rad(np.random.randint(-5, 5, 1)[0]))
y8 = (700/(f + 10) + 45) * np.cos(np.deg2rad(np.random.randint(-5, 5, 1)[0]))
y9 = (700/(f + 10) + 50) * np.cos(np.deg2rad(np.random.randint(-5, 5, 1)[0]))
y10 = (700/(f + 10) + 55) * np.cos(np.deg2rad(np.random.randint(-5, 5, 1)[0]))

a = 0.7
# Добавление шума
y1 = np.round(y1 + a * np.random.randn(len(y1)))
y2 = np.round(y2 + a * np.random.randn(len(y2)))
y3 = np.round(y3 + a * np.random.randn(len(y3)))
y4 = np.round(y4 + a * np.random.randn(len(y4)))
y5 = np.round(y5 + a * np.random.randn(len(y5)))
y6 = np.round(y6 + a * np.random.randn(len(y6)))
y7 = np.round(y7 + a * np.random.randn(len(y7)))
y8 = np.round(y8 + a * np.random.randn(len(y8)))
y9 = np.round(y9 + a * np.random.randn(len(y9)))
y10 = np.round(y10 + a * np.random.randn(len(y10)))

# Вычисление площади под кривыми (ИСПРАВЛЕНО: используем np.trapz вместо scipy.integrate.trapz)
s1 = trapezoid(y1, f)
s2 = trapezoid(y2, f)
s3 = trapezoid(y3, f)
s4 = trapezoid(y4, f)
s5 = trapezoid(y5, f)
s6 = trapezoid(y6, f)
s7 = trapezoid(y7, f)
s8 = trapezoid(y8, f)
s9 = trapezoid(y9, f)
s10 = trapezoid(y10, f)

print(f's1 = {s1:.4f}')
print(f's2 = {s2:.4f}')
print(f's3 = {s3:.4f}')
print(f's4 = {s4:.4f}')
print(f's5 = {s5:.4f}')
print(f's6 = {s6:.4f}')
print(f's7 = {s7:.4f}')
print(f's8 = {s8:.4f}')
print(f's9 = {s9:.4f}')
print(f's10 = {s10:.4f}')

# Вычисление коэффициентов
k = (300 - 20) / (s1 - s10)
b = 300 - k * s1

# Вывод C
print(f'k = {k:.4f}')
print(f'c1 = {k * s1 + b:.4f}')
print(f'c2 = {k * s2 + b:.4f}')
print(f'c3 = {k * s3 + b:.4f}')
print(f'c4 = {k * s4 + b:.4f}')
print(f'c5 = {k * s5 + b:.4f}')
print(f'c6 = {k * s6 + b:.4f}')
print(f'c7 = {k * s7 + b:.4f}')
print(f'c8 = {k * s8 + b:.4f}')
print(f'c9 = {k * s9 + b:.4f}')
print(f'c10 = {k * s10 + b:.4f}')

# Построение графиков
plt.figure()
plt.plot(f, y1, '*-')
plt.plot(f, y2, '*-')
plt.plot(f, y3, '*-')
plt.plot(f, y4, '*-')
plt.plot(f, y5, '*-')
plt.plot(f, y6, '*-')
plt.plot(f, y7, '*-')
plt.plot(f, y8, '*-')
plt.plot(f, y9, '*-')
plt.plot(f, y10, '*-')
plt.ylabel('Z,kOm')
plt.xlabel('f,kHz')
plt.legend(['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10'])
plt.show()

# h: 10 eq 300 - 55 eq 20
c = []
a = 0.7
for h in range(55,10,-1):
    y = [(700/(f+10)+h)*np.cos(np.deg2rad(np.random.randint(-5, 5)))]
    y = np.round(y+a*np.random.randn(len(y)))
    s = trapezoid(y,f)
    c_val = k*s+b
    c.append(c_val[0])
ts = [0,60,160,210,410,530,660,760,830,930,960,1060]
ys = [120,120,250,50,140,120,120,200,60,130,120,120]
y_interp, t_new = create_dynamix(ts,ys,0,1060,1000)
y_ref = y_interp

values = y_ref
massiv = []
for i in range(len(values)):
    buf = (700/(f + 10) + values[i]) * np.cos(np.deg2rad(np.random.randint(-5, 5,)))
    buf = np.round(buf + a * np.random.randn(len(buf)))
    massiv.append(buf)

massiv = np.array(massiv)
final_c = []
# Вычисление результата как студент
for i in range(massiv.shape[0]):
    buf = trapezoid(massiv[i, :], f)
    temp = k * buf + b
    #print(f'temp({i}): {temp}')
    final_c.append(temp)
final_c = np.array(final_c)
# Построение графиков
plt.figure(figsize=(10, 8))
plt.plot(t_new, final_c,'--')
plt.title('student')
plt.tight_layout()
plt.show()
plt.figure()
for i in range(massiv.shape[0]):
    plt.plot(f,massiv[i,:], '*-')
plt.ylabel('Z,kOm')
plt.xlabel('f,kHz')
plt.show()
