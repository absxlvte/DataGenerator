import math

import numpy as np
import matplotlib.pyplot as plt
import math
def pchip_slopes(x, y):
    n = len(x)
    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros(n)
    # Случай двух точек
    if n == 2:
        d[:] = delta[0]
        return d
    # Внутренние точки
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0:
            d[i] = 0.0
        else:
            h1, h2 = h[i - 1], h[i]
            δ1, δ2 = delta[i - 1], delta[i]
            w1 = 2 * h2 + h1
            w2 = h2 + 2 * h1
            d[i] = (w1 + w2) / (w1 / δ1 + w2 / δ2)
    # Граничные точки с коррекцией
    d[0] = ((2 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    if np.sign(d[0]) != np.sign(delta[0]):
        d[0] = 0
    elif np.sign(delta[0]) != np.sign(delta[1]) and abs(d[0]) > abs(3 * delta[0]):
        d[0] = 3 * delta[0]
    d[-1] = ((2 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    if np.sign(d[-1]) != np.sign(delta[-1]):
        d[-1] = 0
    elif np.sign(delta[-1]) != np.sign(delta[-2]) and abs(d[-1]) > abs(3 * delta[-1]):
        d[-1] = 3 * delta[-1]
    return d


def pchip_interpolate(x, y, x_new):
    x = np.asarray(x)
    y = np.asarray(y)
    x_new = np.asarray(x_new)
    # Проверка сортировки
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be in ascending order")
    # Рассчитываем производные
    d = pchip_slopes(x, y)
    # Находим интервалы для новых точек
    indices = np.searchsorted(x, x_new) - 1
    indices = np.clip(indices, 0, len(x) - 2)
    # Коэффициенты полиномов
    h = x[1:] - x[:-1]
    delta = (y[1:] - y[:-1]) / h
    i = indices
    h_i = h[i]
    x_loc = x_new - x[i]
    # Коэффициенты кубических полиномов
    c0 = y[i]
    c1 = d[i]
    c2 = (3 * delta[i] - 2 * d[i] - d[i + 1]) / h_i
    c3 = (d[i] + d[i + 1] - 2 * delta[i]) / (h_i ** 2)
    # Вычисление интерполированных значений
    return c0 + c1 * x_loc + c2 * x_loc ** 2 + c3 * x_loc ** 3

def create_dynamix(time_intervals_array_np,extremum_values_array_np,T_start,T_stop,N_points):
    t_new = np.linspace(T_start, T_stop, N_points)
    result = pchip_interpolate(time_intervals_array_np,extremum_values_array_np, t_new)
    return result, t_new

def create_pulse_wave(Amp=1,t_start=0,t_stop=5,zero_offset=0, points=1000, inverse=False):
    t_ref = [0.5, 0.7, 0.8, 1.5, 1.7]
    V_ref = [1, 0.7, 0.8, 0, 0]
    period = 1.5
    T = math.ceil((t_stop - t_start)/period)
    t = [0, 0.2]
    V = [0, 0]
    for i in range(T):
        t.extend([x + i * period for x in t_ref])
        V.extend(V_ref)
    V_new, t_new = create_dynamix(t, V, t_start, T * period, points)
    if inverse:
        V_new = V_new*(-1)
    return t_new,V_new*Amp+zero_offset

# Пример использования
#t = np.array([0, 60, 160, 210, 410, 530, 660, 760, 830, 930, 960, 1060])
#y = np.array([120, 120, 250, 50, 140, 120, 120, 200, 60, 130, 120, 120])
#t_new = np.linspace(0, 1060, 1000)
#y_interp = pchip_interpolate(t, y, t_new)

#plt.figure()
#plt.plot(t_new,y_interp)
#plt.figure()
#plt.plot(t,y)
#plt.show()