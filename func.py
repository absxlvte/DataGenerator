import math
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
import math
from random import randrange
from enum import Enum
from scipy.integrate import trapezoid



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
    t_ref = [0.5, 0.7, 1.0, 1.5, 1.7]
    V_ref = [1, 0.7, 0.75, 0, 0]
    #V_ref = [1-0.5, 0.3-0.5, 0.4-0.5, 0-0.5, 0-0.5]
    period = 1.5
    T = math.ceil((t_stop - t_start)/period)
    t = [0, 0.2]
    V = [0, 0]
    for i in range(T):
        t.extend([x + i * period for x in t_ref])
        V.extend(V_ref)
    V = [x-0.5 for x in V]
    V_new, t_new = create_dynamix(t, V, t_start, T * period, points)
    if inverse:
        V_new = V_new*(-1)
    return t_new,V_new*2*Amp+zero_offset

def v_in_z(V,N,V_ref,bip=False):
    if not bip:
        return (2**N*V)/V_ref
    else:
        if V == 0:
            Z = 0
        else:
            Zmsb = 0 if V>= 0 else 1
            Zother = round((abs(V)*2**(N-1))/(V_ref/2)-1)
            Z = int(str(Zmsb)+str(bin(Zother)[2:].zfill(N)[1:]),2)
        return Z

def z_in_v(Z,N,V_ref,bip=False):
    if not bip:
        return (Z*V_ref)/(2**N)
    else:
        Zmsb = int(bin(Z)[2:].zfill(N)[0])
        Zother = int(bin(Z)[2:].zfill(N)[1:],2)
        return ((-1)**Zmsb*Zother*V_ref/2)/(2**(N-1))


def createNitrate(product: Literal['Tomatoes','Spinach','Beet','Cabbage','Carrot','Potato','Cucumbers'],
                  points:int, state: Literal['normal','increase','decrease']):
    if product not in ['Tomatoes','Spinach','Beet','Cabbage','Carrot','Potato','Cucumbers']:
        raise ValueError(f'missing product')
    if state not in ['normal','increase','decrease']:
        raise ValueError(f'missing state')
    z_values = {'Tomatoes':(33,36),
                'Spinach':(15,20),
                'Beet':(17,21),
                'Cabbage':(22,25),
                'Carrot':(25,31),
                'Potato':(28,33),
                'Cucumbers':(31,33)}
    Hlimit,Llimit =  85,12
    t = range(points)
    match state:
        case 'normal':
            z = [randrange(z_values[product][0],z_values[product][1]) for _ in t]
        case 'increase':
            z = [randrange(z_values[product][1]+1, Hlimit) for _ in t]
        case 'decrease':
            z = [randrange(Llimit, z_values[product][0]-1) for _ in t]
    return z

def createGlucoza(a: float=0.7,points: int=1000):
    k = -0.0704
    b = 451.6080
    f = np.linspace(10, 100,10)
    c = []
    y = np.array([])
    for h in range(55,10-1,-1):
        y = (700/(f+10)+h)*np.cos(np.deg2rad(randrange(-5,5+1)))
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
        buf = (700/(f+10)+values[i])*np.cos(np.deg2rad(randrange(-5,5+1)))
        buf = np.round(buf+a*np.random.rand(len(y)))
        array[i, :] = buf
    #calculating result
    final_c = []
    for i in range(np.size(array,0)):
        buf = trapezoid(array[i,:],f)
        final_c.append(k*buf+b)
    '''plt.figure()
    for i in range(array.shape[0]):
        plt.plot(array[i, :], label=f'Строка {i + 1}')
    #plt.plot(t,final_c)
    plt.show()'''
    return array,final_c,t

def generate_geiger_data(time_points, frequency_values, total_time=60, sampling_rate=1000, k=1.0):
    """
    Генерация данных гегер-счетчика с динамической частотой
    Args:
        time_points: моменты времени изменения частоты [0, 15, 30, 45, 60]
        frequency_values: значения частоты в эти моменты [1, 0.2, 0.8, 0, 0]
        total_time: общее время измерения (сек)
        sampling_rate: частота дискретизации (Гц)
        k: коэффициент масштабирования частоты
    """
    time_array_freq = np.linspace(0, total_time, int(total_time) + 1)
    freq_values,time_array_freq = create_dynamix(time_points, frequency_values * k, 0, total_time,int(total_time) + 1)
    def get_freq(t):
        idx = np.abs(time_array_freq - t).argmin()
        return freq_values[idx]
    n_samples = int(total_time * sampling_rate)
    time_array_sensor = np.linspace(0, total_time, n_samples)
    freqs = np.array([get_freq(t) for t in time_array_sensor])
    freqs = np.clip(freqs, 0, sampling_rate / 2)
    dt = 1.0 / sampling_rate
    probability = freqs * dt
    binary_array = np.random.random(n_samples) < probability
    return time_array_sensor, binary_array.astype(int), time_array_freq, freq_values
def scale_signal(signal, target_min, target_max, offset=0):
    signal = np.array(signal, dtype=float)
    current_min = np.min(signal)
    current_max = np.max(signal)
    current_amplitude = current_max - current_min
    target_amplitude = target_max - target_min
    if current_amplitude == 0:
        return np.full_like(signal, (target_min + target_max) / 2 + offset)
    current_center = (current_max + current_min) / 2
    target_center = (target_max + target_min) / 2
    scaled = (signal - current_center) * (target_amplitude / current_amplitude) + target_center
    scaled += offset
    return scaled


def bubbleCreate(time=np.array([0, 50, 100, 150, 250, 400, 450, 500, 550, 600, 700, 900, 950, 1000]),
                   value=np.array([0, 7, 4, 5, 0, 0, 2, 2, 3, 1, 1, 5, 5, 0]),
                   points=1000,
                   time_step=1,
                   d=0.1,
                   vliq=1.400,
                   noise=0.01):
    t_liq = d / vliq
    delta_t, time_new = create_dynamix(time, value, 0, points * time_step, points)
    delta_t = scale_signal(delta_t, t_liq, 1)
    T_send = time_new + t_liq * np.random.uniform(1.0, 1.5, size=len(time_new)) + noise * np.random.normal(size=len(time_new)) * np.max(np.abs(time_new))
    T_recv = T_send + delta_t
    T_recv = np.clip(T_recv,0,max(T_recv))
    T_send = np.clip(T_send,0, max(T_recv))
    state = [t > 1.1 * t_liq for t in delta_t]
    return T_send, T_recv, state, delta_t, time_new

