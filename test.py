import matplotlib.pyplot as plt
import numpy as np
from func import *
# Создаем временную ось
t = np.linspace(0, 2, 500)  # 2 секунды
# Создаем "пульсовую волну" - основную гармонику сердца
heartbeat = np.sin(2 * np.pi * 1.2 * t) * np.exp(-0.5 * (t-1)**2)  # Импульс в районе 1 секунды
# DC компоненты (постоянные уровни)
dc_660 = 1.5
dc_940 = 1.8

# AC компоненты (переменные, пульсирующие части)
# Для нормальной сатурации AC компоненты противофазны
ac_660 = 0.3 * heartbeat  # Красный сигнал: больше крови -> больше сигнала
ac_940 = -0.4 * heartbeat # ИК сигнал: больше крови -> меньше сигнала

# Формируем полные сигналы
V_660 = dc_660 + ac_660
V_940 = dc_940 + ac_940

#t_ref = [0.5,0.7,0.8,1.5,1.7 ]
#V_ref = [1,0.7,0.8,0, 0]
#period = 1.5
#T = 3
#t = [0, 0.2]
#V = [0, 0]
#for i in range(T):
#    t.extend([x+i*period for x in t_ref])
#    V.extend(V_ref)
#print(t)

#V_new, t_new = create_dynamix(t,V,0,T*period,1000)
t_new, V_new = create_pulse_wave(Amp=10,zero_offset=4)
plt.figure()
plt.plot(t_new,V_new)

plt.show()

# Построение графиков
#plt.figure(figsize=(12, 6))

#plt.plot(t, V_660, color='red', label='V_660(t) (Красный, 660 нм)', linewidth=2)
#plt.plot(t, V_940, color='darkviolet', label='V_940(t) (ИК, 940 нм)', linewidth=2)

#plt.axhline(y=dc_660, color='red', linestyle='--', alpha=0.7, label='DC_660')
#plt.axhline(y=dc_940, color='darkviolet', linestyle='--', alpha=0.7, label='DC_940')

#plt.title('Сигналы фотодетектора в пульсоксиметре (нормальная сатурация)')
#plt.xlabel('Время, с')
#plt.ylabel('Амплитуда сигнала, В')
#plt.legend(loc='upper right')
#plt.grid(True, alpha=0.3)
#plt.ylim(0.5, 2.5)
#plt.show()






