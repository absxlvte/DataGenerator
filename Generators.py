from abc import ABC, abstractmethod
from locale import normalize

import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
from fontTools.merge.util import first
from scipy.signal import find_peaks
from func import *

class DataGenerator(ABC):
    def __init__(self):
        self.data = None
        self.time = None
        self.params = {}
        self.def_params = {}
    @abstractmethod
    def configurate(self):
        pass
    def generate(self):
        pass
    def plot(self,ax):
        if self.data is not None:
            ax.clear()
            ax.plot(self.data)
            ax.set_title(f"{self.__class__.__name__} Data")
            return ax

class GeneratorManager:
    def __init__(self):
        self.generators = {
            'Test - Sin()': SinusoidalGenerator(),
            'Датчик температуры': TemperatureSensor(),
            'Гидравлический датчик давления': HydraulicSensor(),
            'Датчик наличия крови': BloodSensor(),
            'Датчик насыщения крови кислородом': PPGSensor(),
            'Датчик ЧСС': HeartRateSensor(),
            'Датчик pH': pHSensor(),
            'Датчик уровня жидкости': LiquidLvlSensor(),
            'Датчик наличия пузырьков': BubbleSensor(),
            'Счетчик Гейгера': GeigerSensor(),
            'Датчик артериального давления': BloodPressureSensor(),
            'Датчик расхода': ConsSensor(),
            'Датчик нитратов': NitrateSensor(),
            'Датчик глюкозы': GlucozeSensor(),
            'Капнограф': Capnograph(),
            'Датчик проводимости': ConductivitySensor()
        }
        self.current_generator = None
    def set_generator(self,name):
        self.current_generator = self.generators.get(name)
        return self.current_generator
    def generators_list(self):
        return list(self.generators.keys())

class SinusoidalGenerator(DataGenerator):
    def __init__(self):
        super().__init__()
        self.data = None
        self.time = None
        self.params = {
            'amp':1.0,
            'freq':1.0,
            'phase':0.0,
            'offset':0.0,
            'points':1000,
            't_min':0.0,
            't_max':10.0,
            'T_interval': None,
            'Val_interval': None
        }
        self.def_params = {
            'amp': 1.0,
            'freq': 1.0,
            'phase': 0.0,
            'offset': 0.0,
            'points': 1000,
            't_min': 0.0,
            't_max': 10.0,
            'T_interval': [-1],
            'Val_interval': [-1]
        }
    def configurate(self,amp=None,freq=None,offset=None,phase=None,points=None,t_min=None,t_max=None,T_interval=None,Val_interval=None):
        if amp is not None: self.params['amp'] = amp
        if freq is not None: self.params['freq'] = freq
        if phase is not None: self.params['phase'] = phase
        if offset is not None: self.params['offset'] = offset
        if points is not None: self.params['points'] = points
        if t_min is not None: self.params['t_min'] = t_min
        if t_max is not None: self.params['t_max'] = t_max
        if T_interval is not None: self.params['T_interval'] = T_interval
        if Val_interval is not None: self.params['Val_interval'] = Val_interval
    def generate(self):
        if self.params['T_interval'] is not None and self.params['Val_interval'] is not None and  len(self.params['T_interval']) >= 2 and len(self.params['Val_interval']) >= 2:
            if len(self.params['T_interval']) != len(self.params['Val_interval']):
                raise ValueError("T_interval и Val_interval должны иметь одинаковую длину")
            self.data, self.time = create_dynamix(np.array(self.params['T_interval']), np.array(self.params['Val_interval']), self.params['t_min'], self.params['t_max'], self.params['points'])
        else:
            self.time = np.linspace(self.params['t_min'],self.params['t_max'], self.params['points'])
            self.data = self.params['offset']+self.params['amp']*np.sin(self.params['freq']*2*np.pi*self.time+self.params['phase'])
    def plot(self,ax):
        ax.clear()
        if (self.data is not None) and (self.time is not None):
            ax.plot(self.time,self.data,'b-')
            if (self.params['T_interval'] is not None and self.params['Val_interval'] is not None) and (self.params['T_interval'] != [-1] and self.params['Val_interval'] != [-1]):
                ax.plot(self.params['T_interval'],
                        self.params['Val_interval'],
                        'rx')
                ax.plot(self.params['T_interval'], self.params['Val_interval'], 'b--', linewidth=1, alpha=0.7)
            title = ("Интерполированная ломаная"
                     if (self.params['T_interval'] is not None and
                         self.params['Val_interval'] is not None)
                     else f"Синусоида: A={self.params['amp']}, "
                          f"f={self.params['freq']}, φ={self.params['phase']}")
            ax.set_title(title)
            ax.grid(True)
class TemperatureSensor(DataGenerator):
    def __init__(self):
        super().__init__()
        self.data = None
        self.time = None
        self.params = {
            'n_outliers': 5,
            'strength': 1.5,
            'noise_level':0.01,
            'points': 1000,
            'z_min':0,
            'z_max':65535,
            't_min': 0.0,
            't_max': 10.0,
            'T_interval': None,
            'Val_interval': None
        }
        self.def_params = {
            'n_outliers': 5,
            'noise_level': 0.05,
            'points': 1000,
            't_min': 0.0,
            't_max': 10.0,
            'T_interval': [-1],
            'Val_interval': [-1]
        }
        self.signal = None

    def configurate(self, n_outliers=None, noise_level=None, points=None,t_min=None,t_max=None,T_interval=None,Val_interval=None):
        if n_outliers is not None: self.params['n_outliers'] = n_outliers
        if noise_level is not None: self.params['noise_level'] = noise_level
        if points is not None: self.params['points'] = points
        if t_min is not None: self.params['t_min'] = t_min
        if t_max is not None: self.params['t_max'] = t_max
        if T_interval is not None: self.params['T_interval'] = T_interval
        if Val_interval is not None: self.params['Val_interval'] = Val_interval
    def generate(self):
        if self.params['T_interval'] is not None and self.params['Val_interval'] is not None and  len(self.params['T_interval']) >= 2 and len(self.params['Val_interval']) >= 2:
            if len(self.params['T_interval']) != len(self.params['Val_interval']):
                raise ValueError("T_interval и Val_interval должны иметь одинаковую длину")
            self.data, self.time = create_dynamix(np.array(self.params['T_interval']), np.array(self.params['Val_interval']), self.params['t_min'], self.params['t_max'], self.params['points'])
            self.add_outliers(self.params['n_outliers'], self.params['strength'])
            self.add_noise(self.params['noise_level'])
            self.normalize(self.params['z_min'], self.params['z_max'])
            self.data = self.data.astype(int)
    def normalize(self,min_val,max_val):
        current_min = np.min(self.data)
        current_max = np.max(self.data)
        current_min_base = np.min(self.params['Val_interval'])
        current_max_base = np.max(self.params['Val_interval'])
        self.data = min_val + (self.data - current_min) * (max_val - min_val) / (current_max - current_min)
        self.signal = min_val + (self.params['Val_interval'] - current_min_base) * (max_val - min_val) / (current_max_base - current_min_base)

    def add_outliers(self,n_outliers,strength):
        indices = np.random.choice(len(self.time), n_outliers, replace=False)
        direction = np.random.choice([-1, 1], n_outliers)
        outlier_values = strength * np.max(np.abs(self.data)) * direction
        self.data[indices] += outlier_values
    def add_noise(self,noise_level):
        noise = noise_level * np.max(np.abs(self.data)) * np.random.normal(size=len(self.time))
        self.data += noise
    def plot(self, ax):
        if self.data is not None:
            ax.clear()
            ax.set_title(f"{self.__class__.__name__} Data")
            ax.plot(self.time, self.data, 'b-')
            ax.plot(self.params['T_interval'],
                    self.signal,
                    'rx')
            ax.plot(self.params['T_interval'], self.signal, 'r--', linewidth=1, alpha=0.7)
            ax.set_ylim(0,65535)
            ax.grid(True)
            return ax

class HydraulicSensor(DataGenerator):
    def __init__(self):
        super().__init__()
        self.data = None
        self.params = {
            'n_outliers': 5,
            'strength': 1.5,
            'noise_level': 0.01,
            'points': 1000,
            'z_min': 0,
            'z_max': 1023,
            't_min': 0.0,
            't_max': 10.0,
            'T_interval': None,
            'Val_interval': None
        }
        self.def_params = {
            'n_outliers': 5,
            'noise_level': 0.05,
            'points': 1000,
            't_min': 0.0,
            't_max': 10.0,
            'T_interval': [-1],
            'Val_interval': [-1]
        }
        self.signal = None
        self.time = None

    def configurate(self, n_outliers=None, noise_level=None, points=None,t_min=None,t_max=None,T_interval=None,Val_interval=None):
        if n_outliers is not None: self.params['n_outliers'] = n_outliers
        if noise_level is not None: self.params['noise_level'] = noise_level
        if points is not None: self.params['points'] = points
        if t_min is not None: self.params['t_min'] = t_min
        if t_max is not None: self.params['t_max'] = t_max
        if T_interval is not None: self.params['T_interval'] = T_interval
        if Val_interval is not None: self.params['Val_interval'] = Val_interval

    def generate(self):
        if self.params['T_interval'] is not None and self.params['Val_interval'] is not None and len(
                self.params['T_interval']) >= 2 and len(self.params['Val_interval']) >= 2:
            if len(self.params['T_interval']) != len(self.params['Val_interval']):
                raise ValueError("T_interval и Val_interval должны иметь одинаковую длину")
            self.data, self.time = create_dynamix(np.array(self.params['T_interval']),
                                                  np.array(self.params['Val_interval']), self.params['t_min'],
                                                  self.params['t_max'], self.params['points'])
            self.add_outliers(self.params['n_outliers'], self.params['strength'])
            self.add_noise(self.params['noise_level'])
            self.normalize(self.params['z_min'], self.params['z_max'])
            self.data = self.data.astype(int)

    def normalize(self, min_val, max_val):
        current_min = np.min(self.data)
        current_max = np.max(self.data)
        current_min_base = np.min(self.params['Val_interval'])
        current_max_base = np.max(self.params['Val_interval'])
        self.data = min_val + (self.data - current_min) * (max_val - min_val) / (current_max - current_min)
        self.signal = min_val + (self.params['Val_interval'] - current_min_base) * (max_val - min_val) / (
                    current_max_base - current_min_base)

    def add_outliers(self, n_outliers, strength):
        indices = np.random.choice(len(self.time), n_outliers, replace=False)
        direction = np.random.choice([-1, 1], n_outliers)
        outlier_values = strength * np.max(np.abs(self.data)) * direction
        self.data[indices] += outlier_values

    def add_noise(self, noise_level):
        noise = noise_level * np.max(np.abs(self.data)) * np.random.normal(size=len(self.time))
        self.data += noise

    def plot(self, ax):
        if self.data is not None:
            ax.clear()
            ax.set_title(f"{self.__class__.__name__} Data")
            ax.plot(self.time, self.data, 'b-')
            ax.plot(self.params['T_interval'],
                    self.signal,
                    'rx')
            ax.plot(self.params['T_interval'], self.signal, 'r--', linewidth=1, alpha=0.7)
            ax.set_ylim(0, 1023)
            ax.grid(True)
            return ax

class BloodSensor(DataGenerator):
    def __init__(self):
        super().__init__()
        self.data = None
        self.params = {
            'base_I': 7.0,
            'points': 1000,
            'Vref': 5.0,
            'N': 12,
            'k_det': 0.5,
            'shift': 1600,
            'n_outliers': 5,
            'strength': 1.5,
            'noise_level': 0.01,
            't_min': 0.0,
            't_max': 10.0,
            'T_interval': None,
            'Val_interval': None
        }
        self.def_params = {
            'base_I': 7.0,
            'points': 1000,
            'shift': 1600,
            'n_outliers': 5,
            'noise_level': 0.05,
            't_min': 0.0,
            't_max': 10.0,
            'T_interval': [-1],
            'Val_interval': [-1]
        }
        self.signal = None
        self.time = None
    def configurate(self, base_I=None,points=None,shift=None, n_outliers=None, noise_level=None,t_min=None,t_max=None,T_interval=None,Val_interval=None):
        if base_I is not None: self.params['base_I'] = base_I
        if points is not None: self.params['points'] = points
        if shift is not None: self.params['shift'] = shift
        if n_outliers is not None: self.params['n_outliers'] = n_outliers
        if noise_level is not None: self.params['noise_level'] = noise_level
        if points is not None: self.params['points'] = points
        if t_min is not None: self.params['t_min'] = t_min
        if t_max is not None: self.params['t_max'] = t_max
        if T_interval is not None: self.params['T_interval'] = T_interval
        if Val_interval is not None: self.params['Val_interval'] = Val_interval
    def generate(self):
        if self.params['T_interval'] is not None and self.params['Val_interval'] is not None and len(
                self.params['T_interval']) >= 2 and len(self.params['Val_interval']) >= 2:
            if len(self.params['T_interval']) != len(self.params['Val_interval']):
                raise ValueError("T_interval и Val_interval должны иметь одинаковую длину")
            self.data, self.time = create_dynamix(np.array(self.params['T_interval']),
                                                  np.array(self.params['Val_interval']), self.params['t_min'],
                                                  self.params['t_max'], self.params['points'])
            self.add_outliers(self.params['n_outliers'], self.params['strength'])
            self.add_noise(self.params['noise_level'])
            min_offset, max_offset, offset = self.I_to_Z(self.params['base_I'])-self.params['shift'], self.I_to_Z(self.params['base_I'])+self.params['shift'], self.I_to_Z(self.params['base_I'])
            self.normalize(min_offset, max_offset,offset)
            self.data = self.data.astype(int)
            #student:
            vst = [z*5/(2**12) for z in self.data]
            Ist = [v/0.5 for v in vst]
            plt.figure()
            plt.plot(range(len(Ist)),Ist, range(len(Ist)), np.ones(len(Ist))*7)
            plt.show()




    def normalize(self, min_val, max_val, offset):
        current_min = np.min(self.data)
        current_max = np.max(self.data)

        normalized = 2 * (self.data - current_min) / (current_max-current_min) - 1
        amplitude = min(offset - min_val, max_val - offset)
        self.data = offset + amplitude * normalized


    def add_outliers(self, n_outliers, strength):
        indices = np.random.choice(len(self.time), n_outliers, replace=False)
        direction = np.random.choice([-1, 1], n_outliers)
        outlier_values = strength * np.max(np.abs(self.data)) * direction
        self.data[indices] += outlier_values

    def add_noise(self, noise_level):
        noise = noise_level * np.max(np.abs(self.data)) * np.random.normal(size=len(self.time))
        self.data += noise
    def I_to_Z(self,i):
        return 2**self.params['N']*self.params['k_det']*i/self.params['Vref']
    def Z_to_I(self,z):
        return z*self.params['Vref']/(2**self.params['N']*self.params['k_det'])
    def plot(self, ax):
        if self.data is not None:
            ax.clear()
            ax.plot(self.data)
            ax.set_title(f"{self.__class__.__name__} Data")
            ax.set_ylim(0,4095)
            return ax

class PPGSensor(DataGenerator):
    def __init__(self):
        super().__init__()
        self.data = None
        self.data_660 = None
        self.data_940 = None
        self.params = {
            'SpO2': 99,
            'N': 16,
            'Vref': 5,
            'points': 1000,
            'A': 110,
            'B': 25
        }
        self.def_params = {
            'SpO2': 99,
            'points': 1000,
        }
        self.signal = None
        self.time = None
        self.result = None
    def configurate(self,SpO2,points):
        if SpO2 is not None: self.params['SpO2'] = SpO2
        if points is not None: self.params['points'] = points
    def generate(self):
        DC660,DC940 = 1.3, 1.5
        AC660 = 0.6
        AC940 = (AC660*self.params['B']*DC940)/(DC660*(self.params['A']-self.params['SpO2']))
        print(f"AC940 = {AC940} AC660 = {AC660}")
        print(f"R = {(AC660/DC660)/(AC940/DC940)}")
        t, V660 = create_pulse_wave(Amp=AC660,zero_offset=DC660,points=self.params['points'],inverse=True) #need think
        t, V940 = create_pulse_wave(Amp=AC940,zero_offset=DC940,points=self.params['points'],inverse=False)
        self.time = t
        self.data_660 = v_in_z(V660,self.params['N'],self.params['Vref'])
        self.data_660 = [max(0, min(x, 2**self.params['N'])) for x in  self.data_660]
        self.data_940 = v_in_z(V940,self.params['N'],self.params['Vref'])
        self.data_940 = [max(0, min(x, 2**self.params['N'])) for x in self.data_940]

        ac660first = (max(self.data_940 - np.mean(self.data_940))-min(self.data_940 - np.mean(self.data_940)))/2
        ac940second = (max(self.data_660 - np.mean(self.data_660))-min(self.data_660 - np.mean(self.data_660)))/2
        print(f"AC940* = {z_in_v(ac660first,self.params['N'],self.params['Vref'])}, AC660* = {z_in_v(ac940second,self.params['N'],self.params['Vref'])}")
        print(f"DC660* = {z_in_v(np.mean(self.data_660),self.params['N'],self.params['Vref'])} DC940* = {z_in_v(np.mean(self.data_940),self.params['N'],self.params['Vref'])}")
        print(f"DC660 = {DC660} DC940 = {DC940}")
        R = (ac940second/np.mean(self.data_660))/(ac660first/np.mean(self.data_940))
        print(f"R* = {R}")
        self.result = self.params['A']-self.params['B']*R
        self.data = np.column_stack((self.data_660,self.data_940)) #1st col - 660 2nd col - 940
        '''self.time = np.linspace(0, 5, self.params['points'])
        DC_660 = self.U_to_Z(1)
        DС_940 = ((self.params['A']-self.params['SpO2'])*self.U_to_Z(0.45))/(self.params['B']*(self.U_to_Z(0.55)/DC_660))
        Z_660 = self.U_to_Z(1-0.1)/2*np.sin(2*np.pi*1.2*self.time)+self.U_to_Z(0.55)
        Z_940 = self.U_to_Z(1-0.1-0.1)/2*np.sin(2*np.pi*1.2*self.time)+self.U_to_Z(0.45)
        zero_crossings_660 = np.where(np.diff(np.sign(Z_660 - self.U_to_Z(0.55))))[0]
        zero_crossings_940 = np.where(np.diff(np.sign(Z_940 - self.U_to_Z(0.45))))[0]
        window_size = 50
        half_window = window_size // 2
        cnt = 0
        for i in zero_crossings_660:
            if cnt%2==1:
                start = max(i - half_window, 0)
                end = min(i + half_window, len(Z_660))
                x = np.linspace(0, 2 * np.pi, end - start)
                Z_660[start:end] = -self.U_to_Z(0.1) * np.sin(x) + self.U_to_Z(0.55)
            cnt += 1
        cnt = 0
        for j in zero_crossings_940:
            if cnt%2==1:
                start = max(j - half_window, 0)
                end = min(j + half_window, len(Z_660))
                x = np.linspace(0, 2 * np.pi, end - start)
                Z_940[start:end] = -self.U_to_Z(0.1) * np.sin(x) + self.U_to_Z(0.45)
            cnt += 1
        Z_660 += DC_660
        Z_940 += DС_940
        noise_660 = np.random.normal(0, 1, 1000)
        noise_940 = np.random.normal(0, 1, 1000)
        Z_660 += 10*noise_660
        Z_940 += 10*noise_940
        self.data_660 = Z_660
        self.data_940 = Z_940
        self.data = np.column_stack((Z_660, Z_940))'''
        #print(f"DC_660={DC_660} DС_940={DС_940}")

        #data = np.loadtxt('qwe.txt', dtype=int)
        #Z_660 = data[:, 0]
        #Z_940 = data[:, 1]
        # print(f"{Z_660[0]} {Z_940[0]}")
        #DC_660 = np.mean(Z_660)
        #DC_940 = np.mean(Z_940)
        # print(f"{DC_660} {DC_940}")
        #AC_660 = Z_660 - DC_660
        #AC_940 = Z_940 - DC_940
        #AC_660 = (np.abs(np.max(AC_660)) + np.abs(np.min(AC_660))) / 2
        #AC_940 = (np.abs(np.max(AC_940)) + np.abs(np.min(AC_940))) / 2
        # print(f"{AC_660} {AC_940}")
        #DC_660 -= AC_660
        #DC_940 -= AC_940
        #R = (AC_660 / DC_660) / (AC_940 / DC_940)
        #print(f"{R}")
        #A, B = 110, 25
        #SpO2 = A - B * R
        #print(f"SpO2 = {SpO2}%")
    def plot(self,ax):
        if hasattr(self, 'data_660') and hasattr(self, 'data_940'):
            ax.clear()
            ax.plot(self.time, self.data_660, label='Z660', color='blue')
            ax.plot(self.time, self.data_940, label='Z940', color='green')
            ax.set_title(f"{self.__class__.__name__} Data")
            ax.set_xlabel("Время (с)")
            ax.set_ylabel("Z")
            ax.legend(loc='upper right')
            #ax.set_ylim(0,65535)
            print(f"task: {self.params['SpO2']}")
            print(f"result: {self.result}")
            return ax

class HeartRateSensor(DataGenerator):
    def __init__(self):
        super().__init__()
        self.data = None
        self.params = {
            'duration':6,
            'HeartRate':80,
            'noise_lvl': 0.1,
        }
        self.def_params = {
            'duration':6,
            'HeartRate':80,
            'noise_lvl': 0.1,
        }
        self.signal = None
        self.time = None
    def configurate(self, duration,HeartRate,noise_lvl):
        if duration is not None: self.params['duration'] = duration
        if HeartRate is not None: self.params['HeartRate'] = HeartRate
        if noise_lvl is not None: self.params['noise_lvl'] = noise_lvl

    def generate(self):
        ecg_signal = nk.ecg_simulate(
            duration=self.params['duration'],
            heart_rate=self.params['HeartRate'],
            sampling_rate=1000,
            noise=self.params['noise_lvl']
        )
        ecg_signal = ecg_signal[100:]
        time = np.linspace(1,5,1000*self.params['duration'])[100:]
        self.data = scale_signal(ecg_signal, 0.1, 3)
        self.time = time

    def plot(self, ax):
        if self.data is not None:
            ax.clear()
            ax.plot(self.time,self.data)
            ax.set_title(f"{self.__class__.__name__} Data")
            ax.set_xlabel("Время (с)")
            ax.set_ylabel("Напряжение (В)")
            return ax

class pHSensor(DataGenerator):
    def __init__(self):
        super().__init__()
        self.data = None
        self.params = {
            'points': 1000,
            'diration': 10,
            'tr_time': 0.5,
            'N': 12,
            'Vref': 5,
            'T': 298,
            'F': 96485,
            'R': 8.314,
            'T_interval': None,
            'Val_interval': None,
            'noise_lvl': 0.1,
            'n_outliers': 5,
            'strength':0.3
        }
        self.def_params = {
            'points': 1000,
            'duration': 10,
            'T_interval': [-1],
            'Val_interval': [-1],
            'noise_lvl': 0.1,
            'n_outliers': 5,
            'strength':0.3
        }
        self.signal = None
        self.time = None
    def configurate(self,points,duration, T_interval, Val_interval,noise_lvl,n_outliers,strength):
        if points is not None: self.params['points'] = points
        if duration is not None: self.params['duration'] = duration
        if T_interval is not None: self.params['T_interval'] = T_interval
        if Val_interval is not None: self.params['Val_interval'] = Val_interval
        if n_outliers is not None: self.params['n_outliers'] = n_outliers
        if noise_lvl is not None: self.params['noise_lvl'] = noise_lvl
        if strength is not None: self.params['strength'] = strength
    def generate(self):
        if self.params['T_interval'] is not None and self.params['Val_interval'] is not None and len(
                self.params['T_interval']) >= 2 and len(self.params['Val_interval']) >= 2:
            if len(self.params['T_interval']) != len(self.params['Val_interval']):
                raise ValueError("T_interval и Val_interval должны иметь одинаковую длину")
            self.data, self.time = create_dynamix(np.array(self.params['T_interval']),np.array(self.params['Val_interval']),0, self.params['duration'],self.params['points'])
            self.data = scale_signal(self.data,
                                     v_in_z(self.pH_to_V(0.1), self.params['N'], self.params['Vref'], bip=True),
                                     v_in_z(self.pH_to_V(14), self.params['N'], self.params['Vref'], bip=True))
            self.add_noise(self.params['noise_lvl'])
            self.add_outliers(self.params['n_outliers'],self.params['strength'])
            if max(self.data) > 2**self.params['N']-1 or min(self.data) < 0:
                self.data = np.clip(self.data, 0, 2**self.params['N']-1)
    def plot(self,ax):
        if self.data is not None:
            ax.clear()
            ax.plot(self.time,self.data,'k',linewidth=1.5)
            ax.plot(self.time,v_in_z(self.pH_to_V(0.1),self.params['N'], self.params['Vref'], bip=True)*np.ones_like(self.time), '--',label= 'pH = 0',alpha=0.3)
            ax.plot(self.time,v_in_z(self.pH_to_V(7),self.params['N'], self.params['Vref'], bip=True)*np.ones_like(self.time), '--',label= 'pH = 7',alpha=0.3)
            ax.plot(self.time,v_in_z(self.pH_to_V(14),self.params['N'], self.params['Vref'], bip=True)*np.ones_like(self.time), '--',label= 'pH = 14',alpha=0.3)
            ax.set_title(f"{self.__class__.__name__} Data")
            ax.set_ylim([0,4096])
            ax.legend()
            return ax
    def add_outliers(self, n_outliers, strength):
        indices = np.random.choice(len(self.time), n_outliers, replace=False)
        direction = np.random.choice([-1, 1], n_outliers)
        outlier_values = strength * np.max(np.abs(self.data)) * direction
        self.data[indices] += outlier_values

    def add_noise(self, noise_level):
        noise = noise_level * np.max(np.abs(self.data)) * np.random.normal(size=len(self.time))
        self.data += noise
    def pH_to_V(self, pH):
        return -(self.params['R']*self.params['T']*2.303*pH)/self.params['F']
    def V_to_pH(self, V):
        return -(self.params['F']*V)/(2.303*self.params['R']*self.params['T'])


class LiquidLvlSensor(DataGenerator):
    def __init__(self):
        super().__init__()
        self.data = None
        self.params = {
            'num_sens': 5,
            'points': 50,
            'max_v': 5.0,
            'min_v': (0, 0.2),
            'err_prob': 0.05
        }
        self.def_params = {
            'num_sens': 5,
            'points': 50,
            'err_prob': 0.05
        }
        self.signal = None
        self.time = None
    def configurate(self,num_sens,points,err_prob):
        if num_sens is not None: self.params['num_sens'] = num_sens
        if points is not None: self.params['points'] = points
        if err_prob is not None: self.params['err_prob'] = err_prob
    def generate(self):
        num_sensors = self.params['num_sens']
        num_states = self.params['points']
        max_voltage = self.params['max_v']
        low_voltage_range = self.params['min_v']
        error_probability = self.params['err_prob']
        voltage_matrix = np.zeros((num_states, num_sensors))
        for sensor in range(num_sensors):
            # Определяем момент перехода к высоким значениям
            transition_point = int(num_states * (sensor + 1) / (num_sensors + 1))
            for state in range(num_states):
                # На начальном этапе уровень жидкости будет колебаться от 0 до 0.2
                if state < transition_point:
                    voltage_matrix[state, sensor] = np.random.uniform(*low_voltage_range)
                else:
                    # После перехода устанавливаем уровень в диапазоне от 4.8 до 5.0
                    voltage_matrix[state, sensor] = np.random.uniform(4.8, max_voltage)
                    # Добавляем случайные ошибки только в этом диапазоне
                    if np.random.rand() < error_probability:
                        voltage_matrix[state, sensor] = np.random.uniform(*low_voltage_range)
        self.signal = voltage_matrix
        self.data = np.column_stack((range(1, num_states+1, 1), voltage_matrix))
        self.time = range(num_states)
    def plot(self, ax):
        if self.signal is not None:
            ax.clear()
            for sensor in range(self.params['num_sens']):
                ax.plot(self.signal[:,sensor],label=f'№{sensor + 1}')
            ax.set_title(f"{self.__class__.__name__} Data")
            ax.legend(loc='lower center')
            return ax

class BubbleSensor(DataGenerator):
    def __init__(self):
        super().__init__()
        self.data = None
        self.Trecv = None
        self.Tsend = None
        self.delt_v = None
        self.state = None
        self.params = {
            'points': 1000,
            'd': 0.1,
            'Vliq': 1.400,
            'time_step': 1,
            'T_interval': None,
            'Val_interval': None
        }
        self.def_params = {
            'points': 1000,
            'time_step': 1,
            'T_interval': [-1],
            'Val_interval': [-1]
        }
        self.signal = None
        self.time = None
    def configurate(self,points,time_step,T_interval,Val_interval):
        if points is not None: self.params['points'] = points
        if time_step is not None: self.params['time_step'] = time_step
        if T_interval is not None: self.params['T_interval'] = T_interval
        if Val_interval is not None: self.params['Val_interval'] = Val_interval
    def generate(self):
        self.Tsend, self.Trecv, self.state, self.delt_v, time_new = bubbleCreate(
            points=self.params['points'],
            d=self.params['d'],
            time_step=self.params['time_step'],
            vliq = self.params['Vliq'],
            time= np.array(self.params['T_interval']),
            value = np.array(self.params['Val_interval'])
        )
        plt.figure()
        plt.plot(self.Tsend)
        plt.plot(self.Trecv)
        plt.show()

    def plot(self,ax):
        if (self.Trecv is not None) and (self.Tsend is not None):
            ax.clear()
            ax.plot(self.delt_v)
            ax.plot((self.params['d']/self.params['Vliq'])*np.ones_like(self.delt_v))
            ax.plot(self.state)
            ax.set_title(f"{self.__class__.__name__} Data")
            """plt.figure()
            plt.plot(self.delt_v)
            plt.show()"""
            return ax

class GeigerSensor(DataGenerator):
    def __init__(self):
        super().__init__()
        self.data = None
        self.params = {
            'points': 60,
            'T_interval': None,
            'Val_interval': None
        }
        self.def_params = {
            'points': 60,
            'T_interval': [-1],
            'Val_interval': [-1]
        }
        self.signal = None
        self.time = None
    def configurate(self,points=None, T_interval=None, Val_interval=None):
        if points is not None: self.params['points'] = points
        if T_interval is not None: self.params['T_interval'] = T_interval
        if Val_interval is not None: self.params['Val_interval'] = Val_interval

    def generate(self):
        if self.params['T_interval'] is not None and self.params['Val_interval'] is not None and  len(self.params['T_interval']) >= 2 and len(self.params['Val_interval']) >= 2:
            if len(self.params['T_interval']) != len(self.params['Val_interval']):
                raise ValueError("T_interval и Val_interval должны иметь одинаковую длину")
            #interp
            t_sensor, binary, t_freq, freq_values = generate_geiger_data(
                time_points=np.array(self.params['T_interval']),
                frequency_values=np.array(self.params['Val_interval']),
                total_time=60,
                sampling_rate=1000,
                k=10
            )
            self.data = binary
            self.time = t_sensor
            self.signal = dict(zip(t_freq,freq_values/10))
        else:
            #random
            p = np.random.randint(0, 95)/100
            arr = np.random.choice([0, 1], size=self.params['points'], p=[1 - p, p])
            self.data = arr
            self.time = np.linspace(0,60,self.params['points'])
    def plot(self,ax):
        if self.data is not None:
            ax.clear()
            if self.signal is None:
                ax.plot(self.time,self.data)
            else:
                ax.plot(self.time, self.data,alpha=0.5)
                ax.plot(*zip(*self.signal.items()), 'r-')
            ax.set_title(f"{self.__class__.__name__} Data")
            return ax

class BloodPressureSensor(DataGenerator):
    def __init__(self):
        super().__init__()
        self.data = None
        self.params = {
            'duration': 10,
            'sampling_rate': 1000,
            'SBP': 120,
            'DBP': 80,
            'Vref': 6.0,
            'N': 12,
            'k1': 30,
            'k2': 25
        }
        self.def_params = {
            'SBP': 120,
            'DBP': 80,
        }
        self.signal = None
        self.time = None
    def configurate(self,SBP,DBP):
        if SBP is not None: self.params['SBP'] = SBP
        if DBP is not None: self.params['DBP'] = DBP
    def generate(self):
        start_pressure = self.params['DBP'] / self.params['k2']
        end_pressure = self.params['SBP'] / self.params['k1']
        initial_pressure = 5.8
        final_pressure = 0.2
        np.random.seed(0)
        duration = self.params['duration']
        sampling_rate = self.params['sampling_rate']
        num_points = int(duration * sampling_rate)
        time = np.linspace(0, duration, num_points)
        pressure_drop = np.linspace(initial_pressure, final_pressure, num_points)
        pulse_frequency = 1.8
        pulse_amplitude = 0.5
        for i in range(len(time)):
            if start_pressure <= pressure_drop[i] <= end_pressure:
                pressure_drop[i] += pulse_amplitude * np.sin(2 * np.pi * pulse_frequency * time[i])
        noise = np.random.normal(0, 0.02, pressure_drop.shape)
        pressure_drop += noise
        Q1 = np.percentile(pressure_drop, 25)
        Q3 = np.percentile(pressure_drop, 75)
        IQR = Q3 - Q1
        filter_mask = (pressure_drop >= (Q1 - 1.5 * IQR)) & (pressure_drop <= (Q3 + 1.5 * IQR))
        filtered_pressure = pressure_drop[filter_mask]
        peaks, _ = find_peaks(filtered_pressure)
        troughs, _ = find_peaks(-filtered_pressure)
        valid_peaks = peaks[(filtered_pressure[peaks] >= start_pressure) & (filtered_pressure[peaks] <= end_pressure)]
        valid_troughs = troughs[(filtered_pressure[troughs] >= start_pressure) & (filtered_pressure[troughs] <= end_pressure)]
        systolic_pressure = 0
        diastolic_pressure = 0
        Z = [int(self.V_to_Z(value)) for value in filtered_pressure]
        self.data = Z
        delta = 50
        val1,val2 = 0, 0
        dZ = np.gradient(Z)
        for i in range(0,len(Z)-1,1):
            if abs(dZ[i+1]-dZ[i])>delta:
                systolic_pressure = z_in_v(Z[i], self.params['N'], self.params['Vref'])
                val1 = i
                break
        for i in range(len(Z)-1, 0, -1):
            if abs(dZ[i-1]-dZ[i])>delta:
                diastolic_pressure = z_in_v(Z[i], self.params['N'], self.params['Vref'])
                val2 = i
                break
        print(f"syst: {systolic_pressure * self.params['k1']}")
        print(f"diast: {diastolic_pressure*self.params['k2']}")
    def V_to_Z(self,v):
        return 2**self.params['N']*v/self.params['Vref']
    def plot(self,ax):
        if self.data is not None:
            ax.clear()
            ax.plot(self.data)
            ax.set_title(f"{self.__class__.__name__} Data")
            return ax

class ConsSensor(DataGenerator):
    def __init__(self):
        super().__init__()
        self.data = None
        self.params = {
            'points': 100,
            'T_interval': None,
            'Val_interval': None
        }
        self.def_params = {
            'points': 100,
            'T_interval': [-1],
            'Val_interval': [-1]
        }
        self.signal = None
        self.time = None
    def configurate(self,points=None, T_interval=None, Val_interval=None):
        if points is not None: self.params['points'] = points
        if T_interval is not None: self.params['T_interval'] = T_interval
        if Val_interval is not None: self.params['Val_interval'] = Val_interval

    def generate(self):
        if self.params['T_interval'] is not None and self.params['Val_interval'] is not None and len(
                self.params['T_interval']) >= 2 and len(self.params['Val_interval']) >= 2:
            if len(self.params['T_interval']) != len(self.params['Val_interval']):
                raise ValueError("T_interval и Val_interval должны иметь одинаковую длину")
            # interp
            t_sensor, binary, t_freq, freq_values = generate_geiger_data(
                time_points=np.array(self.params['T_interval']),
                frequency_values=np.array(self.params['Val_interval']),
                total_time=60,
                sampling_rate=1000,
                k=10
            )
            self.data = binary
            self.time = t_sensor
            self.signal = dict(zip(t_freq, freq_values / 10))
        else:
            # random
            p = np.random.randint(0, 95) / 100
            arr = np.random.choice([0, 1], size=self.params['points'], p=[1 - p, p])
            self.data = arr
            self.time = np.linspace(0, 60, self.params['points'])

    def plot(self, ax):
        if self.data is not None:
            ax.clear()
            if self.signal is None:
                ax.plot(self.time, self.data)
            else:
                ax.plot(self.time, self.data, alpha=0.5)
                ax.plot(*zip(*self.signal.items()), 'r-')
            ax.set_title(f"{self.__class__.__name__} Data")
            return ax

class NitrateSensor(DataGenerator):
    def __init__(self):
        super().__init__()
        self.data = {}
        self.params = {
            'points': 100,
            'Tomatoes': None,
            'Spinach': None,
            'Beet': None,
            'Cabbage': None,
            'Carrot': None,
            'Potato': None,
            'Cucumbers': None
        }
        self.def_params = {
            'points': 100,
            'Tomatoes': 1,
            'Spinach': 1,
            'Beet': 1,
            'Cabbage': 1,
            'Carrot': 1,
            'Potato': 1,
            'Cucumbers': 1
        }
        self.signal = None
        self.time = None
    def configurate(self,points,Tomatoes,Spinach,Beet,Cabbage,Carrot,Potato,Cucumbers):
        if points is not None: self.params['points'] = points
        if Tomatoes is not None: self.params['Tomatoes'] = Tomatoes
        if Spinach is not None: self.params['Spinach'] = Spinach
        if Beet is not None: self.params['Beet'] = Beet
        if Cabbage is not None: self.params['Cabbage'] = Cabbage
        if Carrot is not None: self.params['Carrot'] = Carrot
        if Potato is not None: self.params['Potato'] = Potato
        if Cucumbers is not None: self.params['Cucumbers'] = Cucumbers

    def generate(self):
        for product in ['Tomatoes','Spinach','Beet','Cabbage','Carrot','Potato','Cucumbers']:
            state = None
            match self.params[product]:
                case 0:
                    state = 'increase'
                case 1:
                    state = 'normal'
                case 2:
                    state = 'decrease'
            self.data[product] = createNitrate(product,self.params['points'],state)
        #print(self.data)
    def plot(self,ax):
        if self.data is not None:
            ax.clear()
            for product, values in self.data.items():
                ax.plot(values, label=product, marker='.')
                ax.legend(loc='upper left',framealpha=1)
            ax.set_title(f"{self.__class__.__name__} Data")
            return ax
class GlucozeSensor(DataGenerator):
    def __init__(self):
        super().__init__()
        self.data = None
        self.val = None
        self.time = None
        self.params = {
            'points': 1000
        }
        self.def_params = {
            'points': 1000
        }
    def configurate(self, points):
        if points is not None: self.params['points'] = points
    def generate(self):
        self.data, self.val, self.time = createGlucoza(points=self.params['points'])
    def plot(self,ax):
        if self.data is not None:
            ax.clear()
            ax.plot(self.time,self.val)
            ax.set_title(f"{self.__class__.__name__} Data")
            return ax
class Capnograph(DataGenerator):
    def __init__(self):
        super().__init__()
        self.data = None
        self.params = {
            'n_outliers': 5,
            'strength': 1.5,
            'noise_level': 0.01,
            'points': 1000,
            'z_min': 0,
            'z_max': 1023,
            't_min': 0.0,
            't_max': 10.0,
            'N': 10,
            'T_interval': None,
            'Val_interval': None
        }
        self.def_params = {
            'n_outliers': 5,
            'noise_level': 0.05,
            'points': 1000,
            't_min': 0.0,
            't_max': 10.0,
            'T_interval': [-1],
            'Val_interval': [-1]
        }
        self.signal = None
        self.time = None

    def configurate(self, n_outliers=None, noise_level=None, points=None, t_min=None, t_max=None, T_interval=None, Val_interval=None):
        if n_outliers is not None: self.params['n_outliers'] = n_outliers
        if noise_level is not None: self.params['noise_level'] = noise_level
        if points is not None: self.params['points'] = points
        if t_min is not None: self.params['t_min'] = t_min
        if t_max is not None: self.params['t_max'] = t_max
        if T_interval is not None: self.params['T_interval'] = T_interval
        if Val_interval is not None: self.params['Val_interval'] = Val_interval

    def generate(self):
        if self.params['T_interval'] is not None and self.params['Val_interval'] is not None and len(
                self.params['T_interval']) >= 2 and len(self.params['Val_interval']) >= 2:
            if len(self.params['T_interval']) != len(self.params['Val_interval']):
                raise ValueError("T_interval и Val_interval должны иметь одинаковую длину")
            self.data, self.time = create_dynamix(np.array(self.params['T_interval']),
                                                  np.array(self.params['Val_interval']), self.params['t_min'],
                                                  self.params['t_max'], self.params['points'])
            self.add_outliers(self.params['n_outliers'], self.params['strength'])
            self.add_noise(self.params['noise_level'])
            self.normalize(self.params['z_min'], self.params['z_max'])
            self.data = self.data.astype(int)

    def normalize(self, min_val, max_val):
        current_min = np.min(self.data)
        current_max = np.max(self.data)
        current_min_base = np.min(self.params['Val_interval'])
        current_max_base = np.max(self.params['Val_interval'])
        self.data = min_val + (self.data - current_min) * (max_val - min_val) / (current_max - current_min)
        self.signal = min_val + (self.params['Val_interval'] - current_min_base) * (max_val - min_val) / (
                    current_max_base - current_min_base)

    def add_outliers(self, n_outliers, strength):
        indices = np.random.choice(len(self.time), n_outliers, replace=False)
        direction = np.random.choice([-1, 1], n_outliers)
        outlier_values = strength * np.max(np.abs(self.data)) * direction
        self.data[indices] += outlier_values

    def add_noise(self, noise_level):
        noise = noise_level * np.max(np.abs(self.data)) * np.random.normal(size=len(self.time))
        self.data += noise

    def plot(self, ax):
        if self.data is not None:
            ax.clear()
            ax.set_title(f"{self.__class__.__name__} Data")
            ax.plot(self.time, self.data, 'b-')
            ax.plot(self.params['T_interval'],
                    self.signal,
                    'rx')
            ax.plot(self.params['T_interval'], self.signal, 'r--', linewidth=1, alpha=0.7)
            ax.set_ylim(0, 1023)
            ax.grid(True)
            return ax

class ConductivitySensor(DataGenerator):
    def __init__(self):
        super().__init__()
        self.data = None
        self.params = {
            'n_outliers': 5,
            'strength': 1.5,
            'noise_level': 0.01,
            'points': 1000,
            'z_min': 0,
            'z_max': 65535,
            't_min': 0.0,
            't_max': 10.0,
            'N': 16,
            'T_interval': None,
            'Val_interval': None
        }
        self.def_params = {
            'n_outliers': 5,
            'noise_level': 0.05,
            'points': 1000,
            't_min': 0.0,
            't_max': 10.0,
            'T_interval': [-1],
            'Val_interval': [-1]
        }
        self.signal = None
        self.time = None

    def configurate(self, n_outliers=None, noise_level=None, points=None, t_min=None, t_max=None, T_interval=None, Val_interval=None):
        if n_outliers is not None: self.params['n_outliers'] = n_outliers
        if noise_level is not None: self.params['noise_level'] = noise_level
        if points is not None: self.params['points'] = points
        if t_min is not None: self.params['t_min'] = t_min
        if t_max is not None: self.params['t_max'] = t_max
        if T_interval is not None: self.params['T_interval'] = T_interval
        if Val_interval is not None: self.params['Val_interval'] = Val_interval

    def generate(self):
        if self.params['T_interval'] is not None and self.params['Val_interval'] is not None and len(
                self.params['T_interval']) >= 2 and len(self.params['Val_interval']) >= 2:
            if len(self.params['T_interval']) != len(self.params['Val_interval']):
                raise ValueError("T_interval и Val_interval должны иметь одинаковую длину")
            self.data, self.time = create_dynamix(np.array(self.params['T_interval']),
                                                  np.array(self.params['Val_interval']), self.params['t_min'],
                                                  self.params['t_max'], self.params['points'])
            self.add_outliers(self.params['n_outliers'], self.params['strength'])
            self.add_noise(self.params['noise_level'])
            self.normalize(self.params['z_min'], self.params['z_max'])
            self.data = self.data.astype(int)

    def normalize(self, min_val, max_val):
        current_min = np.min(self.data)
        current_max = np.max(self.data)
        current_min_base = np.min(self.params['Val_interval'])
        current_max_base = np.max(self.params['Val_interval'])
        self.data = min_val + (self.data - current_min) * (max_val - min_val) / (current_max - current_min)
        self.signal = min_val + (self.params['Val_interval'] - current_min_base) * (max_val - min_val) / (
                    current_max_base - current_min_base)

    def add_outliers(self, n_outliers, strength):
        indices = np.random.choice(len(self.time), n_outliers, replace=False)
        direction = np.random.choice([-1, 1], n_outliers)
        outlier_values = strength * np.max(np.abs(self.data)) * direction
        self.data[indices] += outlier_values

    def add_noise(self, noise_level):
        noise = noise_level * np.max(np.abs(self.data)) * np.random.normal(size=len(self.time))
        self.data += noise

    def plot(self, ax):
        if self.data is not None:
            ax.clear()
            ax.set_title(f"{self.__class__.__name__} Data")
            ax.plot(self.time, self.data, 'b-')
            ax.plot(self.params['T_interval'],
                    self.signal,
                    'rx')
            ax.plot(self.params['T_interval'], self.signal, 'r--', linewidth=1, alpha=0.7)
            ax.set_ylim(0, 65535)
            ax.grid(True)
            return ax
