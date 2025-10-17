from random import randint

from func import *
import random
from scipy.integrate import trapezoid
from scipy import interpolate
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk

a = np.array([1,2,3])
b = np.array([-1,-2,-3])
data_to_save = np.hstack((a, b))
print(data_to_save)
#debugging errors
"""error_message = (
                f"Произошла ошибка:\n"
                f"• Тип: {type(e).__name__}\n"
                f"• Сообщение: {str(e)}\n\n"
                f"Стек вызовов:\n{traceback.format_exc()}"
            )
            QtWidgets.QMessageBox.critical(self, "Ошибка генерации", error_message)"""
#converting \n\r -> None
'''qwe = ""
with open("repl.txt",'r') as file:
    qwe = file.read()
qwe = qwe.replace('\n', '').replace('\r', '').replace(' ','')
print(qwe)'''