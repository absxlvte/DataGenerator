#python -m PyQt5.uic.pyuic -x des.ui -o des.py
import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from Generators import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from des import Ui_MainWindow
import traceback
import sys

class FPIBS_Generator(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        #Generators
        self.manager = GeneratorManager()
        self.current_generator_name = None
        self.data = None
        #ComboBox
        self.Parameters.setCurrentIndex(0)
        self.GeneratorsList.blockSignals(True)
        self.GeneratorsList.setCurrentIndex(-1)
        self.GeneratorsList.blockSignals(False)
        self.GeneratorsList.currentIndexChanged.connect(self.switch_page)


        #Buttons
        self.button_Quit.pressed.connect(self.close)
        self.button_Reset.pressed.connect(self.reset_all_fields)
        self.button_Generate.pressed.connect(self.generate_values)
        self.button_setDefault.pressed.connect(self.set_default_values)
        self.button_Save.pressed.connect(self.save_to_txt)
        #Graphics
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout = QtWidgets.QVBoxLayout(self.Graphics)
        layout.addWidget(self.canvas)
        self.Ax = self.figure.add_subplot()
        #Checkbox
        self.isInterpolate.stateChanged.connect(self.toggle_interpolation_fields)
        self.toggle_interpolation_fields(False)


    def switch_page(self,index):
        if 0 <= index < self.Parameters.count()+1:
            self.Parameters.setCurrentIndex(index+1)
        else: self.Parameters.setCurrentIndex(0)
        self.current_generator_name = self.GeneratorsList.currentText()
        #print(self.current_generator_name)

    def reset_all_fields(self):
        current_page = self.Parameters.currentWidget()
        for widget in current_page.findChildren(QtWidgets.QTextEdit):
            widget.clear()
        for widget in current_page.findChildren(QtWidgets.QComboBox):
            widget.setCurrentIndex(1)
        self.figure.clear()
        self.Ax = self.figure.add_subplot()
        self.canvas.draw_idle()
    def generate_values(self):
        if not self.current_generator_name:
            QtWidgets.QMessageBox.warning(self,"Ошибка!", "Выберите задание")
            return
        params_widget = {}
        match self.current_generator_name:
            case 'Test - Sin()':
                if self.isInterpolate.isChecked():
                    params_widget = {
                        'T_interval': self.T_interval,
                        'Val_interval': self.Val_interval,
                        'points': self.Points,
                        't_min': self.T_min,
                        't_max': self.T_max
                    }
                else:
                    params_widget = {
                        'amp': self.Amp,
                        'freq': self.Freq,
                        'phase': self.Phase,
                        'offset': self.Offset,
                        'points': self.Points,
                        't_min': self.T_min,
                        't_max': self.T_max
                    }
            case 'Датчик температуры':
                if self.isInterpolate.isChecked():
                    params_widget = {
                        'n_outliers': self.n_out_temp,
                        'points': self.n_points_temp,
                        'noise_level': self.noiselvl_temp,
                        'T_interval': self.T_interval_temp,
                        'Val_interval': self.Val_interval_temp,
                        't_min': self.T_min_temp,
                        't_max': self.T_max_temp
                    }
            case 'Гидравлический датчик давления':
                if self.isInterpolate.isChecked():
                    params_widget = {
                        'n_outliers': self.n_out_hydr,
                        'points': self.n_points_hydr,
                        'noise_level': self.noiselvl_hydr,
                        'T_interval': self.T_interval_hydr,
                        'Val_interval': self.Val_interval_hydr,
                        't_min': self.T_min_hydr,
                        't_max': self.T_max_hydr
                    }
            case 'Датчик наличия крови':
                if self.isInterpolate.isChecked():
                    params_widget = {
                        'base_I': self.base_i,
                        'points': self.n_points_bl,
                        'shift': self.drop_val,
                        'n_outliers': self.n_out_bl,
                        'noise_level': self.noiselvl_bl,
                        't_min': self.T_min_bl,
                        't_max': self.T_max_bl,
                        'T_interval': self.T_interval_bl,
                        'Val_interval': self.Val_interval_bl
                    }
            case 'Датчик насыщения крови кислородом':
                params_widget = {
                    'SpO2': self.SpO2,
                    'points': self.n_points_4

                }
            case 'Датчик ЧСС':
                params_widget = {
                    'duration': self.hr_duration,
                    'HeartRate': self.heartRate,
                    'noise_lvl': self.hr_boise_lvl,
                }
            case 'Датчик pH':
                params_widget = {
                    'points': self.n_points_ph,
                    'duration': self.time_ph,
                    'T_interval': self.T_interval_ph,
                    'Val_interval': self.Val_interval_ph,
                    'noise_lvl': self.time_ph_2,
                    'n_outliers': self.time_ph_3,
                    'strength': self.strength_ph
                }
            case 'Датчик уровня жидкости':
                params_widget = {
                    'num_sens': self.n_ell,
                    'points': self.n_points_8,
                    'err_prob': self.err_prob
                }
            case 'Датчик наличия пузырьков':
                params_widget = {
                    'points': self.n_points_9,
                    'time_step': self.stepT,
                    'T_interval': self.T_interval_bubble,
                    'Val_interval': self.Val_interval_bubble
                }
            case 'Счетчик Гейгера':
                params_widget = {
                    'points': self.n_points_geig,
                    'T_interval': self.T_interval_geig,
                    'Val_interval': self.Val_interval_geig
                }
            case 'Датчик артериального давления':
                params_widget = {
                    'SBP': self.sbp,
                    'DBP': self.dbp
                }
            case 'Датчик расхода':
                params_widget = {
                    'points': self.n_points_cons,
                    'T_interval': self.t_interval_cons,
                    'Val_interval': self.Val_interval_cons
                }
            case 'Датчик нитратов':
                params_widget = {
                    'points': self.NitratePoints,
                    'Tomatoes': self.TomatoesState,
                    'Spinach': self.SpinachState,
                    'Beet': self.BeetState,
                    'Cabbage': self.CabbageState,
                    'Carrot': self.CarrotState,
                    'Potato': self.PotatoState,
                    'Cucumbers': self.CucumbersState
                }
            case 'Датчик глюкозы':
                params_widget = {
                    'points': self.GlukozaPoints
                }
            case 'Капнограф':
                params_widget = {
                    'n_outliers': self.n_out_capn,
                    'points': self.Points_cap,
                    'noise_level': self.noiselvl_capn,
                    'T_interval': self.T_interval_cap,
                    'Val_interval': self.Val_interval_cap,
                    't_min': self.T_min_cap,
                    't_max': self.T_max_cap
                }
            case 'Датчик проводимости':
                params_widget = {
                    'n_outliers': self.n_out_cond,
                    'points': self.Points_cond,
                    'noise_level': self.noiselvl_cond,
                    'T_interval': self.T_interval_cond,
                    'Val_interval': self.Val_interval_cond,
                    't_min': self.T_min_cond,
                    't_max': self.T_max_cond
                }
        params, errors = self.validate_inputs(params_widget)
        if errors:
            QtWidgets.QMessageBox.warning(self, "Ошибка ввода!","\n".join(errors))
            return
        try:
            generator = self.manager.set_generator(self.current_generator_name)
            generator.configurate(**params)
            generator.generate()
            generator.plot(self.Ax)
            self.canvas.draw_idle()
            self.data = generator.data
        except Exception as e:
            QtWidgets.QMessageBox.critical(self,"Ошибка генерации",f"ошибка: {str(e)}")

    def validate_inputs(self, params_widgets):
        errors = []
        params = {}
        int_params = [
            'points',
            'n_outliers',
            'shift',
            'SpO2',
            'HeartRate',
            'num_sens',
            'time_step',
            'duration'
        ]
        splitting_params = [
            'T_interval',
            'Val_interval'
        ]
        chosen_params = [
            'Tomatoes',
            'Spinach',
            'Beet',
            'Cabbage',
            'Carrot',
            'Potato',
            'Cucumbers'
        ]
        for param_name, widget in params_widgets.items():
            if not widget.isVisible():
                continue
            if param_name in chosen_params:
                value = widget.currentIndex()
            else:
                text = widget.toPlainText().strip()
                if not text:
                    errors.append(f"Поле <<{param_name}>> не заполнено")
                    continue
            try:
                if param_name in chosen_params:
                    params[param_name] = value
                    #print(f'value - {value}')
                    continue
                if param_name in int_params:
                    value = int(text)
                elif param_name in splitting_params:
                    values = list(map(float, text.split()))
                    if len(values) < 2:
                        errors.append(f"Для интерполяции нужно минимум 2 значения в поле <<{param_name}>>")
                        continue
                    value = values
                else:
                    value = float(text)
                params[param_name] = value
            except ValueError:
                errors.append(f"Некорректное значение в поле <<{param_name}>>")
        return params, errors
    def set_default_values(self):
        if not self.current_generator_name:
            QtWidgets.QMessageBox.warning(self,"Ошибка!", "Выберите задание")
            return
        try:
            generator = self.manager.set_generator(self.current_generator_name)
            generator.configurate(**generator.def_params)
            generator.generate()
            generator.plot(self.Ax)
            self.canvas.draw_idle()
            self.data = generator.data
        except Exception as e:
            QtWidgets.QMessageBox.critical(self,"Ошибка генерации",f"ошибка: {str(e)}")
    def save_to_txt_adv(self):
        if not hasattr(self,'data') or self.data is None:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Нечего сохранять")
            return
        if self.current_generator_name == 'Датчик нитратов':
            directory = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                "Выберите папку для сохранения",
                QtCore.QDir.currentPath(),
                QtWidgets.QFileDialog.ShowDirsOnly
            )
            if not directory:
                return
            for filename, values in self.data.items():
                safe_filename = f"{filename.replace(' ', '_')}.txt"
                file_path = QtCore.QDir(directory).filePath(safe_filename)

                with open(file_path, 'w', encoding='utf-8') as file:
                    values_str = '\n'.join(map(str, values))
                    file.write(values_str)
            QtWidgets.QMessageBox.information(
                self,
                "Сохранение завершено",
                f"Файлы успешно сохранены в папку:\n{directory}\n\n"
                f"Сохранено файлов: {len(self.data)}")
            return

        if self.current_generator_name == 'Датчик наличия пузырьков':
            file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Сохранить данные",
                QtCore.QDir.currentPath()+'/Trecv',
                "Text files (*.txt);;All Files (*)"
            )
            if not file_path:
                return
            try:
                if not file_path.endswith('.txt'):
                    file_path += '.txt'
                np.savetxt(
                    file_path,
                    self.data[1],
                    fmt='%.6f',
                    delimiter=' ',
                    newline='\n'
                )
                QtWidgets.QMessageBox.information(self, "Успех", "Файл сохранен!")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Ошибка", f'ошибка: {str(e)}')
            if self.current_generator_name == 'Датчик наличия пузырьков':
                file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    "Сохранить данные",
                    QtCore.QDir.currentPath()+'/Tsend',
                    "Text files (*.txt);;All Files (*)"
                )
                if not file_path:
                    return
                try:
                    if not file_path.endswith('.txt'):
                        file_path += '.txt'
                    np.savetxt(
                        file_path,
                        self.data[0],
                        fmt='%.6f',
                        delimiter=' ',
                        newline='\n'
                    )
                    QtWidgets.QMessageBox.information(self, "Успех", "Файл сохранен!")
                except Exception as e:
                    QtWidgets.QMessageBox.critical(self, "Ошибка", f'ошибка: {str(e)}')

            return

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Сохранить данные",
            QtCore.QDir.currentPath() + '/' + str(self.current_generator_name),
            "Text files (*.txt);;All Files (*)"
        )
        int_format = [
            'Датчик температуры',
            'Гидравлический датчик давления',
            'Датчик наличия крови',
            'Датчик насыщения крови кислородом',
            'Датчик pH',
            'Датчик артериального давления',
            'Счетчик Гейгера',
            'Датчик расхода',
            'Датчик глюкозы'

        ]
        if self.current_generator_name in int_format:
            xformat = '%d'
        elif self.current_generator_name == 'Датчик уровня жидкости':
            xformat = '%d %1.3f %1.3f %1.3f %1.3f %1.3f'
        else:
            xformat = '%.4f'
        if not file_path:
            return
        try:
            if not file_path.endswith('.txt'):
                file_path += '.txt'
            np.savetxt(
                file_path,
                self.data,
                fmt=xformat,
                delimiter=' ',
                newline='\n'
            )
            QtWidgets.QMessageBox.information(self, "Успех", "Файл сохранен!")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f'ошибка: {str(e)}')


    def save_to_txt(self):
        if not hasattr(self,'data') or self.data is None:
            QtWidgets.QMessageBox.warning(self,"Ошибка","Нечего сохранять")
            return
        if self.current_generator_name == 'Датчик нитратов':
            directory = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                "Выберите папку для сохранения",
                QtCore.QDir.currentPath(),
                QtWidgets.QFileDialog.ShowDirsOnly
            )
            if not directory:
                return
            for filename, values in self.data.items():
                safe_filename = f"{filename.replace(' ', '_')}.txt"
                file_path = QtCore.QDir(directory).filePath(safe_filename)

                with open(file_path, 'w', encoding='utf-8') as file:
                    values_str = '\n'.join(map(str, values))
                    file.write(values_str)
            QtWidgets.QMessageBox.information(
                self,
                "Сохранение завершено",
                f"Файлы успешно сохранены в папку:\n{directory}\n\n"
                f"Сохранено файлов: {len(self.data)}")
            return

        if self.current_generator_name == 'Датчик наличия пузырьков':
            file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Сохранить данные",
                QtCore.QDir.currentPath()+'/Trecv',
                "Text files (*.txt);;All Files (*)"
            )
            if not file_path:
                return
            try:
                if not file_path.endswith('.txt'):
                    file_path += '.txt'
                np.savetxt(
                    file_path,
                    self.data[1],
                    fmt='%.6f',
                    delimiter=' ',
                    newline='\n'
                )
                QtWidgets.QMessageBox.information(self, "Успех", "Файл сохранен!")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Ошибка", f'ошибка: {str(e)}')
            if self.current_generator_name == 'Датчик наличия пузырьков':
                file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    "Сохранить данные",
                    QtCore.QDir.currentPath()+'/Tsend',
                    "Text files (*.txt);;All Files (*)"
                )
                if not file_path:
                    return
                try:
                    if not file_path.endswith('.txt'):
                        file_path += '.txt'
                    np.savetxt(
                        file_path,
                        self.data[0],
                        fmt='%.6f',
                        delimiter=' ',
                        newline='\n'
                    )
                    QtWidgets.QMessageBox.information(self, "Успех", "Файл сохранен!")
                except Exception as e:
                    QtWidgets.QMessageBox.critical(self, "Ошибка", f'ошибка: {str(e)}')

            return

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Сохранить данные",
            QtCore.QDir.currentPath() + '/' + str(self.current_generator_name),
            "Text files (*.txt);;All Files (*)"
        )
        int_format = [
            'Датчик температуры',
            'Гидравлический датчик давления',
            'Датчик наличия крови',
            'Датчик насыщения крови кислородом',
            'Датчик pH',
            'Датчик артериального давления',
            'Счетчик Гейгера',
            'Датчик расхода',
            'Датчик глюкозы'

        ]
        if self.current_generator_name in int_format:
            xformat = '%d'
        elif self.current_generator_name == 'Датчик уровня жидкости':
            xformat = '%d %1.3f %1.3f %1.3f %1.3f %1.3f'
        else:
            xformat = '%.4f'
        if not file_path:
            return
        try:
            if not file_path.endswith('.txt'):
                file_path += '.txt'
            np.savetxt(
                file_path,
                self.data,
                fmt=xformat,
                delimiter=' ',
                newline='\n'
            )
            QtWidgets.QMessageBox.information(self, "Успех", "Файл сохранен!")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка", f'ошибка: {str(e)}')

    def toggle_interpolation_fields(self, state):
        is_visible = state == QtCore.Qt.Checked

        hiddenWidgets= [
            self.Amp,self.Freq,self.Phase,
            self.Offset,self.label_2,self.label_3,
            self.label_4,self.label_5,self.button_setDefault,
            self.label_29,self.n_points_geig,self.n_points_cons,
            self.label_32,self.n_points_ph,self.label_56,
            self.label_22,self.time_ph,self.T_interval_ph,
            self.Val_interval_ph,self.t_label_ph,self.Val_label_ph,
            self.time_ph_2, self.time_ph_3, self.label_57, self.label_58,
            self.n_points_9, self.stepT, self.T_interval_bubble, self.Val_interval_bubble,
            self.t_example, self.val_example,self.t_label_bubble, self.val_label_bubble,
            self.label_27, self.label_28, self.strength_ph,self.label_59,
            self.label_60,self.label_61,self.label_62
        ]
        unHiddenWidgets = [
            self.T_interval,self.Val_interval,
            self.T_interval_temp,self.Val_interval_temp,
            self.label_33,self.label_34,self.t_label_temp,
            self.val_label_temp,self.n_out_temp,self.n_points_temp,
            self.noiselvl_temp,self.T_max_temp,self.T_min_temp,
            self.label_9,self.label_10,self.label_11,
            self.label_35,self.label_36,self.label_12,
            self.label_13,self.label_14,self.label_37,
            self.label_38,self.t_label_hydr,self.val_label_hydr,
            self.n_out_hydr,self.n_points_hydr,self.noiselvl_hydr,
            self.T_min_hydr,self.T_max_hydr,self.T_interval_hydr,
            self.Val_interval_hydr,self.base_i,self.drop_val,
            self.n_points_bl,self.n_out_bl,self.noiselvl_bl,
            self.T_interval_bl,self.Val_interval_bl,self.T_max_bl,
            self.T_min_bl,self.label_15,self.label_16,self.label_17,
            self.label_39,self.label_40,self.label_41,self.label_42,
            self.t_label_bl,self.val_label_bl, self.Val_interval_geig,
            self.T_interval_geig, self.t_label_geig,self.Val_label_geig,
            self.Val_label_geig_2, self.Val_label_cons, self.t_interval_cons,
            self.t_label_cons, self.Val_interval_cons, self.Val_label_geig_3,
            self.n_points_ph, self.label_56,
            self.label_22, self.time_ph, self.T_interval_ph,
            self.Val_interval_ph, self.t_label_ph, self.Val_label_ph,
            self.time_ph_2, self.time_ph_3, self.label_57, self.label_58,
            self.n_points_9, self.stepT, self.T_interval_bubble, self.Val_interval_bubble,
            self.t_example, self.val_example, self.t_label_bubble, self.val_label_bubble,
            self.label_27, self.label_28,self.strength_ph,self.label_59,
            self.label_60, self.label_61, self.label_62
        ]
        for widget in hiddenWidgets:
            widget.setVisible(not is_visible)
        for widget in unHiddenWidgets:
            widget.setVisible(is_visible)

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = FPIBS_Generator()
    window.show()
    app.exec_()


