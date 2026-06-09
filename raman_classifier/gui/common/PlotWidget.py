from typing import Dict, List, Union

from PySide6.QtCore import QTimer
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import QWidget, QVBoxLayout

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


class PlotCanvas(FigureCanvas):

    def __init__(self, parent:QWidget=None)->None:
        FigureCanvas.__init__(self, Figure())
        self.setParent(parent)

        self.axes = self.figure.subplots(1, 1)
        self.axes.set_xlabel("")
        self.axes.set_ylabel("")
        self.axes.set_title("")


class PlotWidget(QWidget):

    __first_show:bool = True

    def __init__(self, parent:QWidget=None)->None:
        QWidget.__init__(self, parent)

        layout_main:QVBoxLayout = QVBoxLayout()

        self.canvas = PlotCanvas(self)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.__label_colors:Dict[str, str] = {}
        self.__lines:List[Line2D] = []

        layout_main.addWidget(self.toolbar)
        layout_main.addWidget(self.canvas)
        self.setLayout(layout_main)
        self.grid(True)

    def xlabel(self, label:str)->None:
        self.canvas.axes.set_xlabel(label)

    def ylabel(self, label:str)->None:
        self.canvas.axes.set_ylabel(label)

    def title(self, title:str)->None:
        self.canvas.axes.set_title(title)

    def plot(self, x:np.ndarray, y:np.ndarray, label:str="", color:str="", linestyle:str="", update:bool=True)->Line2D:
        kwargs = {
            "label": label,
            "color": color,
            "linestyle": linestyle
        }

        should_remove_keys = []
        for key, value in kwargs.items():
            if value == "":
                should_remove_keys.append(key)

        for key in should_remove_keys:
            del kwargs[key]
        
        should_get_color:bool = False
        if label in self.__label_colors:
            if not color and label:
                color = self.__label_colors[label]
                kwargs["color"] = color
        elif label:
            should_get_color:bool = True

        line, = self.canvas.axes.plot(x, y, **kwargs)
        if should_get_color:
            self.__label_colors[label] = line.get_color()

        self.__lines.append(line)

        if update:
            self.update()
        
        return line

    def update_legend(self)->None:
        if self.canvas.axes.get_legend():
            self.canvas.axes.get_legend().remove()
        
        visible_lines = []
        visible_labels = []
        for line in self.__lines:
            label:str = line.get_label()
            if line.get_visible() and not self.is_line_empty(line) and label not in visible_labels:
                visible_lines.append(line)
                visible_labels.append(label)
        
        if visible_lines:
            self.canvas.axes.legend(
                visible_lines, 
                visible_labels
            )

    def remove(self, lines:Union[Line2D, List[Line2D]])->None:
        if isinstance(lines, Line2D):
            lines = [lines]

        for i in range(len(self.__lines)-1, -1, -1):
            if self.__lines[i] in lines:
                self.__lines.pop(i)

        for line in lines:
            line.remove()

        self.update()

    def update(self)->None:
        self.update_legend()
        self.canvas.figure.tight_layout()
        self.canvas.draw()
        QWidget.update(self)

    def grid(self, show:bool=True)->None:
        self.canvas.axes.grid(show)
        self.canvas.draw()

    def xlim(self, xmin:float=None, xmax:float=None):
        if xmin is not None or xmax is not None:
            self.canvas.axes.set_xlim(xmin, xmax)
            self.canvas.draw()
        else:
            return self.canvas.axes.get_xlim()

    def ylim(self, ymin:float=None, ymax:float=None):
        if ymin is not None or ymax is not None:
            self.canvas.axes.set_ylim(ymin, ymax)
            self.canvas.draw()
        else:
            return self.canvas.axes.get_ylim()

    def auto_scale(self, x:bool=True, y:bool=True)->None:
        self.canvas.axes.relim()
        self.canvas.axes.autoscale_view(scalex=x, scaley=y)
        self.canvas.draw()

    @staticmethod
    def is_line_empty(line:Line2D)->bool:
        x = np.asarray(line.get_xdata())
        y = np.asarray(line.get_ydata())
        
        if x.size == 0 or y.size == 0:
            return True
        
        try:
            x_float = x.astype(float, casting='same_kind', subok=True, copy=False)
            y_float = y.astype(float, casting='same_kind', subok=True, copy=False)
            return np.all(np.isnan(x_float)) or np.all(np.isnan(y_float))
        except (ValueError, TypeError):
            return (len(x) == 0 or all(v is None for v in x)) and (len(y) == 0 or all(v is None for v in y))
        
    def tight_layout(self):
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def showEvent(self, event:QShowEvent):
        super().showEvent(event)
        if PlotWidget.__first_show:
            QTimer.singleShot(100, self.tight_layout)
            PlotWidget.__first_show = False
        else:
            QTimer.singleShot(0, self.tight_layout)

    def clear(self):
        self.canvas.axes.clear()
        self.canvas.draw()