# Imports
from inspect import _void
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import (QApplication, QLabel, QWidget)

class CanvasWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        #This class abstraction is purely for formatting and resizing purposes
        QtWidgets.QWidget.__init__(self, parent)   # Inherit from QWidget
        self.canvas = MplCanvasWidget()            # Create canvas object
        self.vbl = QtWidgets.QVBoxLayout()         # Set box for plotting
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)

class MplCanvasWidget(QtWidgets.QLabel):

    brushWidth = 3

    def __init__(self):
        QWidget.__init__(self)   # Inherit from QWidget
        #the Qpixmap will now downsize if you do not set a minimum size
        self.setMinimumSize(28, 28);
        QWidget.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        QWidget.updateGeometry(self)
        self.initUI()
        # self.setMouseTracking(True)

    def initUI(self):
        canvas = QtGui.QPixmap(28, 28)
        #fill the canvas with black because it could be random uninitialized data
        canvas.fill(QColor(0,0,0))
        # canvas = canvas.scaled(64, 64, QtCore.Qt.KeepAspectRatio)
        self.setPixmap(canvas)

        self.last_x, self.last_y = None, None

    def mouseMoveEvent(self, e):
        if self.last_x is None: # First event.
            self.last_x = e.x()
            self.last_y = e.y()
            return # Ignore the first time.

        painter = QtGui.QPainter(self.pixmap())
        painter.setPen(QPen(QColor(255, 255, 255), self.brushWidth))
        painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        
        #painter.drawPoint(10, 10)
        painter.end()
        self.update()

        # Update the origin for next time.
        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None

    def resizeEvent(self, event):
        pixmap = self.pixmap()
        pixmap.fill(QColor(0,0,0))
        pixmap=pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pixmap)
