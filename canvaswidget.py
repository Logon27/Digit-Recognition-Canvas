# Imports
from inspect import _void
from math import ceil, floor
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QRadialGradient
from PyQt5.QtCore import Qt, QPointF, QPoint
from PyQt5.QtWidgets import (QApplication, QLabel, QWidget)
import random

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
        QWidget.__init__(self, alignment=QtCore.Qt.AlignTop)   # Inherit from QWidget
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

    def calcPixelLocation(self, xCoord, yCoord):
        pixelLocationX = (self.brushSize / 2) + (self.brushSize * xCoord)
        pixelLocationY = (self.brushSize / 2) + (self.brushSize * yCoord)
        return QPoint(pixelLocationX, pixelLocationY)

    def mouseMoveEvent(self, e):
        #A stupidly complicated way I invented to mimic a larger brush size.
        pixmapWidth = self.pixmap().width()
        self.brushSize = (pixmapWidth / 28)
        xCoord = floor(e.x() / self.brushSize)
        yCoord = floor(e.y() / self.brushSize)
        #print("base {} , {}".format(e.x(), e.y()))
        #print("{} , {}".format(xCoord, yCoord))
        pixelQPoint = self.calcPixelLocation(xCoord, yCoord)

        #draw the center pixel
        painter = QtGui.QPainter(self.pixmap())
        pen = QPen()
        pen.setColor(QColor(255, 255, 255))
        pen.setWidthF(self.brushSize)
        painter.setPen(pen)

        painter.drawPoint(pixelQPoint)

        painter.end()
        self.update()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None

    def resizeEvent(self, event):
        pixmap = self.pixmap()
        pixmap.fill(QColor(0,0,0))
        modWidth = self.width() % 28
        newWidth = self.width() - modWidth
        modHeight = self.height() % 28
        newHeight = self.height() - modHeight
        pixmap=pixmap.scaled(newWidth, newHeight, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # pixmap=pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pixmap)
