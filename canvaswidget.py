# Imports
from inspect import _void
from math import ceil, floor
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QRadialGradient
from PyQt5.QtCore import Qt, QPointF, QPoint
from PyQt5.QtWidgets import (QApplication, QLabel, QWidget)
import random
import numpy as np
from fileio import loadNetwork

class CanvasWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        #This class abstraction is purely for formatting and resizing purposes
        QtWidgets.QWidget.__init__(self, parent)   # Inherit from QWidget
        self.canvas = MplCanvasWidget()            # Create canvas object
        self.vbl = QtWidgets.QVBoxLayout()         # Set box for plotting
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)

class MplCanvasWidget(QtWidgets.QLabel):

    #In memory representation of the canvas
    #Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
    #Modifying to be 0 to 1. 0 means background (white), 1 means foreground (black).
    #I don't actually have to swap the white and black values because 0,0,0 is black in rgb which is my background color.
    canvasState = np.full((28, 28), 0, dtype=np.float32)
    network = loadNetwork("mnistNetwork.pkl")

    def __init__(self):
        QWidget.__init__(self, alignment=QtCore.Qt.AlignTop)   # Inherit from QWidget
        #the Qpixmap will now downsize if you do not set a minimum size
        self.setMinimumSize(28, 28);
        #The expanding was moved to QT designer
        QWidget.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        QWidget.updateGeometry(self)
        self.initUI()

    def initUI(self):
        canvas = QtGui.QPixmap(28, 28)
        #fill the canvas with black because it could be random uninitialized data
        canvas.fill(QColor(0,0,0))
        # canvas = canvas.scaled(64, 64, QtCore.Qt.KeepAspectRatio)
        self.setPixmap(canvas)

    def calcPixelLocation(self, xCoord, yCoord):
        pixelLocationX = (self.brushSize / 2) + (self.brushSize * xCoord)
        pixelLocationY = (self.brushSize / 2) + (self.brushSize * yCoord)
        return QPoint(pixelLocationX, pixelLocationY)

    def mouseMoveEvent(self, e):
        #left click to draw. right click to erase
        if(e.buttons() == Qt.LeftButton):
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
            randomInt = random.randint(200, 255)
            pen.setColor(QColor(randomInt, randomInt, randomInt))
            pen.setWidthF(self.brushSize)
            painter.setPen(pen)

            painter.drawPoint(pixelQPoint)

            #TODO
            #Only repaint a white pixel if it becomes a lighter color
            #Add random gradient variation around the current pixel. To mimic a pressure brush
            #Make an in memory 2d array using the XCoord and YCoord to more easily pass values to the neural network (canvas will be a disconnected visual).

            # QPointF mousePosition = event->pos(); 
            # QRgb rgbValue = pixmap().toImage().pixel(mousePosition.x(), mousePostion.y());
            # Paint surrounding pixels a lighter color. Only paint if the color gets lighter

            painter.end()
            self.update()

            ### UPDATE IN MEMORY ARRAY ###
            #Convert to 0 to 1 scale.
            inputValue = (randomInt / 255)
            if yCoord >= 0 and yCoord <= 27 and xCoord >= 0 and xCoord <= 27:
                #need to flip the coordinates due to how arrays index values. X/Y to Row/Column
                self.canvasState[yCoord][xCoord] = inputValue

            ### PREDICT VALUE ###
            inputArray = self.canvasState.reshape(28 * 28, 1)
            output = self.network.predict(inputArray)
            #print('pred:', np.argmax(output))
        elif(e.buttons() == Qt.RightButton):
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
            pen.setColor(QColor(0, 0, 0))
            pen.setWidthF(self.brushSize)
            painter.setPen(pen)

            painter.drawPoint(pixelQPoint)
            painter.end()
            self.update()

            ### UPDATE IN MEMORY ARRAY ###
            #Convert to 0 to 1 scale.
            inputValue = 0
            if yCoord >= 0 and yCoord <= 27 and xCoord >= 0 and xCoord <= 27:
                #need to flip the coordinates due to how arrays index values. X/Y to Row/Column
                self.canvasState[yCoord][xCoord] = inputValue

    def mouseReleaseEvent(self, e):
        #print(self.canvasState.__str__())
        inputArray = self.canvasState.reshape(28 * 28, 1)
        output = self.network.predict(inputArray)
        print('pred:', np.argmax(output))
        #print('\n'.join([''.join(['{:.2f} '.format(item) for item in row]) for row in self.canvasState]))

    def resizeEvent(self, event):
        pixmap = self.pixmap()
        pixmap.fill(QColor(0,0,0))
        modWidth = self.width() % 28
        newWidth = self.width() - modWidth
        modHeight = self.height() % 28
        newHeight = self.height() - modHeight
        pixmap=pixmap.scaled(newWidth, newHeight, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pixmap)
        #Need to reset the canvasState array because resizing clears the screen.
        self.canvasState = np.full((28, 28), 0, dtype=np.float32)
