from config import *
if enableCuda:
    import cupy as np
else:
    import numpy as np
# Imports
from math import ceil, floor
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor, QRadialGradient
from PyQt5.QtCore import Qt, QPointF, QPoint
from PyQt5.QtWidgets import QApplication, QLabel, QWidget
import random
from fileio import loadNetwork

class CanvasWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        # This class abstraction is purely for formatting and resizing purposes
        QtWidgets.QWidget.__init__(self, parent)   # Inherit from QWidget
        self.canvas = MplCanvasWidget()            # Create canvas object
        self.vbl = QtWidgets.QVBoxLayout()         # Set box for plotting
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)

class MplCanvasWidget(QtWidgets.QLabel):

    # In memory representation of the canvas
    # Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
    # Modifying to be 0 to 1. 0 means background (white), 1 means foreground (black).
    # I don't actually have to swap the white and black values because 0,0,0 is black in rgb which is my background color.
    canvasState = np.full((28, 28), 0, dtype=np.float32)
    network = loadNetwork("mnist-network.pkl")

    def __init__(self):
        QWidget.__init__(self, alignment=QtCore.Qt.AlignTop)   # Inherit from QWidget
        # The Qpixmap will now downsize if you do not set a minimum size
        self.setMinimumSize(28, 28);
        # The expanding was moved to QT designer
        QWidget.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        QWidget.updateGeometry(self)
        self.initUI()

    def initUI(self):
        canvas = QtGui.QPixmap(28, 28)
        # Fill the canvas with black because it could be random uninitialized data
        canvas.fill(QColor(0,0,0))
        self.setPixmap(canvas)

    def passUI(self, ui):
        self.ui = ui

    def calcPixelLocation(self, xCoord, yCoord):
        pixelLocationX = (self.brushSize / 2) + (self.brushSize * xCoord)
        pixelLocationY = (self.brushSize / 2) + (self.brushSize * yCoord)
        return QPoint(pixelLocationX, pixelLocationY)
    
    def drawPoint(self, x, y, min, max):
        #This makes a cleaner brush and prevents very bright pixels from being overwritten.
        if y >= 0 and y <= 27 and x >= 0 and x <= 27 and self.canvasState[y][x] > .75:
            return

        painter = QtGui.QPainter(self.pixmap())
        pen = QPen()
        randomInt = random.randint(min, max)
        pen.setColor(QColor(randomInt, randomInt, randomInt))
        pen.setWidthF(self.brushSize)
        painter.setPen(pen)
        pixelQPoint = self.calcPixelLocation(x, y)
        painter.drawPoint(pixelQPoint)
        painter.end()
        self.update()

        ### UPDATE IN MEMORY ARRAY ###
        # Convert to 0 to 1 scale.
        inputValue = (randomInt / 255)
        if y >= 0 and y <= 27 and x >= 0 and x <= 27:
            # Need to flip the coordinates due to how arrays index values. X/Y to Row/Column
            self.canvasState[y][x] = inputValue

    def mouseMoveEvent(self, e):
        # Left click to draw. Right click to erase
        if(e.buttons() == Qt.LeftButton):
            # A stupidly complicated way I invented to mimic a larger brush size.
            pixmapWidth = self.pixmap().width()
            self.brushSize = (pixmapWidth / 28)
            xCoord = floor(e.x() / self.brushSize)
            yCoord = floor(e.y() / self.brushSize)
            #print("base {} , {}".format(e.x(), e.y()))
            #print("{} , {}".format(xCoord, yCoord))

            # Center of brush
            self.drawPoint(xCoord, yCoord, 200, 255)
            # Pixels surrounding brush
            self.drawPoint(xCoord+1, yCoord, 50, 150)
            self.drawPoint(xCoord-1, yCoord, 50, 150)
            self.drawPoint(xCoord, yCoord+1, 50, 150)
            self.drawPoint(xCoord, yCoord-1, 50, 150)

        elif(e.buttons() == Qt.RightButton):
            # A stupidly complicated way I invented to mimic a larger brush size.
            pixmapWidth = self.pixmap().width()
            self.brushSize = (pixmapWidth / 28)
            xCoord = floor(e.x() / self.brushSize)
            yCoord = floor(e.y() / self.brushSize)
            #print("base {} , {}".format(e.x(), e.y()))
            #print("{} , {}".format(xCoord, yCoord))
            pixelQPoint = self.calcPixelLocation(xCoord, yCoord)

            # Draw the center pixel
            painter = QtGui.QPainter(self.pixmap())
            pen = QPen()
            pen.setColor(QColor(0, 0, 0))
            pen.setWidthF(self.brushSize)
            painter.setPen(pen)

            painter.drawPoint(pixelQPoint)
            painter.end()
            self.update()

            ### UPDATE IN MEMORY ARRAY ###
            # Convert to 0 to 1 scale.
            inputValue = 0
            if yCoord >= 0 and yCoord <= 27 and xCoord >= 0 and xCoord <= 27:
                # Need to flip the coordinates due to how arrays index values. X/Y to Row/Column
                self.canvasState[yCoord][xCoord] = inputValue

    def mouseReleaseEvent(self, e):
        if "Convolutional" in str(self.network.layers[0]):
            # For convolutional
            inputArray = self.canvasState.reshape(1, 28, 28)
        else:
            # For non-convolutional
            inputArray = self.canvasState.reshape(28 * 28, 1)
        output = self.network.predict(inputArray)
        self.resetFontColor()

        self.ui.label_0.setText("{:.2%}".format(output[0][0]))
        self.ui.label_1.setText("{:.2%}".format(output[1][0]))
        self.ui.label_2.setText("{:.2%}".format(output[2][0]))
        self.ui.label_3.setText("{:.2%}".format(output[3][0]))
        self.ui.label_4.setText("{:.2%}".format(output[4][0]))
        self.ui.label_5.setText("{:.2%}".format(output[5][0]))
        self.ui.label_6.setText("{:.2%}".format(output[6][0]))
        self.ui.label_7.setText("{:.2%}".format(output[7][0]))
        self.ui.label_8.setText("{:.2%}".format(output[8][0]))
        self.ui.label_9.setText("{:.2%}".format(output[9][0]))

        predictionIndex = np.argmax(output)
        if predictionIndex == 0:
            self.ui.label_0.setStyleSheet("QLabel { color : red; }");
        elif predictionIndex == 1:
            self.ui.label_1.setStyleSheet("QLabel { color : red; }");
        elif predictionIndex == 2:
            self.ui.label_2.setStyleSheet("QLabel { color : red; }");
        elif predictionIndex == 3:
            self.ui.label_3.setStyleSheet("QLabel { color : red; }");
        elif predictionIndex == 4:
            self.ui.label_4.setStyleSheet("QLabel { color : red; }");
        elif predictionIndex == 5:
            self.ui.label_5.setStyleSheet("QLabel { color : red; }");
        elif predictionIndex == 6:
            self.ui.label_6.setStyleSheet("QLabel { color : red; }");
        elif predictionIndex == 7:
            self.ui.label_7.setStyleSheet("QLabel { color : red; }");
        elif predictionIndex == 8:
            self.ui.label_8.setStyleSheet("QLabel { color : red; }");
        elif predictionIndex == 9:
            self.ui.label_9.setStyleSheet("QLabel { color : red; }");
        #For Debugging
        #print('pred:', np.argmax(output))
        #print('\n'.join([''.join(['{:.2f} '.format(item) for item in row]) for row in self.canvasState]))

    # Canvas gets cleared every time the window is resized
    def resizeEvent(self, event):
        pixmap = self.pixmap()
        pixmap.fill(QColor(0,0,0))
        modWidth = self.width() % 28
        newWidth = self.width() - modWidth
        modHeight = self.height() % 28
        newHeight = self.height() - modHeight
        pixmap=pixmap.scaled(newWidth, newHeight, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pixmap)
        # Need to reset the canvasState array because resizing clears the screen.
        self.canvasState = np.full((28, 28), 0, dtype=np.float32)
        # Reset the predictions
        self.ui.label_0.setText("{:.2%}".format(0))
        self.ui.label_1.setText("{:.2%}".format(0))
        self.ui.label_2.setText("{:.2%}".format(0))
        self.ui.label_3.setText("{:.2%}".format(0))
        self.ui.label_4.setText("{:.2%}".format(0))
        self.ui.label_5.setText("{:.2%}".format(0))
        self.ui.label_6.setText("{:.2%}".format(0))
        self.ui.label_7.setText("{:.2%}".format(0))
        self.ui.label_8.setText("{:.2%}".format(0))
        self.ui.label_9.setText("{:.2%}".format(0))

        self.resetFontColor()

    # Function to clear the canvas when the "Clear Canvas" button is pressed
    def clearCanvas(self):
        #print("Clearing Canvas...")
        pixmap = self.pixmap()
        pixmap.fill(QColor(0,0,0))
        self.setPixmap(pixmap)
        # Need to reset the canvasState array because resizing clears the screen.
        self.canvasState = np.full((28, 28), 0, dtype=np.float32)
        # Reset the predictions
        self.ui.label_0.setText("{:.2%}".format(0))
        self.ui.label_1.setText("{:.2%}".format(0))
        self.ui.label_2.setText("{:.2%}".format(0))
        self.ui.label_3.setText("{:.2%}".format(0))
        self.ui.label_4.setText("{:.2%}".format(0))
        self.ui.label_5.setText("{:.2%}".format(0))
        self.ui.label_6.setText("{:.2%}".format(0))
        self.ui.label_7.setText("{:.2%}".format(0))
        self.ui.label_8.setText("{:.2%}".format(0))
        self.ui.label_9.setText("{:.2%}".format(0))

        self.resetFontColor()

    def resetFontColor(self):
        self.ui.label_0.setStyleSheet("QLabel { color : black; }");
        self.ui.label_1.setStyleSheet("QLabel { color : black; }");
        self.ui.label_2.setStyleSheet("QLabel { color : black; }");
        self.ui.label_3.setStyleSheet("QLabel { color : black; }");
        self.ui.label_4.setStyleSheet("QLabel { color : black; }");
        self.ui.label_5.setStyleSheet("QLabel { color : black; }");
        self.ui.label_6.setStyleSheet("QLabel { color : black; }");
        self.ui.label_7.setStyleSheet("QLabel { color : black; }");
        self.ui.label_8.setStyleSheet("QLabel { color : black; }");
        self.ui.label_9.setStyleSheet("QLabel { color : black; }");