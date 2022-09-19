from PyQt5 import QtWidgets, QtCore
from dialog import Ui_MainWindow
import sys

#TODO
#Only repaint a white pixel if it becomes a lighter color
#Add a larger brush
#Add test set statics to networking trying debug
#https://stackoverflow.com/questions/54633038/how-to-add-masking-noise-to-numpy-2-d-matrix-in-a-vectorized-manner
#Add Noise to the dataset during training

#Link for image processing
#https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.ndimage.rotate.html
#https://docs.scipy.org/doc/scipy/reference/ndimage.html#interpolation

#For Cupy
#import cupy as np
#numpy arrays must be converted to cupy arrays... cp.asarray(x_train)
#Convert input numpy arrays to cupy arrays
#x_train, y_train = (np.asarray(x_train), np.asarray(y_train))
#x_test, y_test = (np.asarray(x_test), np.asarray(y_test))

#image processing imports
#from cupyx.scipy.ndimage import rotate
#https://www.edureka.co/community/52900/how-do-i-share-global-variables-across-modules-python
#add a cuda toggle

class ApplicationWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(ApplicationWindow, self).__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        #A not so ideal way of passing the main ui at runtime.
        #Can't think of a better way since qt designer generates the dialog file
        self.ui.canvasWidget.canvas.passUI(self.ui)
        self.ui.clearCanvas.clicked.connect(self.ui.canvasWidget.canvas.clearCanvas)

    # Debugging Window Size
    # def resizeEvent(self, event):
    #     print("Width: {}, Height: {}".format(self.width(), self.height()))

def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()