from pickletools import uint1
from PyQt5 import QtWidgets, QtCore
from dialog import Ui_MainWindow
import sys

#TODO
#Only repaint a white pixel if it becomes a lighter color
#Add a larger brush

#Add Rotation, Scaling, And Noise to the dataset during training

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