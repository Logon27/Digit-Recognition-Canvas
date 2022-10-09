## About The Project

This project is a digit recognition canvas where you can draw your own custom digits on the canvas and the program will predict what digit it thinks it is. The neural network itself does not use any mainstream framework. I have essentially created my own scaled down neural network framework. I will likely release this neural network framework as its own project (if I haven't already). The current network used for this program is decent, but needs some improvements. The network itself was trained on the mnist dataset (which I modified during training for better generalization). It achieves about an 95% accuracy on the test dataset. The current network is a traditional deep neural network. However, I intend to code convolutional layers in the near future for improved accuracy. A video of the program in action can be found below. The codebase is available on my github.

## Downloading And Running The Program

Version 1.0.0 is now available for download! It will only run on Windows and was built with pyinstaller. You may get a warning from Windows about running the program due to it not being signed. This is just because I didn't purchase a publisher certificate to distribute the executable. However, the program itself is safe to run. You can scan it, decompile it, or view the source code on my github. You can just double click and run the executable.

## Current Network Architecture

```
Dense(28 * 28, 800),
Sigmoid(),
Dense(800, 10),
Sigmoid(),
Dense(10, 10),
Softmax()
```

## Setting Up Local Environment

### Creating A Virtual Environment

```
python -m venv venvYourEnvironmentName
venvYourEnvironmentName/scripts/activate.bat
pip install -r requirements.txt
```

### Modifying The UI
Use QT Designer to modify the dialog.ui file.
Save the dialog.ui file.  
Compile the dialog.ui file to python with...   
```
pyuic5 dialog.ui > dialog.py
```

### Running The Program (Local Development Environment)
```
# For training an mnist network
python mnist-training.py
# For running the canvas
python application.py
```

### Known Limitations
- Resizing the window. You can currently resize the window, but the layout is not good unless the window is square.
- Network accuracy. The network accuracy is around 87% and it still misclassifies digits in some cases.