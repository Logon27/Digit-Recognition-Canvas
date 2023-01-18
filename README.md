## About The Project

This project is a digit recognition canvas where you can draw your own custom digits on the canvas and the program will predict what digit it thinks it is. The neural network itself does not use any mainstream framework. It uses [AeroNet](https://github.com/Logon27/AeroNet), which is a deep learning library I created myself. The current network used for this program is decent, but needs minor improvements. The network itself was trained on the mnist dataset (which I modified during training for better generalization). It achieves about an **95% accuracy** on the test dataset. The canvas now supports both traditional deep neural networks and convolutional neural networks. A video of the program in action can be found below.

[![Digit Recognition Canvas Demo Link](https://img.youtube.com/vi/b7AX3uBqzZ8/0.jpg)](https://youtu.be/b7AX3uBqzZ8)

## Current Network Architecture(s)
You now have the choice to train a traditional deep neural network or a convolutional neural network for the canvas to use. You can modify these networks however you like using AeroNet's syntax. So if you want to try to create a more performant network yourself you can.

Traditional Deep Network
```
Dense(28 * 28, 400),
Sigmoid(),
Dense(400, 10),
Sigmoid(),
Dense(10, 10),
Softmax()
```

Convolutional Network
```
Convolutional((1, 28, 28), 5, 5),
Sigmoid(),
Convolutional((3, 24, 24), 3, 5),
Sigmoid(),
Convolutional((3, 22, 22), 3, 5),
Sigmoid(),
Flatten((5, 20, 20)),
Dense(5 * 20 * 20, 40),
Sigmoid(),
Dense(40, 10),
Softmax()
```

## Setting Up Local Environment

### Creating A Virtual Environment

```
python -m venv venvYourEnvironmentName
venvYourEnvironmentName/scripts/activate.bat
pip install -r requirements.txt
```

## Running The Program (Local Development Environment)

### Convolution vs Non-Convolutional Models

There are automatic checks in the codebase to check if the first layer in your network is convolutional. So the canvas will automatically reshape its input to the proper size before predictions occur to prevent any errors. TLDR: You don't need to configure anything for running convolutional networks against the canvas program.

### Training And Running The Canvas Program

Currently, you have to train your own network. The file size of the trained model I created is larger than github will allow. I am trying to find an alternative solution. The convolutional model will perform much better than the regular dense mnist model.
```bash
# Change into the training directory
cd training
```

You should run the training models from the "training" directory because the saved network file the GUI application runs off of is expected to be named ```mnist-network.pkl``` and be in the ```training``` directory.

```bash
# For training a convolutional mnist network
python mnist_conv.py
# For running the canvas
cd ..
python application.py
```

```bash
# For training an mnist network
python mnist.py
# For running the canvas
cd ..
python application.py
```

---

### Known Limitations
- Resizing the window. You can currently resize the window, but the layout is not good unless the window is square.
- Network accuracy. The network accuracy is around 95%, but it still misclassifies digits in some edge cases.

### Modifying The UI
Use QT Designer to modify the dialog.ui file.
Save the dialog.ui file.  
Compile the dialog.ui file to python with...   
```
pyuic5 dialog.ui > dialog.py
```
