## About The Project

This project is a digit recognition canvas where you can draw your own custom digits on the canvas and the program will predict what digit it thinks it is. The neural network itself does not use any mainstream framework. I have essentially created my own scaled down neural network framework. I will likely release this neural network framework as its own project (if I haven't already). The current network used for this program is decent, but needs some improvements. The network itself was trained on the mnist dataset (which I modified during training for better generalization). It achieves about an **95% accuracy** on the test dataset. The canvas now supports both traditional deep neural networks and convolutional neural networks. A video of the program in action can be found below.

**NOTE:** I am in the process of creating an Windows executable for this project. However, I have had some dependency issues with the final executable. For now local development is the only option for running the program.

[![Digit Recognition Canvas Demo Link](https://img.youtube.com/vi/b7AX3uBqzZ8/0.jpg)](https://youtu.be/b7AX3uBqzZ8)

## Current Network Architecture(s)
You now have the choice to train a traditional deep neural network or a convolutional neural network for the canvas to use.

Traditional Deep Network
```
Dense(28 * 28, 800),
Sigmoid(),
Dense(800, 10),
Sigmoid(),
Dense(10, 10),
Softmax()
```

Convolutional Network
```
Convolutional((1, 28, 28), 5, 2),
Sigmoid(),
Convolutional((2, 24, 24), 3, 2),
Sigmoid(),
Convolutional((2, 22, 22), 3, 2),
Sigmoid(),
Flatten((2, 20, 20)),
Dense(2 * 20 * 20, 40),
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

Currently, you have to train your own network. The file size of the trained model I created is larger than github will allow. I am trying to find an alternative solution.
```
# For training an mnist network
python mnist_training.py
# For running the canvas
python application.py
```
```
# For training a convolutional mnist network
python mnist_conv_training.py
# For running the canvas
python application.py
```

### Known Issues
- Occasionally you will see an error "ValueError: cannot reshape array of size 1 into shape..." when doing multiple consecutive training runs. I have no idea what causes this error, but if you just re-run your command it will work fine (assuming you don't have an legitimate reshaping error).

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
