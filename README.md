python -m venv venvYourEnvironmentName
venvYourEnvironmentName/scripts/activate.bat
pip install -r requirements.txt

Use QT designer to modify the dialog.ui file.  
Save the dialog.ui file.  
Compile the dialog.ui file to python with...   
```
pyuic5 dialog.ui > dialog.py
```

Then run the program...  
```
python application.py
```

https://www.pythonguis.com/tutorials/bitmap-graphics/

TODO:
Implement the network.py file as an actual class instead of just two methods.
Implement saving and loading weights from a file.

Work on the actual canvas application.

Troubleshooting:
Resizing not working. This was why...
https://stackoverflow.com/questions/6044836/how-to-make-a-qt-widget-grow-with-the-window-size
https://stackoverflow.com/questions/27536884/how-to-activate-centralwidget-in-qt-designer