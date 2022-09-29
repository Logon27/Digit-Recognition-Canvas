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

### Running The Program
```
python application.py
```
