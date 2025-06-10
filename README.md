# QR and Barcode Scanner
A pyqt6 app to read 1D barcode and qrcode using classic image processing technique.

## The project contains two applications:
### barcode-scanner - the barcode scanner app
- config - parameters.py contains the parameters configured for the barcode detection using parameterized flow-control 
- ipcv - contains modules for camera, scanner and shape. cvlib is a opencv wrapper for application consumption
- tests - unit tests
- utility - utility used in application and Jupyter NBs
- widgets - scanner_win, scrollable: UI widgets
- job_controller.py - the main controller between scanner_win and scanner and camera  
- main.py - to run the application 
### Jupyter Notebooks - concepts used for developing the 1D barcode and QC code algorithms
- nb-barcode - 1D barcode detection algorithm 1, 2, 3 & 4 and tests
- nb-contour - theory and opencv contour for detecting rectangles (1D barcode) and box (QR code)
- nb-morphology - theory for building the kernel for 1D barcode
- nb-qrcode - QR code detection algorithm 1, 2 and tests
## To run the application:
Pre-requisite: Python was installed and configured in the local machine.
- locate into the dir after cloning the code locally: `cd barcode-scanner`
- create virtual environment (to abstract pip installation): `python -m venv venv`
- activate the virtual environment (bash cmd): `source venv/bin/activate`
- activate the virtual environment (windows): `venv\Scripts\activate`
- install required libraries: `pip install -r requirements.txt`
- run the application: `python main.py`


