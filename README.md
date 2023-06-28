# NowcastGeoelectrics
Note: Scripts are currently being adapted for general use and full versions, with working examples for previous storms upon completion. Currently the scripts will download real-time data from the MagIE website and plot realtime conditions

Scripts for nowcasting geoelectric fields across Ireland are currently being optimised. Scripts adapted from previous work by Campanya et al 2018 (see https://github.com/joancampanya/Compute_GeoElectric_Fields). Python scripts are found in the "scr" folder.

# Packages Required:
numpy
matplotlib
pandas
datetime
scipy
time 
urllib
geopy
Basemap
ffmpy
cv2 
os

# Optional Packages

osgeo
seaborn
shapely
fiona 
gdal


# Nowcasting Geoelectric Fields

Python scripts are found in the "scr" folder. Run the "nowcast_model.py" to run the nowcast geoelectric field model. This scripts pulls functions in from a) "functions_EM_modelling.py" to calculate the nowcast geoelectric field and read in inputs, b) "SECS_interpolation.py" to estimate the magnetic field conditions, c) "mapping_geo_library.py" to plot the mapped data d) "secs_pre.py"  calculates spheric elementary current systems. Inputs can be adjusted in "inputs.py"
