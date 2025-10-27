Scripts for nowcasting geoelectric fields across Ireland are currently being optimised. Scripts adapted from previous work by Campanya et al 2018 (see https://github.com/joancampanya/Compute_GeoElectric_Fields). Python scripts are found in the "scr" folder.

# Using the Model

Python scripts are found in the "scr" folder. Run the "nowcast_model.py" to run the nowcast geoelectric field model. This scripts pulls functions in from a) "functions_EM_modelling.py" to calculate the nowcast geoelectric field and read in inputs, b) "SECS_interpolation.py" to estimate the magnetic field conditions, c) "mapping_geo_library.py" to plot the mapped data d) "secs_pre.py"  calculates spheric elementary current systems. Inputs for the model can be adjusted in "inputs.py". Magnetoelluric transfer functions can be found in the "in/data/" folder

The scripts work as follows: 1) Downloads data from the MagIE website (www.magie.ie) 2) Reads in the and preprocesses the magnetic field data, 3) Performs a SECS interpolation to calculate E fields at each site, 4) Calculates the electric field at each MT site, 5) Plots the data onto a map.


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


