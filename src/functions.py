import os

# Standard library imports      
import json
import math
import requests

# Related third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local application/library-specific imports
from dotenv import load_dotenv
from bs4 import BeautifulSoup # Note: Also install lxml parser
from tqdm import tqdm

def process_coordinate_string(coord_string):
    """
    Process a coordinate string from a KML file, extracting latitude and longitude values.
    
    Parameters:
    coord_string (str): A string containing coordinate data from a KML file
    
    Returns:
    list: A list of alternating latitude and longitude values extracted from the input string
    """
    coordinates = coord_string.strip().split(" ")
    lat_lng_values = []

    for coord in coordinates:
        lat_lng = coord.split(',')

        if len(lat_lng) == 3:
            lat = lat_lng[1]
            lng = lat_lng[0]
            lat_lng_values.extend([lat, lng])

    return lat_lng_values

def kml2df(file_name):
    """
    Convert a KML file containing latitude and longitude coordinates to a DataFrame.
    
    Parameters:
    file_name (str): The input KML file name, including the '.kml' extension
    
    Returns:
    pd.DataFrame: A DataFrame containing 'Lat' and 'Lon' columns with the latitude and longitude values.
    """
    kml_file_path = f"data/kml/{file_name}"
        
    with open(kml_file_path, 'r') as kml_file:
        kml_content = BeautifulSoup(kml_file, 'xml')

    latitudes, longitudes = [], []

    for coordinates in kml_content.find_all('coordinates'):
        data = process_coordinate_string(coordinates.string)
        lat_coords = [float(value) for index, value in enumerate(data) if index % 2 == 0]
        lon_coords = [float(value) for index, value in enumerate(data) if index % 2 == 1]

        latitudes.extend(lat_coords)
        longitudes.extend(lon_coords)

    coordinate_df = pd.DataFrame({'Lat': latitudes, 'Lon': longitudes})

    return coordinate_df