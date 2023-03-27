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

def get_elevation_data(df, API_KEY, num_samples=10):
    """
    Retrieve elevation data for latitude and longitude pairs in the input DataFrame using Google Maps Elevation API.
    
    Parameters:
    df (pd.DataFrame): A DataFrame containing 'Lat' and 'Lon' columns with latitude and longitude values
    API_KEY (str): Your Google Maps Elevation API key
    num_samples (int, optional): The number of samples per latitude and longitude pair. Default is 10.
    
    Returns:
    pd.DataFrame: A DataFrame containing 'Elevation', 'Lat', and 'Lon' columns with retrieved data
    """
    elev_list = []
    lat_list = []
    lon_list = []

    for i in tqdm(range(len(df) - 1), desc="Retrieving elevation data", unit="rows"):
        lat1, lon1 = df.iloc[i]['Lat'], df.iloc[i]['Lon']
        lat2, lon2 = df.iloc[i + 1]['Lat'], df.iloc[i + 1]['Lon']
        
        url = f"https://maps.googleapis.com/maps/api/elevation/json?path={lat1}%2C{lon1}%7C{lat2}%2C{lon2}&samples={num_samples}&key={API_KEY}"
        
        payload = {}
        headers = {}
        response = requests.request("GET", url, headers=headers, data=payload)

        js_str = json.loads(response.text)
        response_len = len(js_str['results'])
        
        for j in range(response_len):
            elev_list.append(js_str['results'][j]['elevation'])
            lat_list.append(js_str['results'][j]['location']['lat'])
            lon_list.append(js_str['results'][j]['location']['lng'])

    return pd.DataFrame({'Lat': lat_list, 'Lon': lon_list, 'Elevation[m]': elev_list})