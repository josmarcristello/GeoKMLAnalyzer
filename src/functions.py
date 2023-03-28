import os

# Standard library imports
import json
import math
import requests

# Related third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

# Local application/library-specific imports
from bs4 import BeautifulSoup # Note: Also install lxml parser
from tqdm import tqdm
from geopy.distance import geodesic


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

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth's surface
    using the Haversine formula.
    NOTE: Deprecated for general use, use geopy.distance.geodesic instead for better accuracy. Still maintained for performance purposes (much faster).
    
    Parameters:
    lat1 (float): Latitude of the first point in decimal degrees
    lon1 (float): Longitude of the first point in decimal degrees
    lat2 (float): Latitude of the second point in decimal degrees
    lon2 (float): Longitude of the second point in decimal degrees
    
    Returns:
    float: The great-circle distance between the two points in meters
    """
    # Earth's radius in meters
    EARTH_RADIUS = 6371000

    # Convert latitude and longitude from decimal degrees to radians
    lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
    lon1_rad, lon2_rad = math.radians(lon1), math.radians(lon2)

    # Calculate the differences in latitude and longitude
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    # Haversine formula
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    distance = 2 * EARTH_RADIUS * math.asin(math.sqrt(a))

    return distance


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
        
        # Calculate distance and determine num_samples
        distance = geodesic_distance(lat1, lon1, lat2, lon2)*1000
        if distance > 1:
            num_samples = min(int(distance), 512)
        else:
            num_samples = 2
        
        url = f"https://maps.googleapis.com/maps/api/elevation/json?path={lat1}%2C{lon1}%7C{lat2}%2C{lon2}&samples={num_samples}&key={API_KEY}"
        
        payload = {}
        headers = {}
        response = requests.request("GET", url, headers=headers, data=payload)

        js_str = json.loads(response.text)
        response_len = len(js_str['results'])
        
        # Adjust the range to exclude the last point
        for j in range(response_len - 1):
            elev_list.append(js_str['results'][j]['elevation'])
            lat_list.append(js_str['results'][j]['location']['lat'])
            lon_list.append(js_str['results'][j]['location']['lng'])

    # Manually add the last point from the last set of points
    elev_list.append(js_str['results'][-1]['elevation'])
    lat_list.append(js_str['results'][-1]['location']['lat'])
    lon_list.append(js_str['results'][-1]['location']['lng'])

    return pd.DataFrame({'Lat': lat_list, 'Lon': lon_list, 'Elevation[m]': elev_list})


def geodesic_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the geodesic distance between two latitude-longitude pairs using the Vincenty function.
    
    Parameters:
    lat1 (float): Latitude of the first point
    lon1 (float): Longitude of the first point
    lat2 (float): Latitude of the second point
    lon2 (float): Longitude of the second point
    
    Returns:
    float: The geodesic distance in kilometers between the two points
    """
    point1 = (lat1, lon1)
    point2 = (lat2, lon2)

    return geodesic(point1, point2).kilometers


def calculate_distances(df):
    """
    Calculate distances between consecutive latitude and longitude pairs in the input DataFrame using the provided Vincenty function.
    
    Parameters:
    df (pd.DataFrame): A DataFrame containing 'Lat' and 'Lon' columns with latitude and longitude values
    
    Returns:
    pd.DataFrame: A DataFrame containing the orDelta_Distance[m]iginal 'Lat' and 'Lon' columns, plus an additional 'Distance' column with the calculated distances in kilometers
    """
    d_list = []

    for i in range(len(df) - 1):
        lat1, lon1 = df.iloc[i]['Lat'], df.iloc[i]['Lon']
        lat2, lon2 = df.iloc[i + 1]['Lat'], df.iloc[i + 1]['Lon']
        
        #dp = haversine(lat1, lon1, lat2, lon2) / 1000  # km
        dp = geodesic_distance(lat1, lon1, lat2, lon2)*1000 # km → m
        d_list.append(dp)

    # Add the distances to the DataFrame and set the last distance value to 0
    df['Delta_Distance[m]'] = d_list + [0]
    
    return df

def write_to_csv(df, file_name):
    """
    Write the input DataFrame with latitude, longitude, elevation, and distance values to a CSV file.
    
    Parameters:
    df (pd.DataFrame): A DataFrame containing 'Lat', 'Lon', 'Elevation', and 'Distance' columns
    file_name (str): The name of the KML file used to generate the DataFrame, including the '.kml' extension
    
    Output:
    Creates a CSV file with the name 'altitudes_{file_name}.csv', containing 'Lat', 'Lon', 'Elevation', and 'Distance' columns.
    """
    
    df['Distance[m]'] = df['Delta_Distance[m]'].cumsum()
        
    output_file_name = f"output/altitudes_{file_name.replace('.kml', '.csv')}"
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)

    df.to_csv(output_file_name, index=False, encoding="ISO-8859-1")    

def plot_elevation_profile(df, file_name, show_plot=True):
    """
    Create elevation profile plots against the number of samples and the estimated distance.
    
    Parameters:
    df (pd.DataFrame): A DataFrame containing 'Lat', 'Lon', 'Elevation', and 'Distance' columns
    file_name (str): The name of the KML file used to generate the DataFrame, including the '.kml' extension
    show_plot (bool): Optional; If True, display the plot; default is True
    
    Output:
    Saves two plots: 'altitude_x_samples_{file_name}.png' and 'altitude_x_distance_{file_name}.png'
    """
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the figure title

    # Plot elevation profile against # Samples
    plt.figure(figsize=(28, 4))
    plt.plot(df['Elevation[m]'])

    plt.title(f'Elevation(m) x # Samples for {file_name}')
    plt.ylabel("Elevation(m)")
    plt.xlabel("Samples (#)")

    plt.xlim([0, len(df)])

    plt.grid()
    #plt.legend(fontsize='small')
    plt.savefig(f'output/altitude_x_samples_{file_name.replace(".kml", ".png")}', dpi=450, bbox_inches='tight', transparent=False)
    if show_plot:
        plt.show()

    # Plot elevation profile against distance (estimated)
    plt.figure(figsize=(28, 4))
    x = np.linspace(0.0, df['Delta_Distance[m]'].sum(), num=len(df))  # Making x axis data
    plt.plot(x, df['Elevation[m]'])

    plt.title(f'Elevation(m) x Distance (km) for {file_name}')
    plt.ylabel("Elevation(m)")
    plt.xlabel("Distance(km)")

    plt.xlim([0, df['Delta_Distance[m]'].sum()])

    plt.grid()
    #plt.legend(fontsize='small')
    plt.savefig(f'output/altitude_x_distance_{file_name.replace(".kml", ".png")}', dpi=450, bbox_inches='tight', transparent=False)
    if show_plot:
        plt.show()


def plot_elevation_profile_with_pins(df, file_name, show_plot=True):
    def avoid_overlapping_labels(labels, y_values, default_offset=1.5):
        sorted_indices = np.argsort(y_values)
        sorted_labels = [labels[i] for i in sorted_indices]
        sorted_y_values = [y_values[i] for i in sorted_indices]

        offsets = [default_offset] * len(labels)
        for i, j in itertools.combinations(range(len(labels)), 2):
            if abs(sorted_y_values[i] - sorted_y_values[j]) < default_offset * 2:
                offsets[j] = offsets[i] + default_offset

        unsorted_offsets = [0] * len(labels)
        for i, offset in zip(sorted_indices, offsets):
            unsorted_offsets[i] = offset
        return unsorted_offsets

    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('axes', titlesize=BIGGER_SIZE)

    plt.figure(figsize=(28, 4))
    x = np.linspace(0.0, df['Delta_Distance[m]'].sum(), num=len(df))
    plt.plot(x, df['Elevation[m]'])

    labels = []
    label_positions = []

    for i, row in df.iterrows():
        if pd.notna(row['Pin']):
            plt.plot(x[i], row['Elevation[m]'], 'ro')
            labels.append(row['Pin'])
            label_positions.append(i)

    y_label_positions = [df.iloc[i]['Elevation[m]'] for i in label_positions]
    label_offsets = avoid_overlapping_labels(labels, y_label_positions)

    for label, x_pos, y_pos, offset in zip(labels, label_positions, y_label_positions, label_offsets):
        plt.annotate(label, (x[x_pos], y_pos + offset), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.title(f'Elevation(m) x Distance (km) with Pins for {file_name}')
    plt.ylabel("Elevation(m)")
    plt.xlabel("Distance(km)")

    plt.xlim([0, df['Delta_Distance[m]'].sum()])

    plt.grid()
    plt.savefig(f'output/altitude_x_distance_with_pins_{file_name.replace(".kml", ".png")}', dpi=450, bbox_inches='tight', transparent=False)
    if show_plot:
        plt.show()

        
def parse_pins(file_name):
    """
    Parse a KML file containing pins with coordinates and return a list of tuples containing the pin name, latitude, and longitude.

    :param file_name: The name of the KML file located in the "data/kml/" directory.
    :type file_name: str
    :return: A list of tuples containing the pin name, latitude, and longitude, e.g., [('pin1', lat1, lon1), ('pin2', lat2, lon2)].
    :rtype: list of tuple
    """
    kml_file_path = f"data/kml/{file_name}"

    with open(kml_file_path, 'r') as kml_file:
        kml_content = BeautifulSoup(kml_file, 'xml')

    pins = []
    for placemark in kml_content.find_all('Placemark'):
        pin_name = placemark.find('name').string
        coordinates = placemark.find('coordinates').string.strip().split(',')
        lat, lon = float(coordinates[1]), float(coordinates[0])
        pins.append((pin_name, lat, lon))

    return pins


def find_closest_points_for_pins(geo_df, pins):
    """
    Find the closest points on the route for each pin and add the pin label to the corresponding row in the DataFrame.

    :param geo_df: A GeoDataFrame with the route data containing columns ['Lat', 'Lon']. It can optionally contain altitude ['Elevation'] and distance ['Distance'] columns.
    :param pins: A list of tuples, where each tuple contains the pin name, latitude, and longitude, e.g., [('pin1', lat1, lon1), ('pin2', lat2, lon2)].
    """
    # Find the closest point on the route for each pin
    for pin_name, pin_lat, pin_lon in pins:
        min_distance = float('inf')
        closest_point_index = -1

        for i, row in geo_df.iterrows():
            lat, lon = row['Lat'], row['Lon']
            distance = haversine(pin_lat, pin_lon, lat, lon)

            if distance < min_distance:
                min_distance = distance
                closest_point_index = i

        # Add the pin label to the closest point on the route
        geo_df.loc[closest_point_index, 'Pin'] = pin_name
    return geo_df