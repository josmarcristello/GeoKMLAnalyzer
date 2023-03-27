# Geo KML Analyzer

## Overview

The Geo Kml Analyzer is a Python-based tool designed to process and analyze elevation data from KML files. The tool reads KML files, extracts latitude and longitude coordinates, calculates distances between consecutive points, fetches elevation data from the Google Maps Elevation API, and generates descriptive statistics and plots for elevation profiles. This project aims to help users visualize and understand elevation profiles of geographical paths, such as hiking trails, roads, or pipelines.

## Features

- Reads and processes KML files containing latitude and longitude data
- Converts KML files to CSV format for easy data manipulation
- Calculates distances between consecutive latitude and longitude pairs using the Haversine formula
- Fetches elevation data from the Google Maps Elevation API
- Generates descriptive statistics for distance, elevation, latitude, and longitude
- Creates plots for elevation profiles against sample count and distance
- Exports processed data to CSV files and saves plots as images
- Displays a progress bar to track the processing status

## Installation

1. Clone the repository to your local machine:
```
git clone https://github.com/josmarcristello/GeoKMLAnalyzer.git
```

2. Change into the project directory:
```
cd elevation-profile-analyzer
```
3. Create a virtual environment and activate it:

```
python3 -m venv venv
source venv/bin/activate
```
4. Install the required packages:
```
pip install -r requirements.txt
```

## Usage

1. Create a .env file and include your Google maps API key (`API_KEY`). Example is not a valid key.
```
API_KEY = 'DCzaXyRrSW1e-7Fx2YEZlSLv81cYc3__-5ndsfFG'
```

2. Place your KML files in the `data/kml` directory. I have included a source file for test purposes.

3. Run the main script:

```
python src/kmz_to_csv.py
```

4. Processed data and plot images will be stored in the `output` directory.

5. To run descriptive statistics run the notebook: 
```
src/notebook/get-google-maps-altitude
```
You can explore the generated descriptive statistics and plots to analyze elevation profiles for your KML files.

## Sample Images
### Progress bar:

![Progress Bar](./public/progress_bar.gif)

### Altitude x Distance Plot:
![Altitude x Distance Plot](./public/altitude_x_distance_sample.png)

## Contributing

We welcome contributions to improve this project. Please submit a pull request with your proposed changes, and ensure your code follows the project's style guidelines and best practices.

## License

This project is licensed under the MIT License. Please see the `LICENSE` file for more information.