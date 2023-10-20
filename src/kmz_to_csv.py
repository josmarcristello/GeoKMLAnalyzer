import os

from functions import *

## User changeable area ##
filename_kml = 'Coastal GasLink.kml' 
filename_kml = 'Wetaskiwin_to_Leduc.kml' # File should be in [kml] folder.
num_samples = 10                         # This is the number of samples to take along the path.
## End of user changeable area ##

load_dotenv()
API_KEY = os.getenv('API_KEY') # Note: This is a Google API key. It should be stored in a .env file in the root directory.

geo_df = kml2df(filename_kml)
geo_df = get_elevation_data(geo_df, API_KEY, num_samples=num_samples)
geo_df = calculate_distances(geo_df)
write_to_csv(geo_df, filename_kml)
plot_elevation_profile(geo_df, filename_kml, show_plot=False)