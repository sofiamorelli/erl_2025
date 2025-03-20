import os
import re
from pyproj import Transformer
import geopandas
import xarray as xr
from scipy import interpolate
import numpy as np

from scr.Region import Country
from scr.settings import (LONGITUDE_NAME, LATITUDE_NAME, END_YEAR, TURBINE_LOCATIONS_DIR, BASE_DIR_DAT,
                          SURFACE_WIND_SPEED, COUNTRY)

from cdo import *
cdo.debug = True
cdo = Cdo()


def save_wind_speeds_at_turbine_locations():
    """
        Interpolates and saves wind speed data at specific turbine locations.

        The turbine locations for Germany are loaded from a shapefile, then filtered for turbines
        built by the final year of consideration. The function transforms turbine coordinates into
        the WGS84 coordinate system and interpolates wind speed data at these turbine locations,
        saving the results in a specified folder.
    """
    # Load turbine location data from shapefile for Germany
    file_path = os.path.join(TURBINE_LOCATIONS_DIR, 'windpower.shp')
    turbine_data = geopandas.read_file(file_path)

    # Filter turbines by construction year, keeping only those built up to END_YEAR
    turbine_data_filtered = turbine_data[turbine_data['Year'] <= END_YEAR]

    # Extract longitude and latitude coordinates from filtered data
    lons, lats = turbine_data_filtered['X_coord'], turbine_data_filtered['Y_coord']

    # Transform coordinates from EPSG:3035 (ETRS89) to EPSG:4326 (WGS84)
    transformer = Transformer.from_crs(crs_from=3035, crs_to=4326, always_xy=True)
    turbine_lons, turbine_lats = transformer.transform(lons.values, lats.values)
    turbine_coordinates = np.vstack((turbine_lats, turbine_lons)).T  # Combine into a 2D array of coordinates

    country = Country(COUNTRY)
    # Use continent-level data files to avoid interpolation issues near borders
    folder_path_continent = os.path.join(BASE_DIR_DAT, country.continent.name)
    new_folder_path = os.path.join(BASE_DIR_DAT, f'{country.name}_Turbine')
    os.makedirs(new_folder_path, exist_ok=True)  # Ensure output directory exists

    # Process each file in the continent folder, interpolating to turbine locations
    for filename in os.listdir(folder_path_continent):
        interpolate_to_turbine_locations_germany(folder_path_continent, filename, turbine_coordinates, new_folder_path)

    return


def interpolate_to_turbine_locations_germany(folder, filename, turbine_coordinates, new_folder_path):
    """
        Interpolates wind speed data from a continent-level file to specific turbine locations for Germany.

        Arguments:
            folder {str}: Path to the folder with continent-level wind data files.
            filename {str}: Name of the data file to process.
            turbine_coordinates {ndarray}: Array of latitude and longitude coordinates for turbine locations.
            new_folder_path {str}: Path to save the interpolated wind speed data for turbines.
    """
    country = Country(COUNTRY)
    country_name = country.name
    continent_name = country.continent.name

    # Replace continent name with country-turbine to generate the new filename
    new_filename = filename.replace(continent_name, f'{country_name}-Turbine')
    model_name = re.search(r'{}(.*?){}'.format(SURFACE_WIND_SPEED, continent_name), filename).group(1)

    # Match the model name with the country-specific filename for consistent naming
    for filename_country in os.listdir(f'dat/{country_name}'):
        if model_name in filename_country:
            new_filename = new_filename.replace(SURFACE_WIND_SPEED, filename_country.split('_')[0])
            break

    file_path = os.path.join(new_folder_path, new_filename)
    if os.path.exists(file_path):
        print(f'Already exists: {file_path}')
    else:
        # Define intermediate file path for cropped data
        intermediate_file_path = os.path.join(new_folder_path, 'intermediate.nc')

        # Crop data to country bounding box, extending boundaries for interpolation stability
        cdo.sellonlatbox(country.min_lon-10, country.max_lon+10, country.min_lat-10, country.max_lat+10,
                         input=f'{folder}/{filename}',
                         output=intermediate_file_path)

        # Open cropped data for interpolation
        ds = xr.open_dataset(intermediate_file_path)
        os.remove(intermediate_file_path)  # Remove intermediate file after loading
        data = ds[SURFACE_WIND_SPEED]

        # Generate latitude and longitude grids from data dimensions
        lon_grid, lat_grid = np.meshgrid(data[LONGITUDE_NAME], data[LATITUDE_NAME])
        points = np.vstack((lat_grid.ravel(), lon_grid.ravel())).T  # Flatten grids for interpolation points

        # Prepare time dimension and create an empty array for interpolated wind speeds
        time = data.time
        interpolated_values = np.empty((len(time), len(turbine_coordinates)))

        # Interpolate wind speed for each time step at turbine locations
        for i, t in enumerate(time):
            data_at_time_t = data.sel(time=t).values  # Get data for specific time
            values = data_at_time_t.ravel()  # Flatten data to match point grid
            # Interpolate wind speed at turbine coordinates
            interpolated_values[i, :] = interpolate.griddata(points, values, turbine_coordinates, method='linear')

        # Create DataArray for interpolated wind speed data with turbine and time dimensions
        interpolated_da = xr.DataArray(
            data=interpolated_values,
            dims=["time", "turbine"],
            coords={
                "time": time,
                "turbine": np.arange(len(turbine_coordinates)),  # Turbine index
                LATITUDE_NAME: ("turbine", turbine_coordinates[:, 0]),  # Latitude at each turbine location
                LONGITUDE_NAME: ("turbine", turbine_coordinates[:, 1])  # Longitude at each turbine location
            },
            name=SURFACE_WIND_SPEED
        )

        # Create a Dataset with the interpolated wind speed data
        interpolated_ds = xr.Dataset({
            SURFACE_WIND_SPEED: interpolated_da
        })

        # Save interpolated data to NetCDF
        interpolated_ds.to_netcdf(file_path)
        print(f'Created: {file_path}')
    return
