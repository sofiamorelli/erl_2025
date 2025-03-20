import glob
import os

import numpy as np
import xarray as xr
import shapely.geometry as sgeom

from scr.Region import Country
from scr.pre_processing.tasks_download_files import TIME_POINTS
from scr.pre_processing.utils_pre_processing import LATITUDE_NAME, LONGITUDE_NAME, correct_coordinates, \
    restrict_data_by_region, get_wind_speeds_from_components, get_coordinate_names, get_country_and_continent_dir
from scr.settings import ORIGINAL_DATA_DIR, GLOBAL_MODEL_NAME, SURFACE_WIND_SPEED, U_COMPONENT_CMIP6, V_COMPONENT_CMIP6, \
    ERA5_NAME, START_YEAR, END_YEAR, COUNTRY, ERA5_FILE_NAME, FILE_NAME, REGIONAL_MODEL_NAME, BASE_DIR_DAT

from cdo import *
cdo.debug = True
cdo = Cdo()


def preprocess_climate_model_data(model_group=GLOBAL_MODEL_NAME):
    """
    Preprocesses climate model data for a specified model group.
    Generates wind speed files by merging time-ranged data, creating land masks,
    and calculating ensemble averages for the specified model group.

    Args:
        model_group {str}: The climate model group to process,
                           default is the global model name specified in the settings file
    """
    country = Country(COUNTRY)
    continent = country.continent
    new_directory_continent, new_directory_country = get_country_and_continent_dir(country, continent)
    model_dir = os.path.join(ORIGINAL_DATA_DIR, model_group)

    # Loop through folders in model directory to process each directory of climate data
    for root, dirs, _ in os.walk(model_dir):
        for dir_name in dirs:
            original_directory = os.path.join(root, dir_name)
            ws_file_name_continent = (FILE_NAME.replace('$MODEL$', dir_name)
                                      .replace('$CATEGORY$', model_group)
                                      .replace('$REGION$', continent.name).replace('$SIZE$', ''))

            # Process three runs of data
            for run in range(1, 4):
                ws_file_name_run = ws_file_name_continent.replace('$RUN$', str(run))
                ws_file = os.path.join(new_directory_continent, ws_file_name_run)
                if not os.path.isfile(ws_file):
                    create_merged_files(original_directory, dir_name, new_directory_continent, run, continent, ws_file)
                    create_wind_speed_file_when_u_and_v_given(ws_file)
                else:
                    print(f"Already exists: '{ws_file}'")

                if run == 1:
                    create_continent_land_files(new_directory_continent, ws_file_name_run, continent, model_group)

            create_ensemble_averages(new_directory_continent, dir_name, ws_file_name_continent)

    create_country_files(new_directory_continent, new_directory_country, country)


def create_merged_files(original_directory, new_directory_continent, run, continent, ws_file):
    """
    Merges files by variable and run, and applies continent boundary filters.
    Generates a single file by merging time series data.

    Args:
        original_directory {str}: Directory with the original data files.
        new_directory_continent {str}: Directory for saving processed continent data.
        run {int}: The model run identifier.
        continent {Region}: Region object, containing lat/lon bounds of the continent for data restriction.
        ws_file {str}: Final output file path for saving the wind speed data.
    """
    # Process each variable needed for wind speed computation
    for var in [SURFACE_WIND_SPEED, U_COMPONENT_CMIP6, V_COMPONENT_CMIP6]:
        file_path_continent_merged = ws_file.replace(SURFACE_WIND_SPEED, var)
        file_list = []

        # Traverse files in directory to create filtered and corrected data files
        for _, _, files in os.walk(original_directory):
            for file_name in files:
                if f'r{run}' in file_name and var in file_name:
                    file_list.append(create_files_with_corrected_continent_data(original_directory,
                                                                                new_directory_continent,
                                                                                file_name,
                                                                                continent))

        input_pattern = os.path.join(new_directory_continent, f'{var}*r{run}i1*.nc')
        matching_files = glob.glob(input_pattern)

        # Merge and clean temporary files if matches were found
        if matching_files:
            cdo.mergetime(input=" ".join(matching_files), output=file_path_continent_merged)
            print(f'Created: {file_path_continent_merged}')
            for file in file_list:
                os.remove(file)
                print(f'Deleted: {file}')


def create_files_with_corrected_continent_data(original_directory, new_directory, file_name, continent):
    """
    Processes and corrects individual files for a specific continent by applying filters and regridding.

    Args:
        original_directory (str): Source directory of files.
        new_directory (str): Directory for processed files.
        file_name (str): Name of the file being processed.
        continent (Country): Continent boundaries to apply to data.
    Returns:
        str: Path to the continent-filtered file.
    """
    file_path_original = os.path.join(original_directory, file_name)
    file_path_continent_single = os.path.join(new_directory, file_name)

    if not os.path.isfile(file_path_continent_single):
        # Initial processing to set time range and regrid if needed
        first_intermediate_file_path = os.path.join(new_directory, 'intermediate1.nc')
        try:
            cdo.selyear(f'{START_YEAR}/{END_YEAR}', input=file_path_original, output=first_intermediate_file_path)
        except:
            print(f'Check again the start year {START_YEAR} and end year {END_YEAR}.')

        first_intermediate_file_path = regrid_data(original_directory, new_directory, file_path_original,
                                                   first_intermediate_file_path)

        # Apply continent filter to spatial boundaries and correct coordinate format
        second_intermediate_file_path = os.path.join(new_directory, 'intermediate2.nc')
        cdo.sellonlatbox(continent.min_lon, continent.max_lon, continent.min_lat, continent.max_lat,
                         input=first_intermediate_file_path, output=second_intermediate_file_path)
        os.remove(first_intermediate_file_path)

        data = xr.open_dataset(str(second_intermediate_file_path))
        os.remove(second_intermediate_file_path)
        lat_name, lon_name = get_coordinate_names(data)
        data = correct_coordinates(data, lat_name, lon_name)

        # Additional processing for time format and data compatibility
        if 'MOHC' in original_directory:
            data = data.convert_calendar("standard", align_on='year', use_cftime=True)
        time = data.time.values.astype('datetime64[ns]')
        data['time'] = time
        if 'plev' in data.dims:
            data = data.isel(plev=0)
            data = data.drop('plev')

        data.to_netcdf(file_path_continent_single)
        print(f'Created: {file_path_continent_single}')
    return file_path_continent_single

def regrid_data(original_directory, new_directory, file_path_original, first_intermediate_file_path):
    """
        Regrids data based on source and destination model types, performing interpolation if necessary.

        Arguments:
            original_directory {str}: Path to the directory containing original data files.
            new_directory {str}: Path to the directory where regridded files will be saved.
            file_path_original {str}: Path to the original data file.
            first_intermediate_file_path {str}: Path to the first intermediate file for regridding.

        Returns:
            intermediate_file_path {str}: Path to the final regridded file.
    """
    if 'MOHC' in file_path_original and REGIONAL_MODEL_NAME not in original_directory:
        intermediate_file_path = os.path.join(new_directory, 'intermediate_MOHC.nc')
        # Interpolates data to temperature grid
        temp_file = str(get_temperature_file(original_directory)[0])
        cdo.remapbil(temp_file, input=first_intermediate_file_path, output=intermediate_file_path)
        print(f'Re-gridded MOHC data.')
        os.remove(first_intermediate_file_path)
        return intermediate_file_path

    elif REGIONAL_MODEL_NAME in original_directory:
        intermediate_file_path = os.path.join(new_directory, 'intermediate_Regional.nc')
        # Interpolates data to regular grid
        cdo.remapbil(f"{BASE_DIR_DAT}grid.txt", input=first_intermediate_file_path, output=intermediate_file_path)
        os.remove(first_intermediate_file_path)
        print(f'Re-gridded CORDEX data.')
        return intermediate_file_path
    else:
        return first_intermediate_file_path

def get_temperature_file(original_directory):
    """
        Finds and returns the temperature file in the specified directory.

        Arguments:
            original_directory {str}: Path to the directory containing original data files.

        Returns:
            temp_file {list}: List containing paths to temperature files.

        Raises:
            Exception: If no temperature file or multiple files are found.
    """
    temp_file = []
    for file in os.listdir(original_directory):
        if 'tas' in file:
            temp_file.append(os.path.join(original_directory, file))
    if not temp_file:
        raise Exception(f'Could not find temperature file to interpolate.')
    elif len(temp_file) > 1:
        raise Exception(f'More than one temperature file found.')
    return temp_file

def create_wind_speed_file_when_u_and_v_given(ws_file):
    """
        Creates a wind speed file from u and v component files if it doesn't already exist.

        Arguments:
            ws_file {str}: Path to the desired wind speed file.
    """
    u_file = ws_file.replace(SURFACE_WIND_SPEED, U_COMPONENT_CMIP6)
    v_file = ws_file.replace(SURFACE_WIND_SPEED, V_COMPONENT_CMIP6)
    if not os.path.isfile(ws_file) and (os.path.isfile(u_file) and os.path.isfile(v_file)):
        command = f"-sqrt -add -sqr -selname,{U_COMPONENT_CMIP6} {u_file} -sqr -selname,{V_COMPONENT_CMIP6} {v_file}"
        cdo.chname(f"{U_COMPONENT_CMIP6},{SURFACE_WIND_SPEED}", input=command, output=ws_file)
        print(f'Created: {ws_file}')
        os.remove(u_file)
        print(f'Deleted: {u_file}')
        os.remove(v_file)
        print(f'Deleted: {v_file}')

def create_ensemble_averages(new_directory_continent, dir_name, ws_file_name):
    """
        Creates an ensemble average file if there are multiple files for a region.

        Arguments:
            new_directory_continent {str}: Directory for the continent-level output files.
            dir_name {str}: Directory name to be used in file search pattern.
            ws_file_name {str}: File name pattern for the wind speed files.
    """
    input_pattern_runs = os.path.join(new_directory_continent, f'*{dir_name}_r*.nc')
    merging_files_for_average = glob.glob(input_pattern_runs)
    file_path_ensemble_average = os.path.join(new_directory_continent, ws_file_name.replace('r$RUN$', 'average'))
    if not os.path.isfile(file_path_ensemble_average) and len(merging_files_for_average) > 1:
        cdo.ensmean(input=" ".join(merging_files_for_average), output=file_path_ensemble_average)
        print(f'Created: {file_path_ensemble_average}')

def create_country_files(new_directory_continent, new_directory_country, country):
    """
        Creates country-specific files from continent-level data by extracting relevant regions.

        Arguments:
            new_directory_continent {str}: Directory with continent-level files.
            new_directory_country {str}: Directory for saving country-specific files.
            country {Country}: Country object with geographic information.
    """
    for file_name in os.listdir(new_directory_continent):
        if SURFACE_WIND_SPEED in file_name and country.continent.name in file_name and GLOBAL_MODEL_NAME in file_name:
            file_path_continent = os.path.join(new_directory_continent, file_name)
            file_path_country = os.path.join(new_directory_country,
                                             file_name.replace(country.continent.name, country.name))
            intermediate_file_path = os.path.join(new_directory_country, 'intermediate.nc')
            cdo.sellonlatbox(country.min_lon, country.max_lon, country.min_lat, country.max_lat,
                             input=file_path_continent,
                             output=intermediate_file_path)
            data = xr.open_dataset(intermediate_file_path)
            lat_name, lon_name = get_coordinate_names(data)
            data.rename({lat_name: LATITUDE_NAME, lon_name: LONGITUDE_NAME})
            temporal_resolution = data.time.size
            all_wind_speeds = data[SURFACE_WIND_SPEED].values.flatten()
            wind_speed_without_nans = all_wind_speeds[~np.isnan(all_wind_speeds)]
            spatial_resolution = int(wind_speed_without_nans.size / temporal_resolution)
            file_path_country = file_path_country.replace(SURFACE_WIND_SPEED,
                                                          "{}{:04d}".format(SURFACE_WIND_SPEED, spatial_resolution))
            data.to_netcdf(file_path_country)
            os.remove(intermediate_file_path)
            print(f"Created: '{file_path_country}'")

def create_continent_land_files(new_directory_continent, ws_file_name, continent, model_class):
    """
        Creates land-only data files for continents by masking out ocean regions.

        Arguments:
            new_directory_continent {str}: Directory for continent-level files.
            ws_file_name {str}: Name of the wind speed file for the continent.
            continent {Continent}: Continent object with geographic information.
            model_class {str}: Model type (e.g., global or regional).
    """
    ws_file_path_continent = os.path.join(new_directory_continent, ws_file_name)
    new_directory_continent_land = (new_directory_continent
                                    .replace(continent.name, f'{continent.name}Land'))
    os.makedirs(new_directory_continent_land, exist_ok=True)
    ws_file_name_continent_land = ws_file_name.replace(continent.name, f'{continent.name}-Land')
    ws_file_path_continent_land = os.path.join(new_directory_continent_land,
                                               ws_file_name_continent_land)
    if not os.path.isfile(ws_file_path_continent_land):
        save_land_file(ws_file_path_continent, continent, ws_file_path_continent_land, model_class)
        print(f'Created: {ws_file_name_continent_land}')

def save_land_file(ws_file_path_region, region, ws_file_path_region_land, model_class):
    """
        Saves a land-only file by applying a mask to exclude ocean areas.

        Arguments:
            ws_file_path_region {str}: Path to the region's wind speed file.
            region {Region}: Region object with geographic boundaries.
            ws_file_path_region_land {str}: Output path for the land-only wind speed file.
            model_class {str}: Model type (e.g., global or regional).
    """
    if model_class == REGIONAL_MODEL_NAME:
        mask_file = create_mask(ws_file_path_region, region)
        cdo.ifthen(input=f'{mask_file} {ws_file_path_region}', output=ws_file_path_region_land)
    elif model_class == GLOBAL_MODEL_NAME:
        wind_speed = xr.open_dataset(ws_file_path_region)[SURFACE_WIND_SPEED]
        wind_speed = apply_mask(wind_speed, region)
        wind_speed.to_netcdf(ws_file_path_region_land)

def apply_mask(wind_speed, region):
    """
        Applies a region mask to filter out specified areas.

        Arguments:
            wind_speed {DataArray}: Wind speed data array.
            region {Region}: Region object providing with geographic bounds.

        Returns:
            DataArray: Masked wind speed data.
    """
    mask = np.ones_like(wind_speed.values, dtype=bool)
    for i, lat in enumerate(wind_speed[LATITUDE_NAME].values):
        for j, lon in enumerate(wind_speed[LONGITUDE_NAME].values):
            if is_outside_region_bounds(region.geometry, lat, lon):
                mask[:, i, j] = False
    return wind_speed.where(mask)



def create_mask(ws_file_path_region, region):
    """
        Creates a mask file for a specified region to distinguish land areas from others, if not already available.

        Arguments:
            ws_file_path_region {str}: Path to the wind speed dataset for the region.
            region {Region}: Region object containing geographical information.

        Returns:
            file_name {str}: Path to the created mask file.
    """
    file_name = os.path.join(ORIGINAL_DATA_DIR, 'mask.nc')
    if not os.path.isfile(file_name):
        wind_speed = xr.open_dataset(ws_file_path_region)[SURFACE_WIND_SPEED]
        lats = wind_speed[LATITUDE_NAME].values  # Retrieve latitude values
        lons = wind_speed[LONGITUDE_NAME].values  # Retrieve longitude values

        # Create a 2D meshgrid of latitudes and longitudes for spatial coverage
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Vectorized check to determine if points are within the region's bounds
        points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
        mask = np.array([not is_outside_region_bounds(region.geometry, lat, lon) for lat, lon in points])
        mask = mask.reshape(lat_grid.shape)  # Reshape mask to fit grid dimensions

        # Create DataArray for mask and Dataset to save as a NetCDF file
        mask_da = xr.DataArray(mask, dims=[LATITUDE_NAME, LONGITUDE_NAME],
                               coords={LATITUDE_NAME: lats, LONGITUDE_NAME: lons})
        mask_ds = xr.Dataset({"mask": mask_da})

        # Save the mask dataset to a NetCDF file
        mask_ds.to_netcdf(file_name)
    return file_name

def is_outside_region_bounds(geometry, lat, lon):
    """
        Checks if a geographic point is outside the bounds of a region.

        Arguments:
            geometry {Geometry}: Geometry object defining region bounds.
            lat {float}: Latitude of the point.
            lon {float}: Longitude of the point.

        Returns:
            {bool}: True if point is outside region bounds, False otherwise.
    """
    point = sgeom.Point(lon, lat)  # Create a point from longitude and latitude
    return not geometry.contains(point)  # Check if point is contained in geometry bounds


def preprocess_era5_data(model_group=GLOBAL_MODEL_NAME):
    """
        Processes ERA5 data by regridding and transforming coordinates and calculates wind speeds
        from the eastward and northward components u and v.

        Arguments:
            model_group {str}: Model group identifier, default is GLOBAL_MODEL_NAME.
    """
    country = Country(COUNTRY)  # Initialize country object for specified region
    continent = country.continent
    new_directory_continent, new_directory_country = get_country_and_continent_dir(country, continent)

    name = ERA5_NAME
    lat_name = 'latitude'
    lon_name = 'longitude'
    folder_path = os.path.join(ORIGINAL_DATA_DIR, name)

    # Loop over defined time points to process each time slice
    for time_name in TIME_POINTS.keys():
        file_name = f'{name}_{START_YEAR}-{END_YEAR}_{time_name}.grib'
        file_path = os.path.join(folder_path, file_name)
        data = xr.open_dataset(file_path, engine="cfgrib")  # Load ERA5 data in grib format

        # Correct coordinates and regrid u and v wind components to calculate wind speed
        data = correct_coordinates(data, lat_name, lon_name)
        desired_order = ['time', lat_name, lon_name]
        data = data.transpose(*desired_order).rename({lat_name: LATITUDE_NAME, lon_name: LONGITUDE_NAME})
        data = get_wind_speeds_from_components(data['u10'], data['v10'])  # Calculate wind speed from components
        data.name = SURFACE_WIND_SPEED

        # Save regridded data as NetCDF for the continent
        file_name_continent = ERA5_FILE_NAME.replace('$REGION$', continent.name).replace('$TIME$', time_name)
        file_path_continent = os.path.join(new_directory_continent, file_name_continent)
        if not os.path.isfile(file_path_continent):
            data.to_netcdf(file_path_continent)
            print(f'Created: {file_path_continent}')

        # Generate land-only files for the continent
        create_continent_land_files(new_directory_continent, file_name_continent, continent, model_group)

        # Restrict data to the specified country's region and calculate resolutions
        data = restrict_data_by_region(data, country)
        temporal_resolution = data.time.size
        wind_speed = data.values.flatten()
        wind_speed = wind_speed[~np.isnan(wind_speed)]
        spatial_resolution = int(wind_speed.size / temporal_resolution)

        # Create file name and path for country-specific data
        file_name_country = (ERA5_FILE_NAME.replace('$REGION$', country.name)
                             .replace('$TIME$', time_name)
                             .replace(SURFACE_WIND_SPEED,
                                      "{}{:04d}".format(SURFACE_WIND_SPEED, spatial_resolution)))
        file_path_country = os.path.join(new_directory_country, file_name_country)
        if not os.path.isfile(file_path_country):
            data.to_netcdf(file_path_country)
            print(f'Created: {file_path_country}')
    delete_idx_files(folder_path)  # Clean up auxiliary index files

def delete_idx_files(folder_path):
    """
        Deletes all index files (.idx) generated during preprocessing in a specified folder.

        Arguments:
            folder_path {str}: Path to the folder where .idx files are located.
    """
    pattern = os.path.join(folder_path, '*.idx')
    idx_files = glob.glob(pattern)

    # Attempt to delete each .idx file and handle any deletion errors
    for file_path in idx_files:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except OSError as e:
            print(f"Error deleting file {file_path}: {e}")
