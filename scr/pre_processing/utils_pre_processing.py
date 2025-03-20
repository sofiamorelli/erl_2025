import os

from scr.settings import BASE_DIR_DAT, LATITUDE_NAME, LONGITUDE_NAME

def get_country_and_continent_dir(country, continent):
    """
        Creates directories for a specified continent and country if they do not already exist.

        Arguments:
            country {Region}: Region object for the country, containing its name.
            continent {Region}: Region object for the continent, containing its name.

        Returns:
            new_directory_continent {str}: Path to the created continent directory.
            new_directory_country {str}: Path to the created country directory.
    """
    new_directory_continent = os.path.join(BASE_DIR_DAT, continent.name)  # Construct path for continent directory
    os.makedirs(new_directory_continent, exist_ok=True)  # Create continent directory if missing
    new_directory_country = os.path.join(BASE_DIR_DAT, country.name)  # Construct path for country directory
    os.makedirs(new_directory_country, exist_ok=True)  # Create country directory if missing
    return new_directory_continent, new_directory_country

def get_coordinate_names(data):
    """
        Determines the coordinate names for latitude and longitude from a dataset by trying different options.

        Arguments:
            data {Dataset}: Dataset object containing coordinate data.

        Returns:
            lat_name {str}: The name of the latitude coordinate in the dataset.
            lon_name {str}: The name of the longitude coordinate in the dataset.
    """
    for lat_name, lon_name in [('latitude', 'longitude'), ('lat', 'lon')]:  # Common name pairs for coordinates
        try:
            lon_name = data[lon_name].name  # Retrieve longitude name if available
            lat_name = data[lat_name].name  # Retrieve latitude name if available
            return lat_name, lon_name  # Return names if successfully found
        except KeyError:
            pass  # Continue to the next pair if names are not found

def correct_coordinates(wind_speed, lat_name, lon_name):
    """
        Adapts longitude values in the wind speed dataset to range from -180 to 180 if needed
        and sorts by latitude and longitude.

        Arguments:
            wind_speed {DataArray}: Wind speed dataset containing latitude and longitude data.
            lat_name {str}: Name of the latitude coordinate in the dataset.
            lon_name {str}: Name of the longitude coordinate in the dataset.

        Returns:
            wind_speed {DataArray}: Dataset with corrected and sorted coordinates.
    """
    wind_speed[lat_name] = wind_speed[lat_name].real  # Ensure latitude is real-valued
    longitudes = wind_speed[lon_name].real  # Retrieve real values for longitude

    # Adjust longitudes if values exceed 180 (common for 0-360 longitude format)
    if max(longitudes.values) > 180:
        longitudes = (longitudes + 180) % 360 - 180  # Convert to -180 to 180 range
    wind_speed[lon_name] = longitudes  # Update dataset with corrected longitudes

    return wind_speed.sortby(lon_name).sortby(lat_name)  # Sort data by latitude and longitude

def restrict_data_by_region(wind_speed, region_object):
    """
        Restricts the wind speed data to a specified region's latitude and longitude bounds.

        Arguments:
            wind_speed {DataArray}: Wind speed dataset.
            region_object {Region}: Region object with min and max lat/lon for restricting data.

        Returns:
            wind_speed {DataArray}: Dataset filtered to the region's lat/lon bounds.
    """
    restricted_latitudes = select_data_range(wind_speed, LATITUDE_NAME, region_object.min_lat, region_object.max_lat)
    restricted_longitudes = select_data_range(wind_speed, LONGITUDE_NAME, region_object.min_lon, region_object.max_lon)
    return wind_speed.sel({LATITUDE_NAME: restricted_latitudes, LONGITUDE_NAME: restricted_longitudes})

def select_data_range(data, sel_variable, sel_start, sel_end):
    """
        Selects a data range for a given coordinate variable (latitude or longitude).

        Arguments:
            data {DataArray}: Dataset with the coordinate variable.
            sel_variable {str}: Name of the coordinate variable.
            sel_start {float}: Start value for the data range.
            sel_end {float}: End value for the data range.

        Returns:
            DataArray: Range of data within the specified bounds for the variable.
    """
    start = data.sel({sel_variable: sel_start}, method="nearest")[sel_variable]  # Closest start value
    end = data.sel({sel_variable: sel_end}, method="nearest")[sel_variable]  # Closest end value
    return data[sel_variable].loc[start:end]  # Return selected range

def restrict_data_to_year(year, start_year_of_data_set, data_points_per_year=365, data_points_per_day=4):
    """
        Calculates start and end indices to restrict data to a specific year.

        Arguments:
            year {int}: Year to restrict data to.
            start_year_of_data_set {int}: First year of the dataset.
            data_points_per_year {int}: Number of data points expected per year (default is 365 days).
            data_points_per_day {int}: Number of data points recorded per day (default is 4).

        Returns:
            start {int}: Start index for data corresponding to the given year.
            end {int}: End index for data corresponding to the given year.
    """
    data_points_per_year = data_points_per_year * data_points_per_day  # Total points per year considering frequency
    number_years = year - start_year_of_data_set  # Calculate the offset in years
    start = data_points_per_year * number_years  # Calculate start index based on year offset

    # Adjust for leap days in the range
    if data_points_per_year == 365:
        leap_days = int((number_years - year % 4 + start_year_of_data_set % 4) / 4)
        if 2000 <= year < 2100 and 1800 < start_year_of_data_set <= 1900:
            leap_days -= 1  # Adjust for leap year exceptions (1900, 2100, etc.)
        start += leap_days * data_points_per_day
    end = start + data_points_per_year  # Calculate end index for year
    return start, end

def get_wind_speeds_from_components(u_data, v_data, chunk_size=1000):
    """
        Calculates wind speeds from u and v wind components using chunked data processing.

        Arguments:
            u_data {DataArray}: U-component of wind (east-west direction).
            v_data {DataArray}: V-component of wind (north-south direction).
            chunk_size {int}: Size of data chunks for memory-efficient processing (default is 1000).

        Returns:
            wind_speed {DataArray}: Calculated wind speeds from u and v components.
    """
    u_data_chunked = u_data.chunk({'time': chunk_size})  # Chunk u-component for processing
    v_data_chunked = v_data.chunk({'time': chunk_size})  # Chunk v-component for processing
    return (u_data_chunked ** 2 + v_data_chunked ** 2) ** 0.5  # Calculate wind speed as magnitude of u and v
