import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
from scr.Region import Country
from scr.analysis.Information import FileInformation
from scr.pre_processing.utils_pre_processing import LATITUDE_NAME, LONGITUDE_NAME
from scr.analysis.utils import save_figures, get_time_average_wind_speed
from scr.settings import COUNTRY, SURFACE_WIND_SPEED

# Constants for folder and plot settings
FOLDER_NAME = 'Grid Averages/'
LABEL_FONT_SIZE = 26
FIG_WIDTH = 8
FIG_HEIGHT = 10
LINE_COLOR = 'grey'
X_LABEL = 'Longitudes'
Y_LABEL = 'Latitudes'
LEGEND_LABEL = 'Average wind speed in m/s'


def plot_first_frames(files_information):
    """
    Plots the first time step (frame) of the wind speed data for each model and ERA5.

    Parameters:
    - files_information: Contains information about the file paths for the models and ERA5 data.
    """
    file_paths = list(files_information.era5_files.values())[:1] + files_information.files

    # Loop over the file paths to plot the first frame
    for file_path in file_paths:
        data = xr.open_dataset(file_path)[SURFACE_WIND_SPEED]
        data = data.isel(time=0)  # Get the first time step
        data.plot()

        model_information = FileInformation(file_path, files_information.run_comparison)
        plt.title(model_information.model_label)

        # Save the figure with the appropriate file name
        name = f'Grid_{files_information.title}_{model_information.model_label}'
        save_figures({name: plt}, FIG_WIDTH, FIG_HEIGHT, files_information, FOLDER_NAME)


def plot_grids_with_average_wind_speeds(files_information):
    """
    Plots wind speed grids for different models and ERA5, with color-coded wind speeds at different locations.

    Parameters:
    - files_information: Contains information about the model and ERA5 files and region settings.
    """
    # Determine the grid extension and properties based on the region
    country = Country(COUNTRY)
    if 'Germany' in files_information.region:
        region_object = country
        distance_between_coordinate_labels = 2
        line_width = 0.001
    elif 'Europe' in files_information.region:
        region_object = country.continent
        distance_between_coordinate_labels = 10
        line_width = 0.0001
    else:
        raise ValueError(f'Region {files_information.region} not recognized')

    if 'Turbine' in files_information.region:
        line_width = 0.3

    # Get the geographical boundaries of the region
    poly = region_object.geometry
    min_lon, min_lat, max_lon, max_lat = (region_object.min_lon, region_object.min_lat,
                                          region_object.max_lon, region_object.max_lat)

    # Initialize variables to track the maximum wind speed and cell sizes
    max_average = 0
    max_cell_size_lat = 0
    max_cell_size_lon = 0

    file_paths = list(files_information.era5_files.values())[:1] + files_information.files

    # Find the maximum wind speed and cell sizes across all files
    for file_path in file_paths:
        wind_speed = get_time_average_wind_speed(file_path, files_information)
        max_annual_wind_speed = np.nanmax(wind_speed)
        if max_annual_wind_speed > max_average:
            max_average = max_annual_wind_speed

        # Calculate cell sizes if not 'Turbine' region
        if not 'Turbine' in files_information.region:
            latitudes = wind_speed[LATITUDE_NAME].values
            longitudes = wind_speed[LONGITUDE_NAME].values
            cell_size_lat = abs(abs(latitudes[1]) - abs(latitudes[0]))
            if cell_size_lat > max_cell_size_lat:
                max_cell_size_lat = cell_size_lat
            cell_size_lon = abs(abs(longitudes[1]) - abs(longitudes[0]))
            if cell_size_lon > max_cell_size_lon:
                max_cell_size_lon = cell_size_lon

    # Adjust the boundaries to account for the maximum cell sizes
    if not 'Turbine' in files_information.region:
        min_lat -= max_cell_size_lat / 2
        max_lat += max_cell_size_lat / 2
        min_lon -= max_cell_size_lon / 2
        max_lon += max_cell_size_lon / 2

    # Plot the legend for wind speed
    plot_legend(files_information, max_average)

    # Create and save the grid plots for each model and ERA5 file
    for file_path in file_paths:
        model_information = FileInformation(file_path, files_information.run_comparison)
        wind_speed = get_time_average_wind_speed(file_path, files_information)

        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), subplot_kw={'projection': ccrs.PlateCarree()})

        # Plot for wind speed at turbine locations
        if 'Turbine' in files_information.region:
            latitudes, longitudes, values = get_turbine_coordinates(wind_speed)
            ax.scatter(longitudes, latitudes, c=values / max_average, s=8, cmap='YlGnBu',
                       edgecolor=LINE_COLOR, linewidth=line_width)
            ax.add_geometries(poly, crs=ccrs.PlateCarree(), facecolor='none', edgecolor='black')
        # Plot for wind speed at original resolution
        else:
            latitudes = wind_speed[LATITUDE_NAME].values
            longitudes = wind_speed[LONGITUDE_NAME].values
            cell_size_lat = abs(abs(latitudes[1]) - abs(latitudes[0]))
            cell_size_lon = abs(abs(longitudes[1]) - abs(longitudes[0]))
            latitudes, longitudes, values = get_coordinates(wind_speed, latitudes, longitudes)
            for lon, lat, value in zip(longitudes, latitudes, values):
                rect = Rectangle((lon - cell_size_lon / 2, lat - cell_size_lat / 2),
                                 cell_size_lon, cell_size_lat,
                                 edgecolor=LINE_COLOR,
                                 facecolor=plt.cm.YlGnBu(value / max_average),
                                 linewidth=line_width)
                ax.add_patch(rect)

        ax.add_geometries(poly, crs=ccrs.PlateCarree(), facecolor='none', edgecolor='black')

        # Plot the grids and save the figure
        plot_grids(fig, ax, min_lon, max_lon, min_lat, max_lat, distance_between_coordinate_labels)
        name = f'Grid_{files_information.title}_{model_information.model_label}'
        save_figures({name: fig}, FIG_WIDTH, FIG_HEIGHT, files_information, FOLDER_NAME)


def get_turbine_coordinates(ws_values):
    """
    Extracts the turbine coordinates and associated wind speed values from the dataset.

    Parameters:
    - ws_values: The wind speed data.

    Returns:
    - lats: The latitudes of the turbines.
    - lons: The longitudes of the turbines.
    - values: The wind speed values at the turbine locations.
    """
    lats = ws_values[LATITUDE_NAME].values
    lons = ws_values[LONGITUDE_NAME].values
    values = ws_values.values
    return lats, lons, values


def plot_grids(fig, ax, min_lon, max_lon, min_lat, max_lat, distance_between_coordinate_labels):
    """
    Plots the grid boundaries, labels, and ticks for the wind speed map.

    Arguments:
    - fig: The figure object to plot.
    - ax: The axes object for plotting.
    - min_lon: The minimum longitude for the grid.
    - max_lon: The maximum longitude for the grid.
    - min_lat: The minimum latitude for the grid.
    - max_lat: The maximum latitude for the grid.
    - distance_between_coordinate_labels: The distance between coordinate labels.
    """
    ax.set_extent([min_lon, max_lon, min_lat, max_lat])

    xticks = np.arange(int(min_lon) + 1, int(max_lon) + 1, distance_between_coordinate_labels)
    ax.set_xticks(xticks)
    ax.set_xticklabels(['{}'.format(tick) for tick in xticks], ha='right')
    ax.set_xlabel(X_LABEL, fontsize=LABEL_FONT_SIZE)

    yticks = np.arange(int(min_lat) + 1, int(max_lat), distance_between_coordinate_labels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(['{}'.format(tick) for tick in yticks])
    ax.set_ylabel(Y_LABEL, fontsize=LABEL_FONT_SIZE)
    ax.set_aspect(1.5)

    ax.tick_params(axis='both', which='major', labelsize=LABEL_FONT_SIZE - 4)
    ax.tick_params(axis='both', which='minor', labelsize=LABEL_FONT_SIZE - 6)
    fig.tight_layout()


def get_coordinates(ws_values, latitudes, longitudes):
    """
    Extracts the grid coordinates and wind speed values for all valid grid cells.

    Arguments:
    - ws_values: The wind speed data.
    - latitudes: The list of latitudes.
    - longitudes: The list of longitudes.

    Returns:
    - lats: The list of latitudes with valid wind speed data.
    - lons: The list of longitudes with valid wind speed data.
    - values: The list of wind speed values for valid grid cells.
    """
    lats = []
    lons = []
    values = []
    for lat in latitudes:
        for lon in longitudes:
            wind_speed = float(ws_values.sel({LATITUDE_NAME: lat, LONGITUDE_NAME: lon}).values)
            if not np.isnan(wind_speed):
                values.append(wind_speed)
                lats.append(lat)
                lons.append(lon)
    return lats, lons, values


def plot_legend(files_information, max_average):
    """
    Plots and saves a color legend for wind speed maps.
    """
    fig, _ = plt.subplots(figsize=(30, 0.2))
    sm = plt.cm.ScalarMappable(cmap='YlGnBu', norm=plt.Normalize(vmin=0, vmax=max_average))
    sm.set_array([])  # Hack to prevent empty array error
    cbar = plt.colorbar(sm, orientation='horizontal', aspect=50)
    cbar.ax.tick_params(labelsize=LABEL_FONT_SIZE - 2)
    cbar.set_label(LEGEND_LABEL, labelpad=15, fontsize=LABEL_FONT_SIZE + 2)
    cbar.ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
    plt.axis('off')

    # Save the figure with the appropriate file name
    save_figures({f'Grid Legend {files_information.title}': fig}, 30, 4, files_information, FOLDER_NAME)
