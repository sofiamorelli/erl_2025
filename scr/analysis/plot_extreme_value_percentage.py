import numpy as np

# Import necessary functions and classes for file handling, plotting, and calculations
from scr.analysis.Information import FileInformation
from scr.analysis.utils import (save_figures, configure_scatter_plot, get_flattened_wind_speed_and_resolution,
                                plot_scatter_point,
                                SCATTER_PLOT_WIDTH, SCATTER_PLOT_HEIGHT, plot_regression_line)

# Folder name to save summary statistics plots and results
FOLDER_NAME = 'Extreme wind speeds'

# List of statistical measures to plot
PLOT_NAMES = ['Low ws', 'High ws', 'Number Zeros', 'Average min values']


def plot_extreme_wind_speed_percentage(files_information):
    """
    Plots various extreme value metrics for ERA5 and model data.
    Also plots regression lines and saves figures for comparison.

    Parameters:
    - files_information: Contains metadata about the ERA5 and model files.
    """
    figures = {}  # Dictionary to store figure objects for each plot
    axes = []  # List to store axis objects for each plot

    # Configure scatter plots for each statistical measure
    for y_label in PLOT_NAMES:
        fig, ax = configure_scatter_plot(y_label)
        figures[f'{y_label} {files_information.title}'] = fig
        axes.append(ax)

    # Get file paths for ERA5 and models
    era5_file_path = list(files_information.era5_files.values())[0]
    file_paths = [era5_file_path] + files_information.files

    # Initialize dictionaries and lists to store biases and summary statistics
    spatial_resolutions = []
    low_speeds = []
    high_speeds = []
    numbers_zero = []
    min_values = []

    # Loop through each file (ERA5 and models) and compute the summary statistics
    for file_path in file_paths:
        model_information = FileInformation(file_path, files_information.run_comparison)
        low_ws, high_ws, number_zero, min_value, number_data_points = plot_extremes(file_path, axes, files_information,
                                                            model_information)
        spatial_resolutions.append(number_data_points)
        low_speeds.append(low_ws)
        high_speeds.append(high_ws)
        numbers_zero.append(number_zero)
        min_values.append(min_value)

    # Convert the spatial resolutions list to a numpy array
    x = np.array(spatial_resolutions)

    # Plot regression lines for the summary statistics
    for i, v in enumerate([low_speeds, high_speeds, numbers_zero, min_values]):
        plot_regression_line(x, v, axes[i], PLOT_NAMES[i], files_information, FOLDER_NAME)

    # Save the generated figures
    save_figures(figures, SCATTER_PLOT_WIDTH, SCATTER_PLOT_HEIGHT, files_information, FOLDER_NAME)

def plot_extremes(file_path, axes, files_information, model_information):
    """
    Computes and plots the extreme value metrics for a given model file.

    Parameters:
    - file_path: The path to the model file to analyze.
    - axes: List of axis objects for plotting the summary statistics.
    - files_information: Contains metadata for the files being compared.
    - model_information: Contains metadata specific to the model being analyzed.

    Returns:
    - low_ws: Percentage of wind speed below cut-in.
    - high_ws: Percentage of wind speed above cut-off.
    """
    # Retrieve and flatten the model wind speed data
    wind_speed, spatial_resolution, _ = get_flattened_wind_speed_and_resolution(file_path, files_information)
    total_number_ws = len(wind_speed)
    sorted_ws = np.sort(wind_speed)
    cut_in = files_information.turbine.cut_in
    index_below_cut_in = np.argmax(sorted_ws > cut_in)
    low_ws = index_below_cut_in / total_number_ws
    cut_off = files_information.turbine.cut_off
    index_above_cut_off = np.argmax(sorted_ws > cut_off)
    high_ws = (total_number_ws - index_above_cut_off) / total_number_ws

    number_zeros = np.argmax(sorted_ws > 0.1)
    min_value = np.mean(np.sort(wind_speed)[:100])

    # Prepare a dictionary to store the values to plot on each axis
    plot_data = {
        axes[0]: low_ws,
        axes[1]: high_ws,
        axes[2]: number_zeros,
        axes[3]: min_value
    }

    print(model_information.model_label, low_ws, high_ws, number_zeros, min_value)

    # Plot each statistic on the corresponding axis for the model
    for ax, value in plot_data.items():
        plot_scatter_point(ax, spatial_resolution, value, model_information)

    # Return the calculated statistics for further use
    return low_ws, high_ws, number_zeros, min_value, spatial_resolution
