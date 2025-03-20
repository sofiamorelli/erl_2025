import os

import numpy as np
from matplotlib import pyplot as plt

from scr.analysis.Information import FileInformation
from scr.analysis.utils import (
    configure_scatter_plot, configure_time_series_plot, save_figures,
    set_legend_at_top,
    plot_scatter_point, SCATTER_PLOT_WIDTH, SCATTER_PLOT_HEIGHT, get_wind_power_per_time_step,
    adapt_axes, get_wind_power_in_tera_watt, get_wind_power_in_giga_watt, save_heat_plot_relative_biases,
    calculate_bias, reverse_dict_order, get_wind_speed, compute_wind_power, SCATTER_PLOT_ANNOTATION_SIZE,
    wind_profile_power_law_with_cmip5_correction, get_npy_file_path
)
from scr.settings import SURFACE_WIND_SPEED, LATITUDE_NAME, LONGITUDE_NAME

# Constants for plotting
FOLDER_NAME = 'Cumulative Wind Power'
GROUP_PLOTS = {FOLDER_NAME: 'Cumulative wp in TW',
               f'{FOLDER_NAME} Difference': 'Cumulative wp differences in TW'}
AXIS_LABEL = 'Total annual wind power in TW'
DIFFERENCE_WIDTH = 30
DIFFERENCE_HEIGHT = 5
LABEL_SIZE = 16


def plot_only_cumulative_wp_for_large_data(files_information):
    """
    Plot cumulative wind power over time for large datasets.
    Includes the total wind power ratio relative to ERA5 data and computes relative biases.
    """
    # Set up figure and axes for cumulative wind power plots
    axes_group = {}
    figures = {}
    title, y_label = FOLDER_NAME, 'Cumulative wind power ratio'
    fig_name = f'{title} {files_information.turbine.turbine_name} {files_information.title}'
    figures, axes_group = configure_time_series_plot(figures, axes_group, 'Time', y_label, fig_name)

    # Initialize variables for comparison with ERA5
    era5_file_path = list(files_information.era5_files.values())[0]
    file_paths = [era5_file_path] + files_information.files

    ratios = []
    relative_biases = {}
    era5_total_wind_power = 0
    rel_bias = 0

    # Process each file and calculate wind power, relative bias, and cumulative wind power
    for file_path in file_paths:
        model_information = FileInformation(file_path, files_information.run_comparison)
        wind_power_values, time = get_wind_power_from_file(file_path, files_information)

        # Compare with ERA5 data
        if 'ERA5' in model_information.model_label:
            era5_total_wind_power = np.sum(wind_power_values)
        else:
            total_wind_power = np.sum(wind_power_values)
            ratios.append(total_wind_power / era5_total_wind_power)
            rel_bias = calculate_bias(total_wind_power, era5_total_wind_power)
            relative_biases[model_information.model_label] = rel_bias

        # Calculate cumulative sum difference relative to ERA5
        time, wind_power_values = get_cumsum_difference(time, wind_power_values, files_information)

        # Add label with relative bias if applicable
        label = model_information.model_label if not rel_bias \
            else f'{model_information.model_label}: {rel_bias:.2f}%' if rel_bias <= 0 \
            else f'{model_information.model_label}:  +{rel_bias:.2f}%'
        axes_group[f'axs_{fig_name}'].plot(time, wind_power_values,
                                           label=label, **model_information.arguments)

    # Print variance of ratios for comparison
    print(f'{np.var(ratios):.4f}')

    # Set the legend at the top of the plot
    set_legend_at_top(axes_group.values(), 12, 2)

    # Save the figure with cumulative wind power
    save_figures(figures, SCATTER_PLOT_WIDTH, SCATTER_PLOT_HEIGHT, files_information, FOLDER_NAME)

    # Save heat plot showing relative biases
    save_heat_plot_relative_biases(reverse_dict_order(relative_biases), 'Relative total wind power bias',
                                   files_information, FOLDER_NAME)


def get_cumsum_difference(time, wind_power_values, files_information):
    """
    Calculate the cumulative sum of wind power values and their differences
    with respect to ERA5 data for a given time range.
    """
    for i in range(2):
        # Get the corresponding ERA5 file and its wind power data
        era5_file_path = list(files_information.era5_files.values())[i]
        era5_wind_power, era5_time = get_wind_power_from_file(era5_file_path, files_information)
        era5_wind_power = era5_wind_power.flatten()

        # Find overlapping time between ERA5 and model data
        time_overlap = np.intersect1d(time, era5_time)
        if time_overlap.size:
            mask = np.isin(time, time_overlap)
            era5_mask = np.isin(era5_time, time_overlap)
            wind_power_values = np.cumsum(wind_power_values[mask]) / np.cumsum(era5_wind_power[era5_mask])
            return time_overlap, wind_power_values


def get_wind_power_file_name(file_path, files_information):
    """
    Generate the file path for wind power data based on model and hub height.
    """
    height = files_information.hub_height
    turbine_name = files_information.turbine.turbine_name
    base_file_path = get_npy_file_path(file_path)
    extension = '.npy'
    return (base_file_path.replace(SURFACE_WIND_SPEED, 'wp')
            + f'_{height}m-{turbine_name}{extension}')


def get_wind_power_from_file(file_path, files_information, chunk_size=1000):
    """
    Retrieve wind power data from a file, processing the data in chunks.
    If the file doesn't exist, compute and save the wind power.
    """
    time = save_time_values_in_file(file_path)
    wind_power_file_path = get_wind_power_file_name(file_path, files_information)

    # If the wind power data doesn't exist, compute it
    if not os.path.isfile(wind_power_file_path):
        wind_speed = get_wind_speed(file_path)
        temporal_resolution = time.size
        spatial_resolution = int(wind_speed[SURFACE_WIND_SPEED].size / temporal_resolution)
        wind_power_values = []

        # Process the data in chunks for efficiency
        for start in range(0, temporal_resolution, chunk_size):
            end = min(start + chunk_size, temporal_resolution)
            wind_speed_chunk = wind_speed[SURFACE_WIND_SPEED][start:end]
            wind_speed_chunk = wind_profile_power_law_with_cmip5_correction(wind_speed_chunk, file_path,
                                                                            files_information)
            wind_speed_chunk = wind_speed_chunk.stack({'coordinates': (LATITUDE_NAME, LONGITUDE_NAME)}).load().values
            wind_speed_chunk = np.nan_to_num(wind_speed_chunk)

            # Compute wind power for the chunk
            wind_power_chunk = np.apply_along_axis(
                lambda ws: compute_wind_power(ws, files_information.turbine),
                axis=1,
                arr=wind_speed_chunk
            )
            wind_power_values.append(np.sum(wind_power_chunk, axis=1) / spatial_resolution)

        # Save the computed wind power values
        with open(wind_power_file_path, 'wb') as f:
            np.save(f, wind_power_values)
    else:
        # If wind power data already exists, load it
        wind_power_values = np.load(wind_power_file_path, allow_pickle=True)
        base_file_path = get_npy_file_path(file_path)
        height = files_information.hub_height
        np.savez(f'{base_file_path}_{height}m_wind_power.npz', time=time, wp=wind_power_values)

    return np.concatenate(wind_power_values), time


def save_time_values_in_file(file_path):
    """
    Save time values from the wind speed data into a file for later use.
    """
    base_file_path = get_npy_file_path(file_path)
    time_file_path = f'{base_file_path}_time.npy'

    # If time values are not already saved, extract and save them
    if not os.path.isfile(time_file_path):
        wind_speed = get_wind_speed(file_path)
        time = wind_speed.time.values
        with open(time_file_path, 'wb') as f:
            np.save(f, time)
        return time
    else:
        # If time values are already saved, load them
        time = np.load(time_file_path)
        return time

def plot_cumulative_wind_power_ma(files_information, power_curve_information, additional_folder_name=''):
    """
        Plots the cumulative wind power for different models and compares it to ERA5 data.
        - Retrieves wind power data from ERA5.
        - Creates scatter plots showing the total wind power for each model compared to ERA5.
        - Generates cumulative wind power plots and the difference in wind power over time for each model.
        - Saves the generated plots and relative bias values.
    """
    full_folder_name = f'{FOLDER_NAME}/{additional_folder_name}'

    # Get ERA5 data for comparison (retrieve first ERA5 file)
    era5_file_path = list(files_information.era5_files.values())[0]
    era5_time, era5_wind_power = get_wind_power_per_time_step(era5_file_path, 'Reanalysis',
                                                              files_information.region, power_curve_information)
    # Calculate the total wind power for ERA5
    era5_total_wind_power = get_wind_power_in_tera_watt(np.sum(era5_wind_power))

    # Set maximum wind power to ERA5 value initially
    max_total_wp = era5_total_wind_power

    # Configure scatter plot for total wind power vs. spatial resolution
    fig1, ax1 = configure_scatter_plot(AXIS_LABEL)

    # Configure group plots for cumulative wind power and difference in wind power
    axes_group = {}
    figures = {}
    for title, y_label in GROUP_PLOTS.items():
        figures, axes_group = configure_time_series_plot(figures, axes_group, 'Time', y_label,
                                                         f'{title} {files_information.title}')

    # Update group plots with ERA5 data
    update_group_plots(era5_time, era5_wind_power, era5_time, era5_wind_power, '', axes_group,
                       files_information, FileInformation(era5_file_path), power_curve_information)

    # Dictionary to store relative biases for each model
    relative_biases = {}

    # Loop through all model files
    for file_path in files_information.files:
        # Get model-specific information
        model_information = FileInformation(file_path, files_information.run_comparison)
        # Get wind power time series for the current model
        time, wind_power = get_wind_power_per_time_step(file_path, model_information.model_category,
                                                        files_information.region, power_curve_information)
        # Calculate the total wind power for the model
        total_wind_power = get_wind_power_in_tera_watt(np.sum(wind_power))

        # Add entry to scatter plot showing spatial resolution vs total wind power
        spatial_resolution = int(
            model_information.file_name.split('/')[-1].split('_')[0].replace(SURFACE_WIND_SPEED, ""))
        plot_scatter_point(ax1, spatial_resolution, total_wind_power, model_information)

        # Set new max wind power if the model has a higher value
        max_total_wp = max(total_wind_power, max_total_wp)

        # Calculate the relative bias compared to ERA5
        rel_bias = calculate_bias(total_wind_power, era5_total_wind_power)
        relative_biases[model_information.model_label] = rel_bias

        # Create an additional time series plot showing the wind power differences to ERA5
        create_difference_plot(time, wind_power, era5_time, era5_wind_power, model_information,
                               files_information, full_folder_name, power_curve_information)

        # Add the model's cumulative wind power to the group plots
        update_group_plots(time, wind_power, era5_time, era5_wind_power, rel_bias, axes_group,
                           files_information, model_information, power_curve_information)

    # Set legend at the top for scatter plot and save the figure
    set_legend_at_top([ax1], SCATTER_PLOT_ANNOTATION_SIZE)
    save_figures({f'Total Wind Power {files_information.title}': fig1},
                 SCATTER_PLOT_WIDTH, SCATTER_PLOT_HEIGHT, files_information, full_folder_name)

    # Set legend for group plots and save those figures
    set_legend_at_top(axes_group.values(), 12)
    save_figures(figures, SCATTER_PLOT_WIDTH, SCATTER_PLOT_HEIGHT, files_information, full_folder_name)

    # Save a heatmap of relative biases
    save_heat_plot_relative_biases(reverse_dict_order(relative_biases), 'Relative total wind power bias',
                                   files_information, full_folder_name)


def update_group_plots(time, wind_power, era5_time, era5_wind_power, rel_bias, axes_group, files_information,
                       model_information, power_curve_information):
    """
    Updates the group plots with the cumulative wind power and the difference in wind power between the model and ERA5.

    Parameters:
    - time: Time values for the model's wind power.
    - wind_power: Wind power values for the model.
    - era5_time: Time values for the ERA5 data.
    - era5_wind_power: Wind power values for the ERA5 data.
    - rel_bias: Relative bias of the model's total wind power compared to ERA5.
    - axes_group: Dictionary of axes for the group plots.
    - files_information: Information about the files and region for the comparison.
    - model_information: Information about the model.
    - power_curve_information: Power curve information for calculations.
    """
    # Update cumulative wind power plot
    fig_name = f'{list(GROUP_PLOTS.keys())[0]} {files_information.title}'
    label = model_information.model_label if not rel_bias else f'{model_information.model_label}: {rel_bias:.2f}%'
    axes_group[f'axs_{fig_name}'].plot(time, np.cumsum(get_wind_power_in_tera_watt(wind_power)),
                                       label=label, **model_information.arguments)

    # Update difference in wind power plot (model - ERA5)
    fig_name_diff = f'{list(GROUP_PLOTS.keys())[1]} {files_information.title}'
    time_overlap, wind_power, era5_wind_power = get_wind_power_on_same_time_range(
        time, wind_power, era5_time, era5_wind_power, files_information, power_curve_information)
    axes_group[f'axs_{fig_name_diff}'].plot(time_overlap, get_wind_power_in_tera_watt(
        get_signed_difference(np.cumsum(wind_power), np.cumsum(era5_wind_power))),
                                            label=model_information.model_label, **model_information.arguments)


def create_difference_plot(time, wind_power, era5_time, era5_wind_power, model_information, files_information,
                           full_folder_name, power_curve_information):
    """
    Creates and saves a plot showing the difference in wind power (model - ERA5) over time.

    Parameters:
    - time: Time values for the model's wind power.
    - wind_power: Wind power values for the model.
    - era5_time: Time values for the ERA5 data.
    - era5_wind_power: Wind power values for the ERA5 data.
    - model_information: Information about the model.
    - files_information: Information about the files and region for the comparison.
    - full_folder_name: Directory where figures will be saved.
    - power_curve_information: Power curve information for calculations.
    """
    fig, ax = plt.subplots(figsize=(DIFFERENCE_WIDTH, DIFFERENCE_HEIGHT))

    # Align the time intervals of model and ERA5 data
    time_overlap, wind_power_overlap, era5_wind_power_overlap = get_wind_power_on_same_time_range(
        time, wind_power, era5_time, era5_wind_power, files_information, power_curve_information)

    # Plot the difference (model wind power - ERA5 wind power)
    ax.plot(time_overlap,
            get_wind_power_in_giga_watt(get_signed_difference(wind_power_overlap, era5_wind_power_overlap)),
            **model_information.arguments)

    ax.set_xlabel('Time', fontsize=LABEL_SIZE)
    ax.set_ylabel('Wind power in GW', fontsize=LABEL_SIZE)
    adapt_axes(ax, LABEL_SIZE)
    fig.tight_layout()

    # Title for the figure
    title = f'wp Differences {files_information.title} {model_information.model_label}'
    save_figures({title: fig}, DIFFERENCE_WIDTH, DIFFERENCE_HEIGHT, files_information, full_folder_name)


def get_wind_power_on_same_time_range(time, wind_power, era5_time, era5_wind_power, files_information,
                                      power_curve_information):
    """
    Ensures that the time intervals of the model and ERA5 data overlap. If not, it tries to retrieve data
    from another ERA5 file to ensure a proper time overlap.

    Returns:
    - time_overlap: The overlapping time range.
    - wind_power: The model's wind power values for the overlapping time range.
    - era5_wind_power: The ERA5 wind power values for the overlapping time range.
    """
    # Get the intersection of time intervals from the model and ERA5 data
    time_overlap = np.intersect1d(time.values, era5_time.values)

    # If no overlap, use the second ERA5 file to try to get an overlap
    if not time_overlap.size:
        era5_file_path = list(files_information.era5_files.values())[1]
        era5_time, era5_wind_power = get_wind_power_per_time_step(era5_file_path, 'Reanalysis',
                                                                  files_information.region, power_curve_information)
        time_overlap = np.intersect1d(time.values, era5_time.values())

    # Create masks to extract the overlapping data
    mask = np.isin(time, time_overlap)
    era5_mask = np.isin(era5_time, time_overlap)

    return time_overlap, wind_power[mask], era5_wind_power[era5_mask]


def get_signed_difference(array1, array2):
    """
    Calculate the signed difference between two arrays, where the difference is
    positive if array1 is greater than array2, and negative if the opposite is true.

    Returns:
    - signed_difference: The signed difference between the two arrays.
    """
    differences = np.abs(np.abs(array1) - np.abs(array2))
    return np.where(array1 < array2, -differences, differences)