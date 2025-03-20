import numpy as np
from scipy import stats

# Import necessary functions and classes for file handling, plotting, and calculations
from scr.analysis.Information import FileInformation
from scr.analysis.utils import (save_figures, configure_scatter_plot, get_flattened_wind_speed_and_resolution,
                                plot_scatter_point,
                                save_heat_plot_relative_biases, set_legend_at_top, SCATTER_PLOT_ANNOTATION_SIZE,
                                calculate_bias, SCATTER_PLOT_WIDTH, SCATTER_PLOT_HEIGHT, reverse_dict_order,
                                plot_regression_line)
from scr.settings import ERA5_NAME, COLORS

# Folder name to save summary statistics plots and results
FOLDER_NAME = 'Summary Statistics'

# List of statistical measures to plot: Mean, Variance, Skewness, Kurtosis, and Average Max Value
PLOT_NAMES = ['Mean', 'Variance', 'Skewness', 'Kurtosis', 'Average Max Value']


def plot_summary_statistics_all_models(files_information):
    """
    Plots various summary statistics (Mean, Variance, Skewness, Kurtosis, Max Value) for ERA5 and model data.
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

    # Plot horizontal lines for ERA5 statistics
    era5_mean, era5_var = plot_era5_h_lines(files_information, axes)

    # Get file paths for ERA5 and models
    era5_file_path = list(files_information.era5_files.values())[0]
    file_paths = [era5_file_path] + files_information.files

    # Initialize dictionaries and lists to store biases and summary statistics
    mean_biases = {}
    spatial_resolutions = []
    means = []
    vars = []
    skews = []
    kurts = []
    max_values = []

    # Loop through each file (ERA5 and models) and compute the summary statistics
    for file_path in file_paths:
        model_information = FileInformation(file_path, files_information.run_comparison)
        mean, maxv, number_data_points, var, skew, kurt = plot_summary_stats(file_path, axes,
                                                                             files_information, model_information)
        spatial_resolutions.append(number_data_points)
        means.append(mean)
        vars.append(var)
        skews.append(skew)
        kurts.append(kurt)
        max_values.append(maxv)

        # Calculate the bias in the mean value of the model compared to ERA5
        mean_bias = calculate_bias(mean, era5_mean)
        if model_information.model_name != ERA5_NAME:
            mean_biases[model_information.model_label] = mean_bias

    # Convert the spatial resolutions list to a numpy array
    x = np.array(spatial_resolutions)

    # Plot regression lines for the summary statistics (Mean, Max Value, etc.)
    for i, v in enumerate([means, max_values]):  # Currently plotting only Mean and Max Value
        plot_regression_line(x, v, axes[i], PLOT_NAMES[i], files_information, FOLDER_NAME)

    # Position the legend at the top of the first plot for better visibility
    set_legend_at_top([axes[0]], SCATTER_PLOT_ANNOTATION_SIZE)

    # Save the generated figures
    save_figures(figures, SCATTER_PLOT_WIDTH, SCATTER_PLOT_HEIGHT, files_information, FOLDER_NAME)

    # Save a heatmap for the relative bias in the mean wind speed
    save_heat_plot_relative_biases(reverse_dict_order(mean_biases), 'Relative mean wind speed bias',
                                   files_information, FOLDER_NAME)


def plot_era5_h_lines(files_information, axes):
    """
    Plots horizontal lines on the scatter plots to represent the ERA5 statistics (mean, variance, skewness, kurtosis, max value).

    Parameters:
    - files_information: Contains metadata about the ERA5 files.
    - axes: List of axis objects to plot the horizontal lines on.

    Returns:
    - era5_mean: The mean wind speed from ERA5.
    - era5_var: The variance of wind speed from ERA5.
    """
    # Retrieve the ERA5 wind speed data
    era5_file_path = list(files_information.era5_files.values())[0]
    era5_wind_speed, _, _ = get_flattened_wind_speed_and_resolution(era5_file_path, files_information)

    # Calculate ERA5 statistics
    era5_mean = era5_wind_speed.mean()
    era5_var = era5_wind_speed.var()
    era5_skewness = stats.skew(era5_wind_speed)
    era5_kurtosis = stats.kurtosis(era5_wind_speed)
    era5_max = max(era5_wind_speed)

    # Set plotting arguments for horizontal lines (custom color and line width)
    arguments = {'color': COLORS[ERA5_NAME], 'linewidth': 0.2, 'zorder': 1}

    # Plot horizontal lines for each ERA5 statistic on the corresponding axes
    axes[0].axhline(y=era5_mean, **arguments)
    axes[1].axhline(y=era5_var, **arguments)
    axes[2].axhline(y=era5_skewness, **arguments)
    axes[3].axhline(y=era5_kurtosis, **arguments)
    axes[4].axhline(y=era5_max, **arguments)

    # Return the mean and variance of the ERA5 data for further use
    return era5_mean, era5_var


def plot_summary_stats(file_path, axes, files_information, model_information):
    """
    Computes and plots the summary statistics (Mean, Variance, Skewness, Kurtosis, Max Value) for a given model file.

    Parameters:
    - file_path: The path to the model file to analyze.
    - axes: List of axis objects for plotting the summary statistics.
    - files_information: Contains metadata for the files being compared.
    - model_information: Contains metadata specific to the model being analyzed.

    Returns:
    - mean: The mean wind speed for the model.
    - max_value: The maximum wind speed for the model (average of top 1% values).
    - spatial_resolution: The spatial resolution of the model.
    - variance: The variance of the wind speed for the model.
    - skewness: The skewness of the wind speed for the model.
    - kurtosis: The kurtosis of the wind speed for the model.
    """
    # Retrieve and flatten the model wind speed data
    wind_speed, spatial_resolution, _ = get_flattened_wind_speed_and_resolution(file_path, files_information)

    # Calculate summary statistics for the model's wind speed data
    mean = wind_speed.mean()
    variance = wind_speed.var()
    skewness = stats.skew(wind_speed)
    kurtosis = stats.kurtosis(wind_speed)

    # Calculate the 99.9999th percentile value for the model's wind speed
    quantile = 0.999999
    threshold = np.quantile(wind_speed, quantile)

    # Retrieve the wind speeds greater than or equal to the threshold (top 1%)
    top_1_percent_wind_speeds = wind_speed[wind_speed >= threshold]
    print(model_information.model_name, top_1_percent_wind_speeds.size, np.mean(top_1_percent_wind_speeds))

    # Calculate the average of the top 100 wind speed values as the max value
    max_value = np.mean(np.sort(wind_speed)[-100:])  # Use the top 100 wind speeds for max value
    print(model_information.model_label, stats.describe(wind_speed))

    # Prepare a dictionary to store the values to plot on each axis
    plot_data = {
        axes[0]: mean,
        axes[1]: variance,
        axes[2]: skewness,
        axes[3]: kurtosis,
        axes[1]: max_value  # This mistakenly uses axes[1] for the max value instead of a new axis
    }

    # Plot each statistic on the corresponding axis for the model
    for ax, value in plot_data.items():
        plot_scatter_point(ax, spatial_resolution, value, model_information)

    # Return the calculated statistics for further use
    return mean, max_value, spatial_resolution, variance, skewness, kurtosis
