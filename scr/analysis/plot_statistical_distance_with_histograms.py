import numpy as np
import scipy
from matplotlib.colors import LinearSegmentedColormap

from scr.analysis.Information import FileInformation
from scr.analysis.utils import (configure_scatter_plot, save_figures, plot_scatter_point,
                                get_flattened_wind_speed_and_resolution, SCATTER_PLOT_WIDTH,
                                SCATTER_PLOT_HEIGHT, plot_heatmap, HEAT_PLOT_WIDTH, HEAT_PLOT_HEIGHT,
                                set_legend_at_top,
                                SCATTER_PLOT_ANNOTATION_SIZE, reverse_dict_order, configure_histogram_plot,
                                HISTOGRAM_LABEL_SIZE, HISTOGRAM_WIDTH, HISTOGRAM_HEIGHT, plot_regression_line)
from scr.settings import WIND_SPEED_LABEL, ERA5_NAME

# Directory name for saving statistical distance plots and results
FOLDER_NAME = 'Statistical Distance'

# Histogram configuration for wind speed data (bins, density, and range)
HISTOGRAM_ARGUMENTS_WS = {'bins': 50, 'density': True, 'range': (0, 30)}

# Custom colormap for visualizations with a red-to-white gradient
colors = [(0, 'darkred'), (0.5, 'white'), (1, 'darkred')]
cmap = LinearSegmentedColormap.from_list('one_color', colors)

# Number of bins for the color map and the corresponding color interpolation
num_bins = 42
cmap_bins = np.linspace(0, 1, num_bins + 1)
colors_interpolated = cmap(cmap_bins)

# Color map for heatmaps with binned color values
COLOR_MAP = LinearSegmentedColormap.from_list('one_color_binned', colors_interpolated, N=num_bins)


def plot_statistical_distance(files_information, div='Jensen-Shannon distance', plot_histogram=False):
    """
    Plots the statistical distance (Jensen-Shannon or Wasserstein-1 distance) between ERA5 data and model data.

    Parameters:
    - files_information: Contains metadata and paths for the ERA5 and model files.
    - div: The type of statistical distance to calculate ('Jensen-Shannon distance' or 'Wasserstein-1 distance').
    - plot_histogram: A flag indicating whether to plot histograms of wind speeds for comparison.
    """
    # Folder path for saving the comparison plots, based on the distance type
    full_folder_name = f'{FOLDER_NAME}/{div}'

    # Initialize the scatter plot with the specified distance type
    fig, ax = configure_scatter_plot(f'{div}')

    # Retrieve ERA5 data for comparison (flattened wind speed values)
    era5_file_path = list(files_information.era5_files.values())[0]
    era5_wind_speed, _, _ = get_flattened_wind_speed_and_resolution(era5_file_path, files_information)
    era5_data = era5_wind_speed
    histogram_arguments = HISTOGRAM_ARGUMENTS_WS

    # Compute histogram for the ERA5 wind speed data
    era5_hist, era5_edges = np.histogram(era5_data, **histogram_arguments)

    # Dictionary to store computed distances for each model file
    distances = {}

    # List to store spatial resolutions of the model files for regression plotting
    spatial_resolutions = []

    # Loop through each model file and compute the statistical distance
    for file_path in files_information.files:
        # Retrieve model-specific information and wind speed data
        model_information = FileInformation(file_path, files_information.run_comparison)
        wind_speed, spatial_resolution, _ = get_flattened_wind_speed_and_resolution(file_path, files_information)
        spatial_resolutions.append(spatial_resolution)

        # Compute histogram for the model wind speed data
        distr_hist, distr_edges = np.histogram(wind_speed, **histogram_arguments)

        # Calculate the selected statistical distance (Jensen-Shannon or Wasserstein-1)
        if div == 'Jensen-Shannon distance':
            distance = scipy.spatial.distance.jensenshannon(era5_hist, distr_hist)
        elif div == 'Wasserstein-1 distance':
            # Alternative: Wasserstein distance (commented out)
            distance = scipy.stats.wasserstein_distance(era5_hist, distr_hist)
        else:
            # Error handling if an unknown distance type is provided
            raise ValueError('Unknown distance')

        # Store the distance for the model if it is not ERA5 itself
        if ERA5_NAME != model_information.model_name:
            distances[model_information.model_label] = distance
            # Plot the scatter point for the model on the distance vs. spatial resolution plot
            plot_scatter_point(ax, spatial_resolution, distance, model_information)

        # Optionally, plot histograms of the wind speed data for the model and ERA5
        if plot_histogram:
            make_histogram_fig(era5_hist, distr_edges, distr_hist, files_information, model_information)

    # Plot regression line for distance vs. spatial resolution, with appropriate labeling and annotations
    plot_name = f'{div}'
    x = np.array(spatial_resolutions)
    v = np.array(list(distances.values()))
    plot_regression_line(x, v, ax, plot_name, files_information, full_folder_name)

    # Set vmax for the heatmap (for Jensen-Shannon distance, the max value is adjusted)
    vmax = 0.01
    if div == 'Jensen-Shannon distance':
        vmax = 0.2
        # Set legend at the top of the plot for better visibility
        set_legend_at_top([ax], SCATTER_PLOT_ANNOTATION_SIZE)

    # Save the scatter plot as a figure
    save_figures({f'{plot_name} {files_information.title}': fig}, SCATTER_PLOT_WIDTH, SCATTER_PLOT_HEIGHT,
                 files_information, full_folder_name)

    # Create and save the heatmap for the distances with proper color mapping
    fig_heatmap = plot_heatmap(reverse_dict_order(distances), plot_name, 0, vmax, 0, COLOR_MAP, True)
    title = f'{plot_name} Heatmap {files_information.title}'
    save_figures({title: fig_heatmap}, HEAT_PLOT_WIDTH, HEAT_PLOT_HEIGHT, files_information, full_folder_name)


def make_histogram_fig(era5_hist, bin_edges, distr_hist, files_information, model_information):
    """
    Creates and saves a histogram plot comparing the wind speed distributions of ERA5 data and a model.

    Parameters:
    - era5_hist: The histogram values of ERA5 wind speed values.
    - bin_edges: The edges of the histogram bins.
    - distr_hist: The histogram values of the model's wind speed values.
    - files_information: Contains metadata about the files.
    - model_information: Contains model-specific information like label and color for the plot.
    """
    # Initialize the histogram plot
    fig, ax = configure_histogram_plot()

    # Plot the ERA5 histogram (in grey, with some transparency)
    ax.bar(bin_edges[:-1], era5_hist, width=np.diff(bin_edges),
           color='grey', label=ERA5_NAME, alpha=0.5)

    # Plot the model histogram (in the model's color, with some transparency)
    ax.bar(bin_edges[:-1], distr_hist, width=np.diff(bin_edges),
           label=model_information.model_label, alpha=0.5, color=model_information.color)

    # Add a legend for the histograms
    ax.legend(frameon=False, fontsize=HISTOGRAM_LABEL_SIZE - 2, loc='upper left', bbox_to_anchor=(0.2, 0.9))

    # Set the x-axis label (wind speed) and title for the plot
    ax.set_xlabel(WIND_SPEED_LABEL, fontsize=HISTOGRAM_LABEL_SIZE)
    title = f'Histogram ws {files_information.title} {model_information.model_label}'

    # Save the histogram figure
    save_figures({title: fig}, HISTOGRAM_WIDTH, HISTOGRAM_HEIGHT, files_information, FOLDER_NAME)
