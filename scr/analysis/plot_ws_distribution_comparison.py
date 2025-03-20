import os

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from sklearn.neighbors import KernelDensity

from scr.analysis.Information import FileInformation
from scr.analysis.plot_statistical_distance_with_histograms import HISTOGRAM_ARGUMENTS_WS
from scr.analysis.utils import save_figures, get_flattened_wind_speed_and_resolution, \
    configure_distribution_plot, DISTRIBUTION_PLOT_WIDTH, DISTRIBUTION_PLOT_HEIGHT, configure_all_kde_plot, \
    set_legend_at_top, KDE_PLOT_LABEL_SIZE, KDE_PLOT_WIDTH, KDE_PLOT_HEIGHT, compute_wind_power, \
    get_wind_power_in_mega_watt, get_npy_file_path
from scr.settings import WS_MAX

FOLDER_NAME = 'Distribution Comparisons'  # The directory where output plots will be saved
PLOT_NAMES = ['KDE', 'KDE Differences', 'KDE Ratio', 'CDF', '1-CDF']  # Plot names for distribution comparisons
PLOT_NAMES_PAPER = ['KDE', 'KDE Difference to ERA5', 'Generated WP in MW']  # Plot names for paper-specific comparisons


def plot_all_kde_together(files_information, generated_power=False):
    """
    Plots Kernel Density Estimates (KDEs) for all wind speed models together, optionally computing wind power.

    Arguments:
        files_information (FileInformation): Contains metadata and file paths for the comparison.
        generated_power (bool): Whether to plot generated wind power instead of raw KDE.
    """
    fig, ax = configure_all_kde_plot()  # Create the figure and axis for plotting KDE
    power_curve_information = files_information.turbine  # Get turbine power curve details
    ax.axvline(x=power_curve_information.cut_in, color='grey', linewidth=0.8)  # Add cut-in wind speed line
    ax.axvline(x=power_curve_information.cut_off, color='grey', linewidth=0.8)  # Add cut-off wind speed line
    # Loop through all wind speed files and plot KDEs
    for file_path in files_information.files:
        model_information = FileInformation(file_path, files_information.run_comparison)  # Get model information
        grid, kde = save_kde_values_in_file(file_path, files_information)  # Get KDE values
        if generated_power:
            kde = compute_wind_power(grid, power_curve_information) * kde  # Compute wind power if flag is set
        ax.plot(grid, kde, alpha=0.8, label=model_information.model_label, **model_information.arguments)  # Plot KDE
    # Plot ERA5 data for comparison
    era5_file_path = list(files_information.era5_files.values())[0]
    era5_model_information = FileInformation(era5_file_path, files_information.run_comparison)
    grid, era5_kde = save_kde_values_in_file(era5_file_path, files_information)
    ax.plot(grid, era5_kde, label=era5_model_information.model_label, **era5_model_information.arguments)
    if generated_power:
        ax.yaxis.set_ticklabels([])  # Remove y-axis tick labels for power plot
        ax.yaxis.set_ticks([])  # Remove y-axis ticks for power plot
    else:
        set_legend_at_top([ax], KDE_PLOT_LABEL_SIZE - 2)  # Adjust the legend position for non-power plot
    additional_text = 'Generated wind power' if generated_power else 'Kernel density estimate'  # Label
    ax.set_ylabel(additional_text, fontsize=KDE_PLOT_LABEL_SIZE - 2)  # Set y-axis label based on plot type
    save_figures({f'{additional_text} {files_information.title}': fig},
                 KDE_PLOT_WIDTH, KDE_PLOT_HEIGHT, files_information, FOLDER_NAME)  # Save the figure


def save_kde_values_in_file(file_path, files_information):
    """
    Saves the computed KDE values for a given file, or loads them from disk if already computed.

    Arguments:
        file_path (str): Path to the file containing wind speed data.
        files_information (FileInformation): Metadata for the files.

    Returns:
        tuple: A tuple containing the grid and KDE values.
    """
    height = files_information.hub_height  # Turbine hub height
    base_file_path = get_npy_file_path(file_path)  # Get base path for .npy files
    kde_file_path = f'{base_file_path}_{height}m_kde.npy'  # Path for KDE file
    grid_file_path = f'{base_file_path}_{height}m_grid.npy'  # Path for grid file
    if not os.path.isfile(kde_file_path) or not os.path.isfile(grid_file_path):
        grid, kde = get_kde(file_path, files_information)  # Compute KDE if not already saved
        with open(grid_file_path, 'wb') as f:
            np.save(f, grid)  # Save grid
        with open(kde_file_path, 'wb') as f:
            np.save(f, kde)  # Save KDE values
        return grid, kde
    else:
        kde_values = np.load(kde_file_path)  # Load precomputed KDE values
        grid = np.load(grid_file_path)  # Load precomputed grid
        return grid, kde_values  # Return loaded values


def plot_all_kde_together_with_highlighted_model(files_information, model_to_highlight):
    """
    Plots KDEs for all models, highlighting a specific model of interest.

    Arguments:
        files_information (FileInformation): Metadata and file paths for the comparison.
        model_to_highlight (str): The model label to highlight in the plot.
    """
    fig, ax = configure_all_kde_plot()  # Configure plot settings
    era5_file_path = list(files_information.era5_files.values())[0]  # Get ERA5 data file
    era5_model_information = FileInformation(era5_file_path, files_information.run_comparison)
    grid, era5_kde = save_kde_values_in_file(era5_file_path, files_information)  # Get ERA5 KDE
    ax.plot(grid, era5_kde, alpha=0.1, label=era5_model_information.model_label,
            **era5_model_information.arguments)  # Plot ERA5 KDE with low opacity
    # Loop through files and highlight the specified model
    for file_path in files_information.files:
        model_information = FileInformation(file_path, files_information.run_comparison)
        grid, kde = save_kde_values_in_file(file_path, files_information)
        if model_to_highlight in model_information.model_label:
            ax.plot(grid, kde, label=model_information.model_label, **model_information.arguments)  # Highlight model
        else:
            ax.plot(grid, kde, alpha=0.1, label=model_information.model_label,
                    **model_information.arguments)  # Plot other models with low opacity
    set_legend_at_top([ax], KDE_PLOT_LABEL_SIZE - 2)  # Adjust legend position
    save_figures({f'KDEs {files_information.title}': fig},
                 KDE_PLOT_WIDTH, KDE_PLOT_HEIGHT, files_information, FOLDER_NAME)  # Save the plot


def plot_distribution_differences_paper(files_information):
    """
    Plots distribution differences for paper comparison, including ERA5 and generated wind power.

    Arguments:
        files_information (FileInformation): Metadata for the comparison files.
    """
    era5_file_path = list(files_information.era5_files.values())[0]
    era5_grid, era5_kde = save_kde_values_in_file(era5_file_path, files_information)  # Get ERA5 KDE
    # Loop through all files and generate the distribution difference plots
    for file_path in files_information.files + [era5_file_path]:
        model_information = FileInformation(file_path, files_information.run_comparison)
        figures = {}  # Dictionary to store figure objects
        axes_dict = {}  # Dictionary to store axes objects
        for name in PLOT_NAMES_PAPER:
            figures, axes_dict = configure_distribution_plot(figures, axes_dict, model_information.model_label,
                                                             name,
                                                             files_information)  # Configure subplots for each plot type
        axes = list(axes_dict.values())  # List of axes to iterate over for plotting
        power_curve_information = files_information.turbine  # Get turbine power curve details
        for ax in axes:
            ax.set_xlim(power_curve_information.cut_in,
                        power_curve_information.cut_off)  # Set x-limits based on power curve
        axes[0].set_ylim(0, 0.2)  # Set y-limits for KDE plot
        axes[1].set_ylim(-0.1, 0.1)  # Set y-limits for KDE difference plot
        axes[1].axhline(y=0, color='black', linestyle='-')  # Add horizontal line at 0 for difference plot
        axes[2].set_ylim(0, 0.15)  # Set y-limits for wind power plot
        arguments = model_information.arguments

        grid, kde = save_kde_values_in_file(file_path, files_information)  # Get KDE for the model
        axes[0].plot(grid, kde, **arguments)  # Plot KDE
        axes[1].plot(grid, kde - era5_kde, **arguments)  # Plot difference with ERA5 KDE
        generated_power = compute_wind_power(grid, power_curve_information) * kde  # Compute generated wind power
        axes[2].plot(grid, get_wind_power_in_mega_watt(generated_power), **arguments)  # Plot generated power in MW
        save_figures(figures, DISTRIBUTION_PLOT_WIDTH, DISTRIBUTION_PLOT_HEIGHT,
                     files_information, FOLDER_NAME)  # Save the figures


def plot_distribution_comparison_single(files_information, power_curve_information):
    """
    Plots a single model's distribution compared to ERA5, including KDE, differences, ratios, and CDFs.

    Arguments:
        files_information (FileInformation): Metadata for the comparison files.
        power_curve_information (TurbinePowerCurve): Information about the turbine's power curve.
    """
    era5_file_path = list(files_information.era5_files.values())[0]
    era5_grid, era5_kde = save_kde_values_in_file(era5_file_path, files_information)  # Get ERA5 KDE
    for file_path in files_information.files:
        model_information = FileInformation(file_path, files_information.run_comparison)
        figures = {}  # Store figure objects
        axes_dict = {}  # Store axes objects
        for name in PLOT_NAMES:
            figures, axes_dict = configure_distribution_plot(figures, axes_dict, model_information.model_label,
                                                             name, files_information)  # Configure subplots
        axes = list(axes_dict.values())  # Get the list of axes
        for ax in axes:
            ax.set_xlim(power_curve_information.cut_in,
                        power_curve_information.cut_off)  # Set x-limits based on turbine
        axes[0].set_ylim(0, 0.2)  # KDE plot y-limits
        axes[1].set_ylim(-0.1, 0.1)  # Difference plot y-limits
        axes[1].axhline(y=0, color='black', linestyle='-')  # Horizontal line at y=0
        axes[2].set_ylim(0, 0.15)  # Wind power plot y-limits
        grid, kde = save_kde_values_in_file(file_path, files_information)  # Get KDE for the model
        axes[0].plot(grid, kde, **model_information.arguments)  # Plot KDE
        axes[1].plot(grid, kde - era5_kde, **model_information.arguments)  # Plot difference with ERA5
        generated_power = compute_wind_power(grid, power_curve_information) * kde  # Compute generated wind power
        axes[2].plot(grid, get_wind_power_in_mega_watt(generated_power),
                     **model_information.arguments)  # Plot generated power
        save_figures(figures, DISTRIBUTION_PLOT_WIDTH, DISTRIBUTION_PLOT_HEIGHT,
                     files_information, FOLDER_NAME)  # Save the figures

def plot_distribution_comparison_group(files_information, model_group_name, power_curve_information):
    """
    Generates and saves a series of distribution comparison plots for a specified model group. Each plot includes
    comparisons to ERA5 baseline data, and results are saved to a specified directory.

    Arguments:
        files_information (FileInformation): Contains paths and metadata for all files to be analyzed.
        model_group_name (str): The name of the model group being analyzed, used in file paths and labels.
        power_curve_information (PowerCurveInformation): Contains information on turbine power curves for plotting.
    """
    # Define the full directory path for saving figures, based on the model group name
    full_folder_name = f'{FOLDER_NAME}/{model_group_name}'
    figures = {}  # Dictionary to hold each figure created
    axes_dict = {}  # Dictionary to store the corresponding axes for each figure

    # Generate individual plots for each specified plot type in PLOT_NAMES
    for i in range(len(PLOT_NAMES)):
        # Determine if x-axis label is needed, based on the presence of 'Global2' in the model group name
        plot_x_label = True if 'Global2' in model_group_name else False
        # Configure the plot, updating figures and axes dictionaries with the new plot details
        figures, axes_dict = configure_distribution_plot(
            figures, axes_dict, model_group_name, PLOT_NAMES[i], files_information, plot_x_label
        )
    axes = list(axes_dict.values())  # Convert axes dictionary to a list for easy access during plotting

    # Retrieve ERA5 data to serve as a baseline for comparison
    era5_file_path = list(files_information.era5_files.values())[0]  # Path to the first ERA5 file
    era5_model_information = FileInformation(era5_file_path, files_information.run_comparison)  # Metadata for ERA5 file
    era5_grid, era5_kde, era5_sorted_data, era5_cdf = get_data(era5_file_path, files_information)  # Load ERA5 data
    era5_arguments = era5_model_information.arguments  # Plotting arguments for ERA5 data

    # Plot ERA5 lines across all designated axes for direct visual comparison with other models
    plot_era5_lines(axes, era5_grid, era5_kde, era5_cdf, era5_sorted_data, era5_arguments, power_curve_information)

    # Set up an additional figure to contain a legend, initializing without any axis display
    fig_label, ax_label = plt.subplots(figsize=(10, 1))
    ax_label.axis('off')  # Hide axes on the legend figure
    axes_dict['legend'] = ax_label  # Store legend axis in the axes dictionary

    # Loop through each file in files_information for comparison plotting
    for file_path in files_information.files:
        # Extract metadata and arguments for the current model file
        model_information = FileInformation(file_path, files_information.run_comparison)
        arguments = model_information.arguments  # Plotting style arguments

        # Generate grid, KDE, sorted data, and CDF for the current model file
        grid, kde, sorted_data, cdf = get_data(file_path, files_information)
        kde = kde(grid)  # Compute KDE values over the grid
        true_kde = era5_kde(grid)  # ERA5 KDE values for baseline comparison

        # Plot each comparison metric on the respective axes
        axes[0].plot(grid, kde, **arguments)  # KDE curve for current model
        axes[1].plot(grid, np.nan_to_num(kde - true_kde), **arguments)  # KDE difference from ERA5 baseline
        axes[2].plot(grid, kde / true_kde, **arguments)  # KDE ratio to ERA5 baseline
        axes[3].plot(sorted_data, cdf, **arguments)  # Cumulative distribution function (CDF)
        axes[4].plot(sorted_data, 1 - cdf, **arguments)  # Survival function (1 - CDF)

        # Add model label to the legend plot
        ax_label.plot([], [], label=model_information.model_label, **arguments)
    ax_label.legend(frameon=False, fontsize=12, loc='center', bbox_to_anchor=(0.5, 0.5), ncol=5)  # Format legend

    # Save legend and each generated plot figure to the specified directory
    save_figures({f'legend_{model_group_name}': fig_label}, 10, 1.5, files_information, full_folder_name)
    save_figures(figures, DISTRIBUTION_PLOT_WIDTH, DISTRIBUTION_PLOT_HEIGHT, files_information, full_folder_name)

def get_data(file_path, files_information):
    """
    Computes the KDE, grid, sorted wind speed data, and CDF from wind speed data in the given file.

    Arguments:
        file_path (str): Path to the file containing wind speed data.
        files_information (FileInformation): Metadata for file processing.

    Returns:
        grid values, kernel density estimate (KDE), sorted data, and cumulative distribution function (CDF).
    """
    # Load wind speed data, along with any additional information needed for processing
    wind_speed, _, _ = get_flattened_wind_speed_and_resolution(file_path, files_information)
    sorted_data = np.sort(wind_speed)  # Sort wind speeds for CDF calculation
    cdf = np.linspace(0, 1, len(sorted_data))  # Generate linear CDF values

    # Determine the maximum wind speed for defining the KDE evaluation range
    max_wind_speed = np.amax(wind_speed)
    grid = np.linspace(0, max_wind_speed, 500)  # Define grid for KDE evaluation
    kde = stats.gaussian_kde(wind_speed)  # KDE estimation on wind speed data

    return grid, kde, sorted_data, cdf


def get_kde(file_path, files_information):
    """
    Computes the Kernel Density Estimate (KDE) for wind speed data in the specified file.

    Arguments:
        file_path (str): Path to the file containing wind speed data.
        files_information (FileInformation): Metadata for the files, including settings for histogram generation.

    Returns:
        tuple: A tuple containing the evaluation grid and the KDE density values.
    """
    # Load wind speed data from the specified file
    wind_speed, _, _ = get_flattened_wind_speed_and_resolution(file_path, files_information)

    # Define the maximum wind speed to set the grid range for KDE evaluation
    max_wind_speed = np.amax(wind_speed)
    grid = np.linspace(0, max_wind_speed, 500)  # Create grid for KDE

    # Calculate histogram of wind speed data for initial KDE computation
    hist, bin_edges = np.histogram(wind_speed, **HISTOGRAM_ARGUMENTS_WS)

    # Determine bin centers from histogram bin edges to serve as data points for KDE
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Prepare bin centers for scikit-learn's KernelDensity, reshaping as required
    bin_centers = bin_centers[:, np.newaxis]
    kde = KernelDensity(bandwidth=1.0, kernel='gaussian')  # Initialize KDE with Gaussian kernel
    kde.fit(bin_centers, sample_weight=hist)  # Fit KDE to bin centers using histogram weights

    # Re-evaluate KDE over the defined grid, obtaining log-density
    grid = np.linspace(0, max_wind_speed, 500)[:, np.newaxis]
    log_dens = kde.score_samples(grid)  # Log-density values
    density = np.exp(log_dens)  # Convert log-density to actual density

    return grid.flatten(), density  # Flatten grid for output consistency with other functions


def get_cdf(file_path, files_information):
    """
    Computes the sorted wind speed data and the cumulative distribution function (CDF) values.

    Arguments:
        file_path (str): Path to the file containing wind speed data.
        files_information (FileInformation): Metadata for file processing.

    Returns:
        sorted wind speed data and corresponding CDF values.
    """
    # Load wind speed data and sort it in ascending order for CDF calculation
    wind_speed, _, _ = get_flattened_wind_speed_and_resolution(file_path, files_information)
    sorted_data = np.sort(wind_speed)

    # Generate linear CDF values that correspond to sorted wind speed data
    cdf = np.linspace(0, 1, len(sorted_data))
    return sorted_data, cdf  # Return sorted data and CDF values


def plot_era5_lines(axes, grid, kde, cdf, sorted_data, arguments, power_curve_information):
    """
    Plots ERA5 data lines (KDE, KDE difference, KDE ratio, CDF, and survival function) on specified axes for comparison.

    Arguments:
        axes (list of matplotlib.axes.Axes): List of axes for each distribution plot type.
        grid (numpy.ndarray): Evaluation grid for KDE.
        kde (function): Kernel Density Estimate function for ERA5 data.
        cdf (numpy.ndarray): Cumulative distribution function values for ERA5 data.
        sorted_data (numpy.ndarray): Sorted wind speed data for ERA5.
        arguments (dict): Plotting style arguments.
        power_curve_information (PowerCurveInformation): Information about power curve for axis limits.
    """
    # Plot KDE on the first axis, applying KDE settings and plotting over the grid
    axes[0] = axes_settings_kde(axes[0], power_curve_information)
    axes[0].plot(grid, kde(grid), **arguments)  # KDE curve

    # Prepare axes for KDE difference and ratio plots with ERA5 baseline
    axes[1] = axes_settings_kde_differences(axes[1], power_curve_information)
    axes[2] = axes_settings_kde_ratio(axes[2], power_curve_information)

    # Plot CDF on the fourth axis, setting x-axis limit and plotting sorted data and CDF
    axes[3].set_xlim(0, power_curve_information.cut_off)
    axes[3].plot(sorted_data, cdf, **arguments)

    # Plot survival function (1 - CDF) on the fifth axis with survival function settings
    axes[4] = axes_settings_survival_function(axes[4], power_curve_information)
    axes[4].plot(sorted_data, 1 - cdf, **arguments)


def axes_settings_kde(ax, power_curve_information):
    """
    Configures the axis for KDE plotting with specified limits.

    Arguments:
        ax (matplotlib.axes.Axes): Axis to configure for KDE plot.
        power_curve_information (PowerCurveInformation): Contains limit information from the power curve.

    Returns:
        matplotlib.axes.Axes: Configured axis.
    """
    # Set x-axis limit based on power curve cutoff and y-axis to range [0, 0.35] for KDE visualization
    ax.set_xlim(0, power_curve_information.cut_off)
    ax.set_ylim(0, 0.35)
    return ax  # Return configured axis


def axes_settings_kde_differences(ax, power_curve_information):
    """
    Configures the axis for KDE difference plotting with a fixed y-axis range.

    Arguments:
        ax (matplotlib.axes.Axes): Axis to configure for KDE difference plot.
        power_curve_information (PowerCurveInformation): Contains limit information from the power curve.

    Returns:
        matplotlib.axes.Axes: Configured axis.
    """
    # Set x-axis from cut-in to cut-off speed and y-axis from -0.1 to 0.1 for KDE difference plot
    ax.set_xlim(power_curve_information.cut_in, power_curve_information.cut_off)
    ax.set_ylim(-0.1, 0.1)
    ax.axhline(y=0, color='black', linestyle='-')  # Add horizontal line at y=0 for visual reference
    return ax  # Return configured axis


def axes_settings_kde_ratio(ax, power_curve_information):
    """
    Configures the axis for KDE ratio plotting with logarithmic y-axis scale.

    Arguments:
        ax (matplotlib.axes.Axes): Axis to configure for KDE ratio plot.
        power_curve_information (PowerCurveInformation): Contains limit information from the power curve.

    Returns:
        matplotlib.axes.Axes: Configured axis.
    """
    # Set x-axis from cut-off speed to maximum wind speed and y-axis to log scale from 1e-3 to 1e3 for ratio plot
    ax.set_xlim(power_curve_information.cut_off, WS_MAX)
    ax.set_ylim(1e-3, 1e3)
    ax.set_yscale('log')  # Set y-axis to logarithmic scale for wide range visibility
    ax.axhline(y=1, color='black', linestyle='-')  # Add line at y=1 to denote no difference
    return ax  # Return configured axis


def axes_settings_survival_function(ax, power_curve_information):
    """
    Configures the axis for survival function plotting with logarithmic y-axis scale.

    Arguments:
        ax (matplotlib.axes.Axes): Axis to configure for survival function plot.
        power_curve_information (PowerCurveInformation): Contains limit information from the power curve.

    Returns:
        matplotlib.axes.Axes: Configured axis.
    """
    # Set x-axis from cut-off to maximum wind speed, and y-axis range for survival function plot
    ax.set_xlim(power_curve_information.cut_off, WS_MAX)
    ax.set_ylim(1e-6, 1e-3)
    ax.set_yscale('log')  # Logarithmic scale to enhance visibility of small values in survival function
    return ax  # Return configured axis

