import os

import geopandas
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import linregress
from windpowerlib import ModelChain

from scr.settings import END_YEAR, SURFACE_WIND_SPEED, \
    WIND_SPEED_LABEL, LATITUDE_NAME, LONGITUDE_NAME

def get_wind_speed(file_path):
    """Load wind speed data from a file using xarray."""
    # Open the file using xarray's open_dataset function, which handles netCDF data.
    return xr.open_dataset(file_path, engine='netcdf4')

def wind_profile_power_law(v_0, hub_height, h_0=10, a=0.143):
    """Calculate wind speed at a given hub height using the power law."""
    # Apply the power law formula to calculate wind speed at a specific height (hub_height).
    # v_0: reference wind speed at height h_0 (default 10m), hub_height: target height, a: exponent (default 0.143)
    return v_0 * (hub_height / h_0) ** a

def get_time_average_wind_speed(file_path, files_information):
    """Calculate time-averaged wind speed for a given file."""
    # Get wind speed data from the file and apply the power law correction
    wind_speed = get_wind_speed(file_path)
    wind_speed = wind_profile_power_law_with_cmip5_correction(
        wind_speed[SURFACE_WIND_SPEED], file_path, files_information)
    # Return the time-averaged wind speed over the entire dataset
    return wind_speed.mean(dim='time')

def get_wind_speeed_chunk(wind_speed, file_path, files_information):
    """Load and flatten a chunk of wind speed data."""
    # Load wind speed data, apply correction, and flatten the array
    wind_speed = wind_speed.load().values
    wind_speed = wind_profile_power_law_with_cmip5_correction(wind_speed, file_path, files_information).flatten()
    # Return the data as a float16 type, ensuring NaN values are excluded
    return np.float16(wind_speed[~np.isnan(wind_speed)])

def wind_profile_power_law_with_cmip5_correction(wind_speed, file_path, files_information):
    """Apply power law correction based on CMIP5 model or other conditions."""
    # If the file path indicates the use of CMIP5 data, apply a specific power law correction
    if 'CMIP5' in file_path:
        return wind_profile_power_law(wind_speed, files_information.hub_height, 1500)
    else:
        # Use the default correction if CMIP5 is not mentioned
        return wind_profile_power_law(wind_speed, files_information.hub_height)

def get_flattened_wind_speed_and_resolution(file_path, files_information):
    """Extract flattened wind speed values and spatial resolution."""
    # Load wind speed data and calculate the temporal resolution
    wind_speed = get_wind_speed(file_path)
    temporal_resolution = wind_speed.time.size
    chunk_size = 1000  # Define chunk size for processing large datasets
    wind_speed_values = np.float16(np.array([]))  # Initialize an empty array to store wind speed values
    # Process the data in chunks to reduce memory usage
    for start in range(0, temporal_resolution, chunk_size):
        end = min(start + chunk_size, temporal_resolution)
        chunk = get_wind_speeed_chunk(wind_speed[SURFACE_WIND_SPEED][start:end], file_path, files_information)
        wind_speed_values = np.float16(np.concatenate((wind_speed_values, chunk)))
    # Calculate spatial resolution based on the size of the flattened data
    spatial_resolution = int(wind_speed_values.size / temporal_resolution)
    return wind_speed_values, spatial_resolution, temporal_resolution

def get_wind_power_per_time_step(file_path, model_category, region, power_curve_information):
    """Calculate wind power per time step, adjusting for turbine count and spatial resolution if needed."""
    # Get wind power data and adjust for turbine count if necessary
    time, wind_power = get_wind_power(file_path, model_category, region, power_curve_information)
    wind_power = np.sum(wind_power, axis=1)  # Sum over time dimension to get total power
    if not model_category == 'Regional' and 'Turbine' not in region:
        # If not a regional model and no turbines are specified, adjust wind power based on spatial resolution
        _, spatial_resolution, _ = get_flattened_wind_speed_and_resolution(file_path)
        wind_power = wind_power / spatial_resolution
        # Read turbine location data and adjust power based on the number of turbines
        turbine_data = geopandas.read_file("dat/turbine_locations/windpower.shp")
        turbine_data_filtered = turbine_data[turbine_data['Year'] <= END_YEAR]
        number_turbine = len(turbine_data_filtered)
        wind_power = wind_power * number_turbine  # Adjust wind power by the number of turbines
    return time, wind_power

def get_wind_power(file_path, model_category, region, power_curve_information, files_information):
    """Retrieve wind speed data, apply power curve, and calculate wind power output."""
    # Load wind speed data and apply power law correction
    wind_speed = get_wind_speed(file_path)
    wind_speed = wind_profile_power_law_with_cmip5_correction(wind_speed[SURFACE_WIND_SPEED], file_path, files_information)
    if not model_category == 'Regional' and 'Turbine' not in region:
        # Stack the wind speed data along spatial coordinates (latitude and longitude)
        wind_speed = wind_speed.stack({'coordinates': (LATITUDE_NAME, LONGITUDE_NAME)})
    time = wind_speed.time  # Time dimension of the wind speed data
    wind_power = compute_wind_power(wind_speed, power_curve_information)  # Calculate wind power from wind speed
    return time, wind_power

def compute_wind_power_dataframe(wind_speed, power_curve_information):
    """Convert wind speed to power and return as a DataArray."""
    # Compute wind power and return the results as an xarray DataArray
    wp = compute_wind_power(wind_speed, power_curve_information)
    return xr.DataArray(wp)

def compute_wind_power(wind_speed, power_curve_information):
    """Calculate wind power output from wind speed data using a power curve."""
    # Use the ModelChain class to apply the power curve and calculate power output
    mc_turbine = ModelChain(power_curve_information.turbine)
    power = mc_turbine.calculate_power_output(wind_speed, 1)  # Calculate power for each time step
    return power

def get_wind_power_in_tera_watt(wind_power):
    """Convert wind power to terawatts."""
    # Convert the wind power from watts to terawatts (1e12 watts)
    return wind_power / 1e12

def get_wind_power_in_giga_watt(wind_power):
    """Convert wind power to gigawatts."""
    # Convert the wind power from watts to gigawatts (1e9 watts)
    return wind_power / 1e9

def get_wind_power_in_mega_watt(wind_power):
    """Convert wind power to megawatts."""
    # Convert the wind power from watts to megawatts (1e6 watts)
    return wind_power / 1e6

def calculate_bias(val, era5_val):
    """Calculate percentage bias between two values."""
    # Calculate the percentage bias between observed value (val) and reference (era5_val)
    return val / era5_val * 100 - 100

def reverse_dict_order(dictionary):
    """Reverse the order of dictionary items."""
    # Reverse the order of dictionary items (i.e., reverse the list of keys/values)
    return dict(list(dictionary.items())[::-1])

def get_npy_file_path(file_path):
    """Adjust file path for .nc or .txt extensions."""
    # Return the file path without the extension, depending on whether it's a .nc or .txt file
    if file_path.endswith('.nc'):
        return file_path.replace('.nc', '')
    elif file_path.endswith('.txt'):
        return file_path.replace('.txt', '')
    else:
        raise ValueError("Unsupported file extension. File must end with either '.nc' or '.txt'.")

def plot_regression_line(x, v, ax, name, files_information, extra_folder):
    """Plot a regression line with calculated statistics."""
    # Perform a log-log regression on the data and calculate the line's parameters
    log_x = np.log10(x)
    slope, intercept, r, p, se = linregress(log_x, np.array(v))
    folder = get_folder(files_information, extra_folder)  # Get the folder to save the output
    # Save the regression parameters to a text file
    with open(f"{folder}/{name}.txt", "w") as file:
        print(f"{name}: Intercept={intercept:.4f} Slope={slope:.4f} R-squared={r ** 2 * 100:.4f} P-value={p:.4f} "
              f"Standard Deviation={se:.4f}", file=file)
    # Generate the regression line for plotting
    x_fit = np.logspace(log_x.min(), log_x.max(), 100)
    log_x_fit = np.log10(x_fit)
    y_fit = linear(log_x_fit, slope, intercept)
    ax.plot(x_fit, y_fit, color='grey', linewidth=1, label='Regression Line', zorder=0)
    std_errors = np.full_like(x_fit, se)
    # Add shaded area representing standard error
    ax.fill_between(x_fit, y_fit - std_errors, y_fit + std_errors,
                    color='grey', alpha=0.3, label="Standard Errors")

def linear(x, a, b):
    """Calculate linear function y = ax + b."""
    # Simple linear function for use in regression fitting
    return a * x + b

# Constants for scatter plot configuration
SCATTER_PLOT_WIDTH = 11 # Width (in inches) for scatter plots
SCATTER_PLOT_HEIGHT = 7 # Height (in inches) for scatter plots
SCATTER_PLOT_LABEL_SIZE = 16 # Label size of axes (in inches) for scatter plots
SCATTER_PLOT_ANNOTATION_SIZE = 12 # Label size of annotations (in inches) for scatter plots
SCATTER_PLOT_X_LABEL = 'Number of spatial data points'
def configure_scatter_plot(y_label):
    """Configure a scatter plot with custom dimensions and labels."""
    # Configure and create a scatter plot with given width/height and custom x/y labels
    fig, ax = plt.subplots(figsize=(SCATTER_PLOT_WIDTH, SCATTER_PLOT_HEIGHT))
    ax.set_ylabel(y_label, fontsize=SCATTER_PLOT_LABEL_SIZE)
    ax.set_xlabel(SCATTER_PLOT_X_LABEL, fontsize=SCATTER_PLOT_LABEL_SIZE)
    ax.set_xscale('log')  # Logarithmic scale for x-axis
    adapt_axes(ax, SCATTER_PLOT_LABEL_SIZE, 4)  # Adapt the axes for clarity
    return fig, ax

def plot_scatter_point(ax, x, y, model_information):
    """Add a scatter point to the given axis with model-specific styling."""
    # Add a scatter point to the plot with custom style and label
    ax.scatter(x, y, marker='+', label=model_information.model_label, s=80, zorder=2, **model_information.arguments)

# Constants for configuring various plot types
TIME_SERIES_PLOT_LABEL_SIZE = 16  # Font size for labels in time series plots
TIME_SERIES_PLOT_WIDTH = 14  # Width (in inches) for time series plot
TIME_SERIES_PLOT_HEIGHT = 12  # Height (in inches) for time series plot

def configure_time_series_plot(figures, axes, x_label, y_label, var_name):
    """Configures and returns a time series plot with customized labels, axis formatting, and figure size."""
    fig, ax = plt.subplots(figsize=(TIME_SERIES_PLOT_WIDTH, TIME_SERIES_PLOT_HEIGHT))  # Create a new figure and axis
    ax.set_xlabel(x_label, fontsize=TIME_SERIES_PLOT_LABEL_SIZE)  # Set the x-axis label with a specified font size
    ax.set_ylabel(y_label, fontsize=TIME_SERIES_PLOT_LABEL_SIZE)  # Set the y-axis label with a specified font size
    ax.set_ylim(0.5, 2.5)  # Set fixed y-axis limits
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))  # Set major x-axis ticks to occur every 2 years
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format x-axis ticks as years
    adapt_axes(ax, TIME_SERIES_PLOT_LABEL_SIZE)  # Adjust axis ticks and label size
    return add_fig_and_axes_to_list(var_name, figures, fig, axes, ax)  # Add figure and axis to the tracking list

# Constants for configuring distribution plots
DISTRIBUTION_PLOT_WIDTH = 12  # Width (in inches) for distribution plot
DISTRIBUTION_PLOT_HEIGHT = 16  # Height (in inches) for distribution plot
DISTRIBUTION_PLOT_LABEL_SIZE = 36  # Font size for labels in distribution plots

def configure_distribution_plot(figures, axes, model_group, title, files_information, x_label=True):
    """Configures and returns a distribution plot with the option to display an x-axis label."""
    fig, ax = plt.subplots(figsize=(DISTRIBUTION_PLOT_WIDTH, DISTRIBUTION_PLOT_HEIGHT))  # Create figure and axis
    if x_label:  # Check if x-axis label is needed
        ax.set_xlabel(WIND_SPEED_LABEL, fontsize=DISTRIBUTION_PLOT_LABEL_SIZE)  # Set the x-axis label
    ax.set_ylabel(title, fontsize=DISTRIBUTION_PLOT_LABEL_SIZE)  # Set the y-axis label (plot title)
    adapt_axes(ax, DISTRIBUTION_PLOT_LABEL_SIZE, 4)  # Adjust axis properties for appropriate label size
    var_name = f'{title} {files_information.title} {model_group}'  # Generate a variable name for the plot
    return add_fig_and_axes_to_list(var_name, figures, fig, axes, ax)  # Return the updated figures and axes

# Constants for configuring histogram plots
HISTOGRAM_WIDTH = 10  # Width (in inches) for histogram plot
HISTOGRAM_HEIGHT = 8  # Height (in inches) for histogram plot
HISTOGRAM_LABEL_SIZE = 28  # Font size for labels in histogram plots

def configure_histogram_plot():
    """Configures and returns a histogram plot with default settings."""
    fig, ax = plt.subplots(figsize=(HISTOGRAM_WIDTH, HISTOGRAM_HEIGHT))  # Create a new histogram figure and axis
    adapt_axes(ax, HISTOGRAM_LABEL_SIZE)  # Adjust the axis for appropriate label sizes
    return fig, ax  # Return the figure and axis

# Constants for configuring KDE (Kernel Density Estimate) plots
KDE_PLOT_LABEL_SIZE = 16  # Font size for labels in KDE plots
KDE_PLOT_WIDTH = 14  # Width (in inches) for KDE plot
KDE_PLOT_HEIGHT = 8  # Height (in inches) for KDE plot

def configure_all_kde_plot():
    """Configures and returns a KDE (Kernel Density Estimate) plot with customized x-axis label and axis formatting."""
    fig, ax = plt.subplots(figsize=(KDE_PLOT_WIDTH, TIME_SERIES_PLOT_HEIGHT))  # Create the KDE plot figure and axis
    ax.set_xlabel(WIND_SPEED_LABEL, fontsize=KDE_PLOT_LABEL_SIZE)  # Set the x-axis label (wind speed)
    adapt_axes(ax, KDE_PLOT_LABEL_SIZE)  # Adjust the axis for appropriate label size
    return fig, ax  # Return the figure and axis

# Constants for configuring quantile plots
QUANTILE_WIDTH = QUANTILE_HEIGHT = 9  # Width and height (in inches) for quantile plot
QUANTILE_LABEL_FONT_SIZE = 32  # Font size for labels in quantile plots
QUANTILE_AXIS_LABEL = 'Quantiles'  # Label for the x and y axes in quantile plots

def create_quantile_plot(files_information, model_information, sample_quantiles, theoretical_quantiles,
                         theoretical_model_name, folder_name):
    """Creates and saves a quantile plot comparing sample quantiles to theoretical quantiles."""
    fig, ax = plt.subplots(figsize=(QUANTILE_WIDTH, QUANTILE_HEIGHT))  # Create the quantile plot figure and axis
    ax.plot([np.min(sample_quantiles), np.max(sample_quantiles)],
            [np.min(sample_quantiles), np.max(sample_quantiles)],
            color='grey', linestyle='--')  # Add a reference line for perfect quantile agreement
    ax.scatter(theoretical_quantiles, sample_quantiles, s=20, color=model_information.color)  # Scatter plot of quantiles
    ax.set_xlabel(f'{theoretical_model_name} {QUANTILE_AXIS_LABEL}', fontsize=QUANTILE_LABEL_FONT_SIZE)  # Set x-label
    ax.set_ylabel(f'{model_information.model_label} {QUANTILE_AXIS_LABEL}', fontsize=QUANTILE_LABEL_FONT_SIZE)  # Set y-label
    adapt_axes(ax, QUANTILE_LABEL_FONT_SIZE)  # Adjust axis formatting
    fig.tight_layout()  # Adjust layout to prevent clipping of labels
    title = f'{folder_name} {files_information.title} {model_information.model_label}'  # Generate the title for the plot
    save_figures({title: fig}, QUANTILE_WIDTH, QUANTILE_HEIGHT, files_information, folder_name)  # Save the figure to a file

# Constants for heatmap plotting
HEAT_PLOT_WIDTH = 12  # Width (in inches) for heatmap plots
HEAT_PLOT_HEIGHT = 28  # Height (in inches) for heatmap plots
HEAT_PLOT_LABEL_SIZE = 20  # Font size for labels in heatmap plots

def save_heat_plot_relative_biases(values, heat_plot_name, files_information, folder):
    """Saves a heatmap plot showing relative biases for the provided values."""
    fig = plot_heatmap(values, f'{heat_plot_name} in %')  # Create heatmap with specified title
    save_figures({files_information.title: fig}, HEAT_PLOT_WIDTH, HEAT_PLOT_HEIGHT, files_information, folder)  # Save the figure

# Constants for defining the color map for heatmap plots
colors = [(0, 'midnightblue'), (0.2, 'blue'), (0.3, 'deepskyblue'), (0.5, 'white'),
              (0.6, 'orange'), (0.75, 'red'), (0.9, 'brown'), (1, 'darkred')]  # Color gradient for the heatmap
hotcold_cmap = LinearSegmentedColormap.from_list('hotcold', colors)  # Generate a custom color map from the color list
num_bins = 60  # Number of bins for color interpolation
cmap_bins = np.linspace(0, 1, num_bins + 1)  # Define bin boundaries for color interpolation
colors_interpolated = hotcold_cmap(cmap_bins)  # Apply the color map to the bins
hotcold_binned_cmap = LinearSegmentedColormap.from_list('hotcold_binned',
                                                        colors_interpolated, N=num_bins)  # Create the final binned color map
VMIN = -60  # Minimum value for color scale in heatmaps
VMAX = 60  # Maximum value for color scale in heatmaps

def plot_heatmap(heatmap_dict, title, vmin=VMIN, vmax=VMAX, center=0, cmap=hotcold_binned_cmap, reverse=False):
    """Generates and returns a heatmap based on the provided values and options."""
    labels, values = get_sorted_sorted_values_dict(list(heatmap_dict.keys()), list(heatmap_dict.values()), True)  # Sort values
    fig, ax = plt.subplots(figsize=(HEAT_PLOT_WIDTH, HEAT_PLOT_HEIGHT))  # Create the figure and axis for the heatmap
    if vmin is None:
        vmin = min(values) - 0.1  # Set lower bound for color scale if not provided
    if vmax is None:
        vmax = max(values) + 0.1  # Set upper bound for color scale if not provided
    sns.heatmap(+values.reshape(1, -1), cmap=cmap, annot=False, fmt=".2f", xticklabels=labels,
                yticklabels=False, ax=ax, square=True, center=center, vmin=vmin, vmax=vmax, cbar=True,
                cbar_kws={"orientation": "horizontal", "aspect": 30, "shrink": 0.7})  # Create the heatmap using seaborn
    ax.set_xticklabels(labels, rotation=270, fontsize=HEAT_PLOT_LABEL_SIZE - 2)  # Set x-tick labels with rotation
    cbar = ax.collections[0].colorbar  # Get the colorbar from the heatmap
    cbar.ax.tick_params(labelsize=HEAT_PLOT_LABEL_SIZE - 4)  # Adjust colorbar tick label size
    cbar.set_label(title, fontsize=HEAT_PLOT_LABEL_SIZE)  # Set the colorbar label
    return fig  # Return the generated figure

def get_sorted_sorted_values_dict(model_list, values_list, reverse=False):
    """Sorts the values and corresponding model list, with an option to reverse the order."""
    sorted_values = sorted(values_list, key=abs, reverse=reverse)  # Sort values by their absolute values
    sorted_keys = sorted(range(len(values_list)), key=lambda k: abs(values_list[k]), reverse=reverse)  # Get sorted indices
    reordered_model_list = [model_list[i] for i in sorted_keys]  # Reorder models based on sorted values
    return reordered_model_list, np.array(sorted_values)  # Return the sorted model list and values

def adapt_axes(axis, label_size, reduce_by=2):
    """Applies formatting to axis ticks and labels to ensure proper display."""
    axis.spines[['right', 'top']].set_visible(False)  # Hide the top and right spines of the plot
    axis.tick_params(axis='both', which='major', labelsize=label_size - reduce_by)  # Adjust major tick label size
    axis.tick_params(axis='both', which='minor', labelsize=label_size - reduce_by - 2)  # Adjust minor tick label size

def add_fig_and_axes_to_list(var_name, figures, fig, axes, ax):
    """Adds a figure and axis to the figures and axes dictionary with a specific variable name."""
    figures[var_name] = fig  # Add the figure to the figures dictionary
    axes[f'axs_{var_name}'] = ax  # Add the axis to the axes dictionary
    return figures, axes  # Return the updated figures and axes dictionaries

def set_legend_at_top(plt_objects, fontsize=12, ncol=3):
    """Positions the legend at the top of the plot for the given matplotlib objects."""
    for plt_object in plt_objects:
        plt_object.legend(frameon=False, fontsize=fontsize, loc='lower center', bbox_to_anchor=(0.5, 1.15), ncol=ncol)  # Set legend properties

def get_folder(files_information, extra_folder=''):
    """Returns the folder path for saving figures, creating the directory if it doesn't exist."""
    try:
        folder_path = files_information.new_folder_path  # Try to get the folder path from the files information
    except:
        folder_path = files_information['new_folder_path']  # Fallback for dictionary-style access
    folder = folder_path if extra_folder == '' else f'{folder_path}/{extra_folder}'  # Append extra folder if provided
    if not os.path.exists(folder):
        os.makedirs(folder)  # Create the directory if it does not exist
    return folder  # Return the folder path

def save_figures(figures, width_cm, height_cm, files_information, extra_folder=''):
    """Saves the figures to the specified folder, both as PDF and PNG files."""
    folder = get_folder(files_information, extra_folder)  # Get the folder for saving the figures
    for name, fig in figures.items():  # Iterate through the figures dictionary
        fig.tight_layout()  # Ensure the layout is tight and labels are not clipped
        name = name.replace('|', '&').replace(' ', '_').lower()  # Format the figure name
        try:
            fig.set_size_inches(width_cm / 2.54, height_cm / 2.54)  # Convert size from cm to inches
        except AttributeError:
            pass
        fig.savefig(f'{folder}/{name}.pdf', dpi=800, bbox_inches="tight")  # Save the figure as a PDF file
        image_folder = f'{folder}/PNG'.replace('//', '/')  # Define the subfolder for PNG images
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)  # Create the subfolder if it does not exist
        fig.savefig(f'{image_folder}/{name}.png', dpi=400, bbox_inches="tight")  # Save the figure as a PNG file
        try:
            plt.close(fig)  # Close the figure to free up memory
            fig.clear()  # Clear any remaining figure elements
        except TypeError:
            fig.close()  # Close the figure if an error occurs
