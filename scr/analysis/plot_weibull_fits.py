import numpy as np
import scipy
from scipy import stats

from scr.analysis.Information import FileInformation
from scr.analysis.plot_statistical_distance_with_histograms import HISTOGRAM_ARGUMENTS_WS
from scr.analysis.utils import configure_scatter_plot, save_figures, get_flattened_wind_speed_and_resolution, \
    plot_scatter_point, set_legend_at_top, SCATTER_PLOT_WIDTH, create_quantile_plot, \
    configure_histogram_plot, HISTOGRAM_LABEL_SIZE, HISTOGRAM_WIDTH, HISTOGRAM_HEIGHT, SCATTER_PLOT_HEIGHT, \
    SCATTER_PLOT_ANNOTATION_SIZE, plot_regression_line
from scr.settings import WIND_SPEED_LABEL

# Define constants for plotting Weibull fits and related parameters
FIT_WIDTH = 10  # Width for Weibull fit plots
FIT_HEIGHT = 4  # Height for Weibull fit plots
FIT_LABEL_SIZE = 12  # Font size for plot labels
FOLDER_NAME = 'Weibull Fit'  # Folder name for saving Weibull fit results


def plot_weibull_fits(files_information, fixed_params={}):
    """
    Plots exponentiated Weibull fits to wind speed data for various models.
    The exponentiated Weibull distribution parameters are estimated, and plots of the parameters
    and their trends are generated.

    Args:
    files_information (FileInformation): Information about the files to be compared.
    fixed_params (dict): Parameters to fix during fitting, specified by index and value.
    """
    dist_weibull = stats.exponweib  # exponentiated Weibull distribution from scipy.stats
    bounds = [(0, 10), (0, 5), (0, 5), (0, 5)]  # Bounds for the Weibull parameters
    # Adjust bounds if fixed parameters are provided
    for i, val in fixed_params.items():
        bounds[i] = (val, val)

    all_parameter_names = list_parameters(dist_weibull)  # Get parameter names for Weibull distribution
    parameter_names = [p for k, p in enumerate(all_parameter_names) if k not in fixed_params.keys()]

    figures = {}  # Dictionary to store figures for plotting
    axes = []  # List to store axes for each parameter plot

    plot_name = 'Parameter Trend'  # Name for the plot title
    for parameter_name in parameter_names:  # For each parameter, create a plot
        fig, axs = configure_scatter_plot(parameter_name)  # Configure scatter plot for parameter trend
        var_name = f'{parameter_name} {files_information.title}'
        locals()[var_name] = fig  # Create a figure with the model's name
        figures[var_name] = fig
        locals()[f'axs_{var_name}'] = axs  # Create axis for each figure
        axes.append(axs)

    # Extract ERA5 file path for comparison
    era5_file_path = list(files_information.era5_files.values())[0]
    file_paths = [era5_file_path] + files_information.files  # Include ERA5 and model file paths

    # Flatten wind speed data from the ERA5 model
    era5_wind_speed, _, _ = get_flattened_wind_speed_and_resolution(era5_file_path, files_information)
    era5_mean = era5_wind_speed.mean()  # Calculate mean wind speed from ERA5 data

    # Prepare the LaTeX table for summarizing results
    title = f':summary_statistics_{files_information.title}'
    print("\\begin{table}")
    print("\\caption{All results}")
    print("\\label{tab:" + title + "}")
    print("\\centering")
    result_categories = (['Mean', 'Relative bias', 'Variance', 'Max value', 'Skewness', 'Kurtosis']
                         + parameter_names + ['JS div', 'Wasserstein div'])
    print("\\begin{tabular}{X" + "|X" * (len(result_categories)) + " }")
    print('Model & ' + " & ".join(list(result_categories)), " \\\\")
    print("\\hline")

    all_parameter_values = []  # List to store the values of the fitted parameters for each model
    spatial_resolutions = []  # List to store spatial resolution for each model
    for file_path in file_paths:  # Iterate over each file (ERA5 and models)
        model_information = FileInformation(file_path, files_information.run_comparison)
        # Flatten the wind speed data for the model
        wind_speed, spatial_resolution, _ = get_flattened_wind_speed_and_resolution(file_path, files_information)
        spatial_resolutions.append(spatial_resolution)
        # Fit the Weibull distribution to the wind speed data
        fit = stats.fit(dist_weibull, wind_speed, bounds=bounds)
        parameter_values = fit.params  # Get the fitted parameters
        all_parameter_values.append(parameter_values)  # Store parameter values for later use
        max_wind_speed = np.amax(wind_speed)  # Get the maximum wind speed in the model data
        grid = np.linspace(0, max_wind_speed, 500)  # Create a grid of values for plotting
        kde = stats.gaussian_kde(wind_speed)(grid)  # Perform kernel density estimation (KDE)

        # Call function to make Weibull fit plots
        make_weibull_fit_plots(wind_speed, grid, kde, parameter_names, parameter_values, dist_weibull,
                               model_information, files_information, fixed_params)

        # Calculate various statistics for the model and compare with ERA5
        mean = wind_speed.mean()  # Mean wind speed
        relative_bias = mean / era5_mean * 100 - 100  # Relative bias with respect to ERA5
        variance = wind_speed.var()  # Variance of wind speed
        max_ws = wind_speed.max()  # Maximum wind speed
        skewness = stats.skew(wind_speed)  # Skewness of wind speed distribution
        kurtosis = stats.kurtosis(wind_speed)  # Kurtosis of wind speed distribution
        # Histogram of wind speed
        distr_ws_hist, bin_edges_ws = np.histogram(wind_speed, **HISTOGRAM_ARGUMENTS_WS)
        era5_ws_hist, _ = np.histogram(era5_wind_speed, **HISTOGRAM_ARGUMENTS_WS)
        # Jensen-Shannon divergence and Wasserstein distance for distribution comparison
        js_div = scipy.spatial.distance.jensenshannon(era5_ws_hist, distr_ws_hist)
        wasserstein_div = stats.wasserstein_distance(era5_wind_speed, wind_speed)
        # Print the results in LaTeX table format
        print(f"{model_information.model_label} & {mean:.2f} & {relative_bias:.2f} "
              f"& {variance:.2f} & {max_ws:.2f} "
              f"& {skewness:.2f} & {kurtosis:.2f} ")

        # Iterate over the parameter names and plot the trends
        i = 0
        for k, parameter_name in enumerate(parameter_names):
            if k in fixed_params.keys():
                i += 1
            fig_name = f'{parameter_name} {files_information.title}'
            ax = locals()[f'axs_{fig_name}']
            parameter_value = parameter_values[i]
            plot_scatter_point(ax, spatial_resolution, parameter_value, model_information)
            # Print parameter values in LaTeX table format
            if parameter_value < 0.01:
                print(f"& {parameter_value:.2e}")
            else:
                print(f"& {parameter_value:.2f}")
            i += 1

        # Print the calculated distribution comparison metrics (JS and Wasserstein)
        print(f'& {js_div:.3f} & {wasserstein_div:.3f} \\\\')

    # Convert the spatial resolution and parameter values to numpy arrays
    x = np.array(spatial_resolutions)
    v = np.array(all_parameter_values)
    # Plot regression lines for each parameter across models
    for i, parameter_name in enumerate(parameter_names):
        plot_regression_line(x, v[:, i], axes[i], parameter_name, files_information, FOLDER_NAME)

    # Set the legend at the top of the plot
    set_legend_at_top([axes[0]], SCATTER_PLOT_ANNOTATION_SIZE)
    # Save all generated figures
    save_figures(figures, SCATTER_PLOT_WIDTH, SCATTER_PLOT_HEIGHT, files_information, FOLDER_NAME)


def make_weibull_fit_plots(wind_speed, grid, kde, parameter_names, parameter_values, distribution, model_information,
                           files_information, fixed_params):
    """
    Create plots showing the Weibull fit for the wind speed distribution.

    Arguments:
    wind_speed (ndarray): The wind speed data for the model.
    grid (ndarray): Grid of values to use for plotting the fit.
    kde (ndarray): The kernel density estimate of the wind speed data.
    parameter_names (list): List of parameter names for the Weibull distribution.
    parameter_values (list): List of the fitted parameter values.
    distribution (scipy.stats distribution): The Weibull distribution used for fitting.
    model_information (FileInformation): Information about the model being analyzed.
    files_information (FileInformation): Information about the files being compared.
    fixed_params (dict): Parameters that are fixed during fitting.
    """
    fig, ax = configure_histogram_plot()  # Configure a histogram plot for the Weibull fit
    ax.set_xlabel(WIND_SPEED_LABEL, fontsize=HISTOGRAM_LABEL_SIZE)  # Set the x-axis label
    color = model_information.color  # Use the model's color for the plot
    ax.plot(grid, kde, color=color, linestyle='dashed', label='kde')  # Plot the KDE
    ax.plot(grid, distribution.pdf(grid, *parameter_values), color=color, label='fit')  # Plot the Weibull fit

    j = 0
    for k, parameter_name in enumerate(parameter_names):  # Annotate the parameters on the plot
        if k in fixed_params.keys():
            j += 1
        parameter_value = parameter_values[j]
        # Annotate the parameter values on the plot, adjusting for small values
        if parameter_value < 0.01:
            ax.annotate('{} = {:.2e}'.format(parameter_name, parameter_value),
                        xy=(0.4, 0.6 - 0.12 * k),
                        xycoords='axes fraction', size=HISTOGRAM_LABEL_SIZE, ha='left', va='top')
        else:
            ax.annotate('{} = {:.2f}'.format(parameter_name, parameter_value),
                        xy=(0.4, 0.6 - 0.12 * k),
                        xycoords='axes fraction', size=HISTOGRAM_LABEL_SIZE, ha='left', va='top')
        j += 1

    # Add the legend to the plot
    ax.legend(frameon=False, fontsize=HISTOGRAM_LABEL_SIZE - 2, loc='upper left', bbox_to_anchor=(0.3, 1.1))
    # Save the figure for the Weibull fit
    title = f'Weibull Fit Curve {files_information.title} {model_information.model_label}'
    save_figures({title: fig}, HISTOGRAM_WIDTH, HISTOGRAM_HEIGHT, files_information, FOLDER_NAME)

    # Create and save quantile-quantile plot comparing sample and theoretical quantiles
    sample_quantiles = np.percentile(wind_speed, np.linspace(0, 99, 100))
    fitted_distribution = distribution(*parameter_values)
    fitted_quantiles = fitted_distribution.ppf(np.linspace(0, 1, 100))
    create_quantile_plot(files_information, model_information, sample_quantiles, fitted_quantiles,
                         'Theoretical', FOLDER_NAME)


def list_parameters(distribution):
    """
    Return a list of the parameters for a given distribution.

    Args:
    distribution (str or distribution): The name or the distribution object.

    Returns:
    list: List of parameter names.
    """
    if isinstance(distribution, str):
        distribution = getattr(stats, distribution)  # Get the distribution from scipy.stats by name
    if distribution.shapes:
        parameters = [name.strip() for name in distribution.shapes.split(',')]  # Get shape parameters
    else:
        parameters = []

    # Add location and scale parameters based on the distribution type
    if distribution.name in stats._discrete_distns._distn_names:
        parameters += ['loc']
    elif distribution.name in stats._continuous_distns._distn_names:
        parameters += ['loc', 'scale']
    else:
        raise ValueError("Distribution name not found in discrete or continuous lists.")
    return parameters  # Return the list of parameters for the distribution
