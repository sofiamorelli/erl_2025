import os

import numpy as np

from scr.analysis.Information import FileInformation
from scr.analysis.utils import get_flattened_wind_speed_and_resolution, create_quantile_plot
from scr.settings import ERA5_NAME

# Directory name where QQ plot comparison results will be saved
FOLDER_NAME = 'QQ Plot Comparison'

def plot_qq_comparisons(files_information):
    """
    Plots the QQ plot comparisons between the model outputs and ERA5 data for each model file.

    Parameters:
    - files_information: Contains metadata about the model files, including the paths to the ERA5 files
      and other necessary information for comparison.
    """
    # Extract the file path for the first ERA5 file (used as the reference dataset for comparison)
    era5_file_path = list(files_information.era5_files.values())[0]

    # Compute and save the quantiles for the ERA5 data (reference data)
    era5_quantiles = save_quantiles_in_file(era5_file_path, files_information)

    # Iterate through each model file and generate a QQ plot comparing it to ERA5 data
    for file_path in files_information.files:
        make_qq_fig(file_path, era5_quantiles, files_information)


def make_qq_fig(file_path, era5_quantiles, files_information):
    """
    Creates a QQ plot for a single model file, comparing its quantiles to the ERA5 reference quantiles.

    Parameters:
    - file_path: The path to the model file for which the QQ plot is to be created.
    - era5_quantiles: The quantiles of the ERA5 data used as the reference for comparison.
    - files_information: Contains metadata about the model files and additional settings for comparison.
    """
    # Create an object to store model-specific information for the plot
    model_information = FileInformation(file_path, files_information.run_comparison)

    # Compute and save the quantiles for the model data
    sample_quantiles = save_quantiles_in_file(file_path, files_information)

    # Generate the QQ plot using the model's quantiles and ERA5's quantiles
    create_quantile_plot(files_information, model_information, sample_quantiles, era5_quantiles,
                         ERA5_NAME, FOLDER_NAME)


def save_quantiles_in_file(file_path, files_information):
    """
    Computes and saves the quantiles for the wind speed data in a file. If the quantiles are already computed
    and saved, it loads them from the saved file.

    Returns:
    - quantiles: The computed quantiles (as a NumPy array) for the wind speed data.
    """
    # Get the hub height from the files information (this is used to identify the specific dataset)
    height = files_information.hub_height

    # Construct the path to save/load the quantiles, based on the height of the turbine
    quantiles_file_path = file_path.replace('.nc', f'_{height}m_quantiles.npy')

    # Check if the quantiles file already exists. If it does, load the quantiles from it.
    if not os.path.isfile(quantiles_file_path):
        # If the quantiles file doesn't exist, calculate the quantiles from the wind speed data
        wind_speed, _, _ = get_flattened_wind_speed_and_resolution(file_path, files_information)

        # Calculate the quantiles (from 0 to 99th percentile in 100 steps)
        quantiles = np.percentile(wind_speed, np.linspace(0, 99, 100))

        # Save the computed quantiles to a file for future use
        with open(quantiles_file_path, 'wb') as f:
            np.save(f, quantiles)

        # Return the computed quantiles
        return quantiles
    else:
        # If the quantiles file exists, load and return the quantiles from the file
        quantiles = np.load(quantiles_file_path)
        return quantiles
