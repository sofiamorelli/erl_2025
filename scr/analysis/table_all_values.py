import numpy as np
import scipy
from scipy import stats

from scr.analysis.Information import FileInformation
from scr.analysis.plot_statistical_distance_with_histograms import HISTOGRAM_ARGUMENTS_WS
from scr.analysis.utils import get_flattened_wind_speed_and_resolution, get_folder, get_npy_file_path


def save_latex_table_all_values(files_information, table_title):
    """
    Generates and saves a LaTeX table containing statistical results for wind speed analysis
    of multiple models and the ERA5 reference data.

    Arguments:
        files_information (FileInformation): Metadata for files, including paths, titles,
                                             and additional parameters.
    """
    # Define the table title and output folder, and open a .txt file for writing LaTeX code
    title = f'{table_title}'  # Table title
    folder = get_folder(files_information)  # Get path to store output
    with open(f'{folder}/{title}.txt', 'w') as f:
        # Write the beginning of the LaTeX table structure, including table title and label
        f.write("\\begin{table}\n")
        f.write("\\caption{All results}\n")
        f.write("\\label{tab:" + title + "}\n")
        f.write("\\centering\n")

        # Define column names and header for table, listing each result category
        result_categories = ['Mean', 'JS div', 'WS div', 'Average Max value', 'Low percentage', 'High percentage']
        f.write("\\begin{tabular}{X" + "|X" * (len(result_categories)) + " }\n")
        f.write('Model & ' + " & ".join(result_categories) + " \\\\\n")  # Header with categories
        f.write("\\hline\n")  # Add a horizontal line under the header

        # Get ERA5 reference file path and load ERA5 wind speed data
        era5_file_path = list(files_information.era5_files.values())[0]
        file_paths = [era5_file_path] + files_information.files  # Include ERA5 and other files
        era5_wind_speed, _, _ = get_flattened_wind_speed_and_resolution(era5_file_path, files_information)

        cut_in = files_information.turbine.cut_in
        print(cut_in)
        cut_off = files_information.turbine.cut_off
        print(cut_off)

        # Loop through each file, including ERA5 and model files, to compute and write metrics
        for file_path in file_paths:
            # Retrieve metadata and wind speed data for the current file
            model_information = FileInformation(file_path, files_information.run_comparison)
            wind_speed, spatial_resolution, _ = get_flattened_wind_speed_and_resolution(file_path, files_information)

            # Calculate mean wind speed and write to LaTeX table
            mean = wind_speed.mean()
            f.write(f"{model_information.model_label} & {mean:.2f} ")

            # Calculate Jensen-Shannon divergence between model and ERA5 distributions
            distr_ws_hist, _ = np.histogram(wind_speed, **HISTOGRAM_ARGUMENTS_WS)
            era5_ws_hist, _ = np.histogram(era5_wind_speed, **HISTOGRAM_ARGUMENTS_WS)
            js_div = scipy.spatial.distance.jensenshannon(era5_ws_hist, distr_ws_hist)
            f.write(f"& {js_div:.3f} ")

            # Calculate Wasserstein distance between ERA5 and model distributions
            wasserstein_div = stats.wasserstein_distance(era5_ws_hist, distr_ws_hist)
            f.write(f"& {wasserstein_div:.3f} ")

            # Sort wind speeds by their value from smallest to biggest
            sorted_ws = np.sort(wind_speed)
            number_data_points = len(sorted_ws)
            # Calculate the average of the top 100 maximum wind speeds and write to table
            max_ws = np.mean(sorted_ws[-100:])
            f.write(f"& {max_ws:.2f} ")

            index_below_cut_in = np.argmax(sorted_ws > cut_in)
            low_ws = 100 * index_below_cut_in / number_data_points
            f.write(f"& {low_ws:.2f} ")

            index_above_cut_off = np.argmax(sorted_ws > cut_off)
            high_ws = 100 * (number_data_points - index_above_cut_off) / number_data_points
            f.write(f"& {high_ws:.3f} \\\\\n")

            # Save computed metrics to an .npz file for each model, for future reference or plotting
            base_file_path = get_npy_file_path(file_path)
            height = files_information.hub_height  # Turbine hub height
            np.savez(f'{base_file_path}_{height}m_trends.npz', x=spatial_resolution, mean=mean,
                     max=max_ws, low=low_ws, high=high_ws, js=js_div, w1=wasserstein_div)
