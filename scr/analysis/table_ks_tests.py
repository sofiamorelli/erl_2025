import re
import numpy as np
from scipy.stats import kstest

from scr.analysis.Information import FileInformation
from scr.analysis.utils import get_flattened_wind_speed_and_resolution


def create_table_ks_tests(files_information):
    """
    Generates and prints a LaTeX table for Kolmogorov-Smirnov test results between different models'
    wind speed distributions. Each pair's $p$-value is displayed, with '0.0' indicating values below
    machine precision.

    Arguments:
        files_information (FileInformation): Metadata containing paths to wind speed files and model labels.
    """
    # Define the list of model files in reverse order and include ERA5 reference as the final entry
    model_file_list = files_information.files[::-1] + list(files_information.era5_files.values())[:1]
    print("Smallest positive normalized float64:", np.finfo(np.float64).tiny)  # For reference on machine precision

    # LaTeX formatting for a landscape table with the caption and description
    print("\\begin{landscape}")
    print("\\begin{table}")
    print("\\caption{$p$-values of Kolmogorov-Smirnov-Tests when comparing wind speed samples from different models, "
          "$0.0$ signifies a value below machine precision.}")
    print("\\label{tab:" + f':kolmogorov_smirnov_tests_{files_information.title}' + "}")
    print("\\begin{adjustbox}{width=\linewidth}")
    print("\\centering")

    # Collect model labels for each file to use as column and row headers in the table
    models = []
    for file_path in model_file_list:
        model_information = FileInformation(file_path, files_information.run_comparison)
        models.append(model_information.model_label)
    number_columns = len(models)

    # LaTeX column formatting for the table
    print("\\begin{tabular}{ c" + "|c" * (number_columns - 1) + " }")
    print("& ", " & ".join(list(models)[:-1]), " \\\\")  # Header row with model labels
    print("\\hline")  # Horizontal line after headers

    # Loop over model pairs to calculate and display the KS test $p$-values
    index1 = 1
    for file_path_1 in model_file_list[1:][::-1]:
        model_information_1 = FileInformation(file_path_1, files_information.run_comparison)
        wind_speed_1, _, _ = get_flattened_wind_speed_and_resolution(file_path_1, files_information)  # Data for model 1
        p_values = []
        index2 = 0
        for file_path_2 in model_file_list[:-1]:
            if (number_columns - index1) > index2:  # Only calculate KS test for upper triangular table
                wind_speed_2, _, _ = get_flattened_wind_speed_and_resolution(file_path_2, files_information)

                # Perform KS test and store the $p$-value for comparison
                ks_statistic, ks_p_value = kstest(wind_speed_1.flatten(), wind_speed_2.flatten())
                p_values.append(ks_p_value)
            else:
                p_values.append(None)  # Fill lower triangular part with None for formatting
            index2 += 1

        # Format each $p$-value for scientific notation or regular notation based on value
        formatted_entries = ["{:.1e}".format(x) if x is not None else "-" for x in p_values]
        formatted_string = ("$ & $".join(formatted_entries).replace('0.0e+00', '0.0')
                            .replace('e', ' \\cdot 10^{'))
        formatted_string = add_brace_before_dollar(formatted_string)  # Adjust braces in LaTeX

        # Write the formatted row to the table, with LaTeX escaping for proper display
        full_string = (f"{model_information_1.model_label} & ${formatted_string}$ \\\\".replace('$-$', '-'))
        print(full_string)
        index1 += 1

    # End LaTeX table structure and close landscape formatting
    print("\\end{tabular}")
    print("\\end{adjustbox}")
    print("\\end{table}")
    print("\\end{landscape}")


def format_p_values(p_values):
    """
    Formats $p$-values for LaTeX output, using scientific notation for values >= 1e-2.

    Arguments:
        p_values (list of float or None): List of $p$-values to format.

    Returns:
        list of str: Formatted $p$-values as strings, or '-' for None values.
    """
    formatted_entries = []
    for x in p_values:
        if x is None:
            formatted_entries.append("-")
        elif abs(x) < 1e-2:
            formatted_entries.append("{:.3f}".format(x))  # Standard format for small values
        else:
            formatted_entries.append("{:.1e}".format(x))  # Scientific notation for larger values
    return formatted_entries


def add_brace_before_dollar(s):
    """
    Modifies LaTeX strings for consistent formatting of scientific notation,
    adding a closing brace before dollar signs where necessary.

    Arguments:
        s (str): LaTeX formatted string to adjust.

    Returns:
        str: Modified string with correct brace placement.
    """
    pattern = r"\{.*?\$"  # Match pattern for braces around dollar signs

    def replace_match(match):
        return match.group(0)[:-1] + "}$"  # Insert brace before closing dollar sign

    result = re.sub(pattern, replace_match, s)
    return result
