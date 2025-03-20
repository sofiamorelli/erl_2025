import os

from matplotlib import pyplot as plt

from scr.analysis.utils import save_figures, adapt_axes
from scr.settings import WIND_POWER_LABEL, WIND_SPEED_LABEL, BASE_DIR_FIG

FIG_WIDTH = 9
FIG_HEIGHT = 2.5
LABEL_SIZE = 8


def plot_power_curve(power_curve_information):
    """
    Plots the power curve for a turbine, highlighting the regions corresponding to the cut-in,
    rated, and cut-off wind speeds. The plot also includes labels and annotations for these key wind speeds.

    Parameters:
    - power_curve_information: Contains the turbine's power curve, cut-in, rated, and cut-off wind speeds.
    """
    # Extract turbine object from the input information
    turbine = power_curve_information.turbine

    # Check if the turbine has a valid power curve
    if turbine.power_curve is not None:
        # Extract wind speed (m/s) and corresponding power values (in MW)
        wind_speed = turbine.power_curve['wind_speed'].to_numpy()
        power_mw = turbine.power_curve['value'].to_numpy() / 1e6  # Convert power from W to MW

        # Create a new figure and axis for plotting
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

        # Plot the power curve (wind speed vs power)
        ax.plot(wind_speed, power_mw, '-', color='black', linewidth=0.5)

        # Plot the cut-in wind speed region (Blue area)
        cut_in = power_curve_information.cut_in
        ax.axvspan(0, cut_in, color='blue', alpha=0.15)  # Fill the area before cut-in with blue
        ax.axvline(x=cut_in, color='grey', linewidth=0.8)  # Vertical line at the cut-in wind speed
        # Add text label for cut-in wind speed ('$w_I$')
        ax.text(cut_in, max(power_mw) * 1.05, '$w_I$', color='grey', ha='center', va='bottom', fontsize=LABEL_SIZE)

        # Plot the rated wind speed region (Orange area)
        rated = power_curve_information.rated
        ax.axvspan(cut_in, rated, color='orange', alpha=0.2)  # Fill the area between cut-in and rated with orange
        ax.axvline(x=rated, color='grey', linewidth=0.8)  # Vertical line at the rated wind speed
        # Add text label for rated wind speed ('$w_R$')
        ax.text(rated, max(power_mw) * 1.05, '$w_R$', color='grey', ha='center', va='bottom', fontsize=LABEL_SIZE)

        # Plot the cut-off wind speed region (Green area)
        cut_off = power_curve_information.cut_off
        ax.axvspan(rated, cut_off, color='green', alpha=0.15)  # Fill the area between rated and cut-off with green
        ax.axvline(x=cut_off, color='grey', linewidth=0.8)  # Vertical line at the cut-off wind speed
        # Add text label for cut-off wind speed ('$w_O$')
        ax.text(cut_off, max(power_mw) * 1.05, '$w_O$', color='grey', ha='center', va='bottom', fontsize=LABEL_SIZE)

        # Plot a small blue region representing the area between cut-off and cut-in wind speeds (secondary blue area)
        ax.axvspan(cut_off, cut_off + cut_in, color='blue', alpha=0.15)

        # Set labels for the axes
        ax.set_xlabel(WIND_SPEED_LABEL, fontsize=LABEL_SIZE)
        ax.set_ylabel(WIND_POWER_LABEL, fontsize=LABEL_SIZE)

        # Adjust axes to make sure labels and text fit nicely
        adapt_axes(ax, LABEL_SIZE, 0)

        # Ensure the layout is tight, reducing unnecessary whitespace around the plot
        fig.tight_layout()

        # Define the folder path to save the figure (inside a 'Power Curve' folder)
        folder_path = os.path.join(BASE_DIR_FIG, 'Power Curve')

        # Save the plot to the designated folder with a specific name based on the turbine's name
        save_figures({f'power_curve_{power_curve_information.turbine_name}': fig}, FIG_WIDTH, FIG_HEIGHT,
                     {'new_folder_path': folder_path})
