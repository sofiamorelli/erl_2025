from matplotlib import pyplot as plt

from scr.analysis.Information import FilesInformation, PowerCurveInformation, FileInformation
from scr.analysis.plot_cumulative_wp import plot_cumulative_wind_power_ma, plot_only_cumulative_wp_for_large_data
from scr.analysis.plot_extreme_value_percentage import plot_extreme_wind_speed_percentage
from scr.analysis.plot_statistical_distance_with_histograms import plot_statistical_distance
from scr.analysis.plot_grids import plot_grids_with_average_wind_speeds, plot_first_frames
from scr.analysis.plot_power_curve import plot_power_curve
from scr.analysis.plot_qq_comparison import plot_qq_comparisons
from scr.analysis.plot_summary_statistics import plot_summary_statistics_all_models
from scr.analysis.plot_weibull_fits import plot_weibull_fits
from scr.analysis.plot_ws_distribution_comparison import plot_distribution_comparison_group, \
    plot_distribution_comparison_single, plot_all_kde_together, plot_distribution_differences_paper, \
    plot_all_kde_together_with_highlighted_model
from scr.analysis.table_all_values import save_latex_table_all_values
from scr.analysis.table_ks_tests import create_table_ks_tests
from scr.analysis.utils import get_flattened_wind_speed_and_resolution
from scr.settings import COUNTRY, MODELS_WITH_SEVERAL_RUNS, START_YEAR, COLORS_RCMs, LINE_STYLES_RCMs

def paper_spatial_resolution_plots():
    """
    Function to generate plots for spatial resolution paper in the region of Europe
    with multiple turbines and heights as specified.
    Includes power curve plots, KDE and QQ plots, distribution comparisons,
    and wind power plots with optional additional analysis depending on the start year.
    """
    for hub_height, turbine_name in [#(78, 'V80/2000'),
                                     #(85, 'E-70/2000'),
                                     (126, 'V126/3450'),
                                     #(134, 'N131/3300'),
                                     (140, 'V164/9500')
                                     ]:
        region = 'EuropeLand'
        # Filter and get files for first model runs
        first_runs_files_information = (FilesInformation(region, hub_height, turbine_name)
                                        .filter_files_for_first_model_runs())

        #plot_power_curve(first_runs_files_information.turbine)  # Plot turbine power curve

        # Generate various spatial resolution plots
        #plot_first_frames(first_runs_files_information) # Visualize first time frames of the gridded data
        #plot_all_kde_together(first_runs_files_information) # Wind speed distribution as kernel density plot
        #plot_all_kde_together(first_runs_files_information, True)  # Generated wind power from KDE
        #plot_qq_comparisons(first_runs_files_information)  # QQ comparisons
        #plot_distribution_differences_paper(first_runs_files_information)  # Distribution differences
        #plot_only_cumulative_wp_for_large_data(first_runs_files_information)  # Cumulative wind power plot

        # Additional analysis based on the starting year
        if START_YEAR == 2005:
            for file_path in first_runs_files_information.files:
                # Get wind speed and resolution data
                _, spatial_resolution, _ = get_flattened_wind_speed_and_resolution(file_path,
                                                                                   first_runs_files_information)
                model_information = FileInformation(file_path, first_runs_files_information.run_comparison)
                print(model_information.model_name, spatial_resolution)  # Print model name and spatial resolution

            save_latex_table_all_values(first_runs_files_information, 'all_values') # LaTeX table with values and plots
            plot_summary_statistics_all_models(first_runs_files_information)  # Summary statistics plot
            plot_extreme_wind_speed_percentage(first_runs_files_information) # Extreme value plots
            plot_statistical_distance(first_runs_files_information)  # Jensen–Shannon distance plot
            plot_statistical_distance(first_runs_files_information, 'Wasserstein-1 distance', True)  # Wasserstein-1 distance plot

        elif START_YEAR == 1995:
            # Process for global boundary models
            for global_boundary_model in list(COLORS_RCMs.keys())[:1]:
                first_runs_files_information_global_boundary_model = (FilesInformation(region, hub_height, turbine_name)
                                                                      .filter_for_global_boundary_model_name(global_boundary_model))
                plot_only_cumulative_wp_for_large_data(first_runs_files_information_global_boundary_model)

            # Process for regional models
            for regional_model in list(LINE_STYLES_RCMs.keys())[:1]:
                first_runs_files_information_regional_model = (FilesInformation(region, hub_height, turbine_name)
                                                               .filter_for_regional_model_name(regional_model))
                plot_only_cumulative_wp_for_large_data(first_runs_files_information_regional_model)

# Call the function to execute the plots

def master_thesis_plots():
    """
    Function to generate all plots and statistical tests required for the master's thesis.
    It processes files for both the first model runs and run comparisons in the region of Germany.
    for various turbine configurations and regions, generating corresponding visualizations and tables.

    - Power curve plot for the turbine.
    - First model runs: includes wind speed grids, statistical tests, KDE plots, etc.
    - Run comparison: compares multiple runs of a model.
    - Cumulative wind power: compares power outputs across regions and model configurations.
    """
    hub_height_ma = 100  # Set the hub height for the turbine
    turbine_name_ma = 'E-115/3200'  # Define the turbine name
    power_curve_information = PowerCurveInformation(turbine_name_ma, hub_height_ma)  # Get power curve information
    plot_power_curve(power_curve_information)  # Plot the power curve for the turbine

    # First Runs - Process files and generate plots for the first model runs
    first_runs_files_information = (FilesInformation(COUNTRY, hub_height_ma, turbine_name_ma)
                                    .filter_files_for_first_model_runs())  # Filter first runs
    plot_grids_with_average_wind_speeds(first_runs_files_information)  # Plot grids with average wind speeds
    create_table_ks_tests(first_runs_files_information)  # Create KS test tables
    plot_all_kde_together(first_runs_files_information)  # Plot KDE for all models
    plot_all_kde_together_with_highlighted_model(first_runs_files_information, 'MOHC')  # Highlight specific model (MOHC)
    plot_all_kde_together_with_highlighted_model(first_runs_files_information, 'NCC')  # Highlight specific model (NCC)
    plot_summary_statistics_all_models(first_runs_files_information)  # Plot summary statistics for all models
    plot_statistical_distance(first_runs_files_information)  # Plot statistical distances
    plot_statistical_distance(first_runs_files_information, 'Wasserstein distance')  # Plot Wasserstein distance
    plot_weibull_fits(first_runs_files_information)  # Plot Weibull fits with summary statistics
    plot_distribution_comparison_single(first_runs_files_information, power_curve_information)  # Distribution comparison with power curve
    plot_qq_comparisons(first_runs_files_information)  # QQ plot comparisons for the data

    # Run comparison - Generate plots comparing multiple runs of a model
    for model_name in MODELS_WITH_SEVERAL_RUNS:  # Loop through models with several runs
        run_comparison_model_list = (FilesInformation(COUNTRY, hub_height_ma, turbine_name_ma)
                                     .filter_for_run_comparison(model_name))  # Filter files for model run comparison
        plot_distribution_comparison_group(run_comparison_model_list, model_name, power_curve_information)  # Compare distributions for the group of runs

    # Consider turbines for cumulative wind power - Analyze and plot cumulative wind power data for different turbines
    region = 'Germany_Turbine'  # Set region for cumulative wind power analysis

    # First Runs - Process files and generate plots for the first model runs
    first_runs_turbine_files_information = (FilesInformation(region, hub_height_ma, turbine_name_ma)
                                            .filter_files_for_first_model_runs())  # Filter first runs for the specific region
    plot_grids_with_average_wind_speeds(first_runs_turbine_files_information)  # Plot grids with average wind speeds for region
    plot_cumulative_wind_power_ma(first_runs_turbine_files_information, power_curve_information)  # Plot cumulative wind power for region

    # Run comparison - Generate plots comparing multiple runs for turbines in the region
    for model_name in MODELS_WITH_SEVERAL_RUNS:  # Loop through models with several runs
        run_comparison_model_list = FilesInformation(region).filter_for_run_comparison(model_name)  # Filter files for run comparison
        plot_cumulative_wind_power_ma(run_comparison_model_list, model_name)  # Plot cumulative wind power for run comparison


if __name__ == '__main__':
    plt.rcParams["font.family"] = "Times New Roman"

    paper_spatial_resolution_plots()
    master_thesis_plots()
