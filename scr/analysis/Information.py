import os

import numpy as np
from windpowerlib import WindTurbine, ModelChain

from scr.pre_processing.tasks_download_files import TIME_POINTS
from scr.settings import ERA5_NAME, BASE_DIR_DAT, BASE_DIR_FIG, COLORS, LINE_STYLES_FOR_RUN_COMPARISON, START_YEAR, \
    END_YEAR, GLOBAL_MODEL_NAME, REGIONAL_MODEL_NAME, COLORS_RCMs, LINE_STYLES_RCMs, CONSTANT_GLOBAL_BOUNDARY_MODEL, \
    CONSTANT_REGIONAL_MODEL


class FilesInformation:
    def __init__(self, region_name, hub_height, turbine_name, run_comparison=False):
        """
            Initializes the FilesInformation class, responsible for handling file paths and filtering wind data files
            based on specified parameters.

            Arguments:
                region_name {str}: Name of the region for file organization.
                hub_height {float}: Hub height of the wind turbine.
                turbine_name {str}: Name/type of the wind turbine.
                run_comparison {bool}: Flag for indicating if run comparison is required.
        """
        self.region = region_name  # Set region name
        original_folder_path = os.path.join(BASE_DIR_DAT, region_name)  # Path to data files
        self.original_folder_path = original_folder_path  # Store path
        all_files = [os.path.join(original_folder_path, f) for f in sorted(os.listdir(original_folder_path), reverse=True)
                     if not f.endswith('.npy') and not f.endswith('.npz')]  # List all valid files in directory
        self.files = all_files  # Store all files
        era5_files = {}  # Dictionary to hold ERA5 files
        for file_path in all_files:
            for time_point in TIME_POINTS.keys():
                if ERA5_NAME in file_path and time_point in file_path:
                    era5_files[f'{ERA5_NAME}_{time_point}'] = file_path  # Add file to ERA5 files dictionary
        self.era5_files = era5_files  # Store ERA5 files
        self.new_folder_path = os.path.join(BASE_DIR_FIG, region_name, f'{hub_height}m')  # Set output path for figures
        self.title = f'{region_name}_{hub_height}m_{START_YEAR}-{END_YEAR}'  # Generate base title
        self.run_comparison = run_comparison  # Set run comparison flag
        self.hub_height = hub_height  # Store hub height
        self.turbine = PowerCurveInformation(turbine_name, hub_height)  # Initialize turbine information

    def filter_files_for_first_model_runs(self):
        """Filters files to include only the first runs of each model."""
        self.files = [file for file in self.files if "r1" in file]  # Keep only files with "r1" in their name
        self.title += '_First-Runs'  # Update title to indicate filtering
        return self

    def filter_files_for_global_climate_models(self):
        """Filters files to include only those associated with global climate models."""
        self.files = [file for file in self.files if GLOBAL_MODEL_NAME in file]  # Filter for global models
        self.title += '_' + GLOBAL_MODEL_NAME  # Update title with global model name
        return self

    def filter_files_for_model_name(self, model_name):
        """Filters files to include only those associated with a specific model name."""
        list_of_model_file_names = []
        for file_name in self.files:
            if model_name in file_name:
                list_of_model_file_names.append(file_name)  # Add files matching model name to list
        self.files = list_of_model_file_names  # Set filtered files
        self.title += f'_{model_name}'  # Update title with model name
        return self

    def filter_for_run_comparison(self, model_name):
        """Filters files for run comparison by a specified model name."""
        self.run_comparison = True  # Enable run comparison
        os.path.join(BASE_DIR_FIG, 'Run Comparison')  # Set path for run comparison figures
        return self.filter_files_for_model_name(model_name)  # Apply model name filter

    def filter_files_for_model_list(self, model_file_list, model_group_name):
        """Filters files to include only those from a specified list of model names."""
        list_of_model_file_names = []
        for file_name in self.files:
            for model_name in model_file_list:
                if model_name in file_name:
                    list_of_model_file_names.append(file_name)  # Add files matching any model name in list
        self.files = list_of_model_file_names  # Set filtered files
        self.title += f'_{model_group_name}'  # Update title with model group name
        return self.filter_files_for_first_model_runs()  # Also filter for first runs

    def filter_for_global_boundary_model_name(self, model_name):
        """Filters files for a specified global boundary model."""
        files_list = self.files.copy()  # Copy list of files
        for file_name in self.files:
            model_information = FileInformation(file_name)  # Get model information
            global_model_label = model_information.model_label.rsplit('-', 1)[0]
            if model_name not in global_model_label or not REGIONAL_MODEL_NAME in file_name:
                files_list.remove(file_name)  # Remove files that don't match criteria
        self.files = files_list  # Set filtered files
        self.title += f'_GCM-{model_name}'  # Update title with global model name
        return self

    def filter_for_regional_model_name(self, model_name):
        """Filters files for a specified regional model."""
        files_list = self.files.copy()  # Copy list of files
        for file_name in self.files:
            model_information = FileInformation(file_name)  # Get model information
            regional_model_label = model_information.model_label.rsplit('-', 1)[-1]
            if model_name not in regional_model_label or not REGIONAL_MODEL_NAME in file_name:
                files_list.remove(file_name)  # Remove files that don't match criteria
        self.files = files_list  # Set filtered files
        self.title += f'_RCM-{model_name}'  # Update title with regional model name
        return self

    def remove_files_by_model_name(self, model_name):
        """Removes files associated with a specific model name."""
        files_list = self.files.copy()  # Copy list of files
        for file_name in self.files:
            if model_name in file_name:
                files_list.remove(file_name)  # Remove files that match model name
        self.files = files_list  # Set updated file list
        return self


class FileInformation:
    def __init__(self, file_name, run_comparison=False):
        """
            Initializes the FileInformation class, extracting model-specific information from the file name.

            Arguments:
                file_name {str}: Name of the file to process.
                run_comparison {bool}: Flag for indicating if run comparison information is required.
        """
        self.file_name = file_name  # Store file name
        components = file_name.split('/')[-1].split('.')[0].split('_')  # Split file name into components
        model_category = components.pop(1)  # Extract model category
        self.model_category = model_category  # Store model category
        model_name = components.pop(1)  # Extract model name
        self.model_name = model_name  # Store model name
        try:
            # Assign color based on model characteristics
            if model_category == REGIONAL_MODEL_NAME:
                global_boundary_model = model_name.rsplit('-', 1)[0]
                regional_model = model_name.rsplit('-', 1)[-1]
                if global_boundary_model == CONSTANT_GLOBAL_BOUNDARY_MODEL and regional_model == CONSTANT_REGIONAL_MODEL:
                    self.color = 'purple'
                elif global_boundary_model == CONSTANT_GLOBAL_BOUNDARY_MODEL:
                    self.color = 'red'
                elif regional_model == CONSTANT_REGIONAL_MODEL:
                    self.color = 'blue'
            else:
                self.color = COLORS[model_name]  # Use predefined color
        except KeyError:
            print(f'WARNING: Color for {model_name} is not defined')
            self.color = 'grey'  # Default color if not defined

        run = components.pop(1)  # Extract run identifier
        self.run = run  # Store run
        line_style = 'solid'  # Default line style
        # Set line style based on run comparison or model specifics
        self.line_style = line_style
        self.model_label = model_name if not run_comparison else f'{model_name}_{run}'  # Set model label
        self.arguments = {'color': self.color, 'linestyle': line_style}  # Define arguments for plotting


class PowerCurveInformation:
    def __init__(self, turbine_name, hub_height):
        """
            Initializes the PowerCurveInformation class for storing turbine power characteristics.

            Arguments:
                turbine_name {str}: Name of the turbine type.
                hub_height {float}: Height of the turbine hub.
        """
        self.turbine_name = turbine_name.split('/')[0]  # Extract base turbine name
        turbine_characteristics = {
            "turbine_type": turbine_name,  # Set turbine type
            "hub_height": hub_height  # Set hub height
        }
        turbine = WindTurbine(**turbine_characteristics)  # Create WindTurbine instance
        self.turbine = turbine  # Store turbine instance
        mc_turbine = ModelChain(turbine)  # Create model chain for turbine
        wind_speed = np.arange(0, 100, 0.1)  # Define wind speed range
        power = mc_turbine.calculate_power_output(wind_speed, 1)  # Calculate power output
        first_nonzero_index = np.argmax(power > 0)  # Find cut-in speed
        self.cut_in = wind_speed[first_nonzero_index]  # Set cut-in speed
        first_zero_after_nonzero = np.argmax(power[first_nonzero_index:] == 0) + first_nonzero_index
        self.cut_off = wind_speed[first_zero_after_nonzero]
