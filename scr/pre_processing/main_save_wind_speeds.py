from scr.pre_processing.tasks_download_files import download_ERA5_data, download_global_data, download_regional_data
from scr.pre_processing.tasks_ws_file_creation import preprocess_climate_model_data, preprocess_era5_data
from scr.settings import REGIONAL_MODEL_NAME

if __name__ == '__main__':
    ### DOWNLOADING ###
    # Download GCM files
    download_global_data()
    # Download ERA5 files
    download_ERA5_data()
    # Download RCM files
    download_regional_data()


    ### PRE-PROCESSING ###
    preprocess_climate_model_data()
    preprocess_climate_model_data(REGIONAL_MODEL_NAME)
    preprocess_era5_data()
