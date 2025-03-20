import os
import zipfile
import requests

from scr.Region import Country
from scr.settings import GLOBAL_MODEL_NAME, SURFACE_WIND_SPEED, U_COMPONENT_CMIP6, V_COMPONENT_CMIP6, \
    START_YEAR, \
    END_YEAR, ERA5_NAME, ORIGINAL_DATA_DIR, CDSAPI_URL, CDSAPI_KEY, COUNTRY, REGIONAL_MODEL_NAME

# MPI API to download CMIP6 data
from cdo import *
cdo.debug = True
cdo = Cdo()

# Copernicus API to download ERA5
import cdsapi
os.environ["CDSAPI_URL"] = CDSAPI_URL
os.environ["CDSAPI_KEY"] = CDSAPI_KEY
c = cdsapi.Client()

# File names of CMIP6 GCM model output to download
# Each entry maps a GCM identifier to a tuple, containing:
#   - A file path pattern (with placeholders) for accessing specific data files on the server.
#   - A list of time ranges, defining periods for each downloaded file based on the way they are uploaded for CMIP6.
GCM_FILES = {
    'MPI-HR': ('MPI-M/MPI-ESM1-2-HR/historical/r$RUN$i1p1f1/6hrPlevPt/$VAR$/gn/v20210901/'
               '$VAR$_6hrPlevPt_MPI-ESM1-2-HR_historical_r$RUN$i1p1f1_gn_$YEARS$.nc',
               [f"{year}01010600-{year+5}01010000" for year in range(START_YEAR - (START_YEAR % 5), END_YEAR, 5)]),
    'MPI-LR': ('MPI-M/MPI-ESM1-2-LR/historical/r$RUN$i1p1f1/6hrPlevPt/$VAR$/gn/v20190815/'
               '$VAR$_6hrPlevPt_MPI-ESM1-2-LR_historical_r$RUN$i1p1f1_gn_$YEARS$.nc',
               [f"{year}01010600-{year+10}01010000" for year in range(START_YEAR - (START_YEAR % 10), END_YEAR, 10)]),
    'EC-EARTH': ('EC-Earth-Consortium/EC-Earth3/historical/r$RUN$i1p1f1/6hrPlev/$VAR$/gr/v20200310/'
               '$VAR$_6hrPlev_EC-Earth3_historical_r$RUN$i1p1f1_gr_$YEARS$.nc',
               [f"{year}01010300-{year}12312100" for year in range(START_YEAR, END_YEAR)]),
    'CMCC': ('CMCC/CMCC-CM2-SR5/historical/r$RUN$i1p1f1/6hrPlev/$VAR$/gn/v20200904/'
             '$VAR$_6hrPlev_CMCC-CM2-SR5_historical_r$RUN$i1p1f1_gn_$YEARS$.nc',
             [f"{year}01010300-{year+4}12312100" for year in range(START_YEAR - (START_YEAR % 5), END_YEAR, 5)]),
    'NCC-HR': ('NCC/NorESM2-MM/historical/r$RUN$i1p1f1/6hrPlev/$VAR$/gn/v20191108/'
               '$VAR$_6hrPlev_NorESM2-MM_historical_r$RUN$i1p1f1_gn_$YEARS$.nc',
               [f"{year}01010300-{year+4}12312100" for year in range(START_YEAR - (START_YEAR % 5), END_YEAR, 5)]),
    'NCC-LR': ('NCC/NorESM2-LM/historical/r$RUN$i1p1f1/6hrPlev/$VAR$/gn/v20191108/'
               '$VAR$_6hrPlev_NorESM2-LM_historical_r$RUN$i1p1f1_gn_$YEARS$.nc',
               [f"{year}01010300-{year+4}12312100" for year in range(START_YEAR - (START_YEAR % 5), END_YEAR, 5)]),
    'MOHC-LR': ('MOHC/HadGEM3-GC31-LL/historical/r$RUN$i1p1f3/6hrPlev/$VAR$/gn/v20201103/'
                '$VAR$_6hrPlev_HadGEM3-GC31-LL_historical_r$RUN$i1p1f3_gn_$YEARS$.nc',
                [f"{year}01010300-{year+14}12312100" for year in range(START_YEAR - (START_YEAR % 15), END_YEAR, 15)]),
    'MOHC-HR': ('MOHC/HadGEM3-GC31-MM/historical/r$RUN$i1p1f3/6hrPlev/$VAR$/gn/v20200923/'
               '$VAR$_6hrPlev_HadGEM3-GC31-MM_historical_r$RUN$i1p1f3_gn_$YEARS$.nc',
               [f"{year}01010300-{year}12302100" for year in range(START_YEAR, END_YEAR)]),
    'JAP': ('MIROC/MIROC6/historical/r$RUN$i1p1f1/6hrPlev/$VAR$/gn/v20191204/'
               '$VAR$_6hrPlev_MIROC6_historical_r$RUN$i1p1f1_gn_$YEARS$.nc',
               [f"{year}01010300-{year}12312100" for year in range(START_YEAR, END_YEAR)]),
    'IPSL': ('IPSL/IPSL-CM6A-LR-INCA/historical/r$RUN$i1p1f1/6hrPlev/$VAR$/gr/v20210216/'
               '$VAR$_6hrPlev_IPSL-CM6A-LR-INCA_historical_r$RUN$i1p1f1_gr_$YEARS$.nc',
               ['185001010300-201412312100']),
}

# List of multiple CMIP6 server URLs to handle data download requests
SERVER_LIST = ['esgf.ceda.ac.uk/thredds/fileServer/esg_cmip6/CMIP6/CMIP/',
              'esgf3.dkrz.de/thredds/fileServer/cmip6/CMIP/',
              'esgf-data1.llnl.gov/thredds/fileServer/css03_data/CMIP6/CMIP/',
              'eagle.alcf.anl.gov/thredds/fileServer/css03_data/CMIP6/CMIP/',
              'noresg.nird.sigma2.no/thredds/fileServer/esg_dataroot/cmor/CMIP6/CMIP/',
              'http://esgf-node2.cmcc.it/thredds/fileServer/esg_dataroot/CMIP6/CMIP/',
              'esgf.nci.org.au/thredds/fileServer/replica/CMIP6/CMIP/',
              'vesg.ipsl.upmc.fr/thredds/fileServer/cmip6/CMIP/',
              'esgf-data02.diasjp.net/thredds/fileServer/esg_dataroot/CMIP6/CMIP/']

# Basic requests to download CORDEX RCMs
RCM_REQUEST = {
    'format': 'zip',
    'temporal_resolution': '6_hours',
    'variable': [
        '10m_u_component_of_the_wind', '10m_v_component_of_the_wind',
    ],
    'ensemble_member': 'r1i1p1',
    'experiment': 'historical',
    'horizontal_resolution': '0_11_degree_x_0_11_degree',
    'domain': 'europe'
}

# List of RCM core models
REGIONAL_MODEL_NAMES = {
    'SMHI': 'smhi_rca4',
    'MOHC': 'mohc_hadrem3_ga7_05',
    'ETH': 'clmcom_eth_cosmo_crclim',
    'DMI': 'dmi_hirham5',
    'CNRM': 'cnrm_aladin63',
    'ICTP': 'ictp_regcm4_6'
}

# List of GCM boundary models of RCMs
GLOBAL_BOUNDARY_MODEL_NAMES = {
    'MPI': 'mpi_m_mpi_esm_lr',
    'EC-EARTH': 'ichec_ec_earth',
    'IPSL': 'ipsl_cm5a_mr',
    'MOHC': 'mohc_hadgem2_es',
    'NCC': 'ncc_noresm1_m',
    'CNRM': 'cnrm_cerfacs_cm5'
}

# Basic request to download ERA5 data
ERA5_REQUEST = {
    'product_type': 'reanalysis',
    'variable': [
        '10m_u_component_of_wind', '10m_v_component_of_wind',
    ],
    'year':  [f"{year}" for year in range(START_YEAR, END_YEAR)],
    'month': [
        '01', '02', '03',
        '04', '05', '06',
        '07', '08', '09',
        '10', '11', '12',
    ],
    'day': [
        '01', '02', '03',
        '04', '05', '06',
        '07', '08', '09',
        '10', '11', '12',
        '13', '14', '15',
        '16', '17', '18',
        '19', '20', '21',
        '22', '23', '24',
        '25', '26', '27',
        '28', '29', '30',
        '31',
    ],
    'format': 'grib',
}

# List of time points for ERA5 download
TIME_POINTS = {
    'start0am': ['00:00', '06:00', '12:00', '18:00',],
    'start3am': ['03:00', '09:00', '15:00', '21:00']
}


def download_global_data():
    """
        Downloads individual CMIP6 GCM data files for each model in `global_climate_data`,
        using the specified file templates and year ranges defined in `global_climate_data`.
    """
    # Loop through each model and its file details
    for model_name, file_information in GCM_FILES.items():
        filename, years = file_information  # Unpack filename template and list of year ranges
        download_dir = os.path.join(ORIGINAL_DATA_DIR, GLOBAL_MODEL_NAME, model_name)
        os.makedirs(download_dir, exist_ok=True)  # Create model directory if it doesn't exist

        # Iterate over each specified year range
        for year_range in years:
            # Loop for three separate model runs
            for run in range(1, 4):
                # Generate file URL by replacing placeholders with actual values
                file_url = (filename.replace('$RUN$', str(run))
                            .replace('$VAR$', SURFACE_WIND_SPEED)
                            .replace('$YEARS$', str(year_range)))

                # Determine if the file already exists in the directory
                file_path, file_exists = create_file_path_and_check_if_file_already_exists(file_url, download_dir)
                if not file_exists:
                    # Attempt to download the file if it doesn't exist
                    download_success = download_file_from_cmip_server(file_url, file_path)
                    if not download_success:  # If download fails, try alternate variables
                        for variable in [U_COMPONENT_CMIP6, V_COMPONENT_CMIP6]:
                            # Generate file URL with an alternate variable
                            file_url = (filename.replace('$RUN$', str(run))
                                        .replace('$VAR$', variable)
                                        .replace('$YEARS$', str(year_range)))

                            # Check if alternate variable file already exists
                            file_path, file_exists = create_file_path_and_check_if_file_already_exists(file_url,
                                                                                                       download_dir)
                            if not file_exists:
                                # Attempt to download the alternate variable file
                                download_success = download_file_from_cmip_server(file_url, file_path)
                                if not download_success:
                                    # Log an error if download ultimately fails
                                    print('Failed to download: ' + filename.replace('$RUN$', str(run))
                                          .replace('$YEARS$', str(year_range)))
    return


def download_file_from_cmip_server(file_url, download_path):
    """
        Attempts to download a file from multiple CMIP servers and saves it locally if successful.

        Arguments:
            file_url {str}: URL of the file to be downloaded.
            download_path {string}: Local path to save the downloaded file.

        Returns:
            download_success {boolean}: True if the download was successful.
    """

    download_success = False
    i = 0  # Server index for retrying with different servers

    # Loop through the list of severs
    while not download_success and i < len(SERVER_LIST):
        url = f'https://{SERVER_LIST[i]}{file_url}'  # Full URL to attempt download
        try:
            # Request file from the server
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for unsuccessful status codes
            with open(download_path, 'wb') as file:
                file.write(response.content)  # Write file content to local path
            print(f"Downloaded: {download_path.split('/')[-1]}")  # Print success message
            download_success = True
        except requests.exceptions.RequestException:
            i += 1  # Move to the next server if there's an error

    return download_success


def create_file_path_and_check_if_file_already_exists(file_url, download_dir):
    """
        Checks if a file with the same name as in `file_url` already exists in the specified `download_dir`.

        Arguments:
            file_url {str}: URL of the file to check.
            download_dir {string}: Directory where the file would be saved.

        Returns:
            file_path {str}: The full path where the file would be stored.
            file_exists {boolean}: True if the file already exists at `file_path`.
    """
    file_name = file_url.split('/')[-1]  # Extract the file name from URL
    file_path = os.path.join(download_dir, file_name)  # Build the full local file path
    file_exists = os.path.isfile(file_path)  # Check if the file already exists locally
    if file_exists:
        print(f"Already exists: '{file_path}'")  # Log that the file is already present
    return file_path, file_exists  # Return path and existence flag


def download_ERA5_data():
    """
        Downloads ERA5 climate data for a specified continent, storing each file in the designated directory.
        The function creates a request for each time period in `TIME_POINTS`, covering the full geographic
        area of the continent, and saves the resulting files as GRIB files.
    """
    continent = Country(COUNTRY).continent  # Obtain continent data from specified country
    download_dir = os.path.join(ORIGINAL_DATA_DIR, ERA5_NAME)
    os.makedirs(download_dir, exist_ok=True)  # Create the ERA5 data directory if it doesn’t exist

    # Iterate over each time period and its list of time points
    for time_title, time_points_list in TIME_POINTS.items():
        # Set file path for the current time period
        file_path = os.path.join(download_dir, f'{ERA5_NAME}_{START_YEAR}-{END_YEAR}_{time_title}.grib')

        if not os.path.exists(file_path):  # Only download if file does not already exist
            new_request = ERA5_REQUEST  # Base request template for ERA5 data
            new_request['time'] = time_points_list  # Set time points for this period
            # Set geographic area to the continent's bounding box
            new_request['area'] = [
                continent.max_lat, continent.min_lon, continent.min_lat, continent.max_lon,
            ]
            # Submit the download request to the climate data retrieval service
            c.retrieve('reanalysis-era5-single-levels', new_request, file_path)
    return


def download_regional_data():
    """
        Downloads regional climate projection data for various model combinations across a set of years.
        For each combination of global and regional climate models (GCM and RCM) and each year from 1995
        to 2004, this function submits a request for climate projection data, downloads the file, and
        extracts its contents to a dedicated folder specified in the settings.
    """
    # Iterate over each Global Climate Model (GCM) and Regional Climate Model (RCM) combination
    for gcm_short, gcm_long in GLOBAL_BOUNDARY_MODEL_NAMES.items():
        for rcm_short, rcm_long in REGIONAL_MODEL_NAMES.items():
            # Loop through each year in the range 1995-2004
            for year in range(1995, 2005):
                year_str = str(year)
                # Define directory and file paths for the current model-year combination
                download_dir = os.path.join(ORIGINAL_DATA_DIR, REGIONAL_MODEL_NAME, f'{gcm_short}-{rcm_short}')
                os.makedirs(download_dir, exist_ok=True)  # Create directory if it doesn’t exist

                file_path = os.path.join(download_dir, f'{gcm_short}-{rcm_short}_{year_str}.zip')

                # Check if the data file already exists; if not, submit a new request
                if not os.path.exists(file_path):
                    new_request = RCM_REQUEST  # Base request template for RCM data
                    new_request['gcm_model'] = gcm_long  # Set the GCM model name
                    new_request['rcm_model'] = rcm_long  # Set the RCM model name
                    new_request['start_year'] = year  # Set the starting year

                    try:
                        new_request['end_year'] = year  # Attempt to set the same end year
                        # Submit download request and save the file as ZIP
                        c.retrieve('projections-cordex-domains-single-levels', new_request, target=file_path)
                    except Exception:
                        new_request['end_year'] = str(year + 1)  # Adjust end year in case of error
                        # Retry download with the adjusted end year
                        c.retrieve('projections-cordex-domains-single-levels', new_request, target=file_path)

                # Extract ZIP file contents if download was successful
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    extract_path = os.path.join(download_dir, f'{gcm_short}-{rcm_short}')  # Define extraction folder
                    zip_ref.extractall(extract_path)  # Extract contents
                    print(f'Extracted {file_path} to {extract_path}')  # Log extraction completion
    return