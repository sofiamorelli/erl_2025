import os

CDSAPI_URL = 'https://cds.climate.copernicus.eu/api/v2'
CDSAPI_KEY = ''# Personal identification on copernicus website $UID$:$API_Key$

START_YEAR = 2005
END_YEAR = 2015

BASE_DIR_DAT = f'dat/{START_YEAR}-{END_YEAR}/'
ORIGINAL_DATA_DIR = os.path.join(BASE_DIR_DAT, 'original_data_sets')
TURBINE_LOCATIONS_DIR = os.path.join(BASE_DIR_DAT, 'turbine_locations')
BASE_DIR_FIG = F'fig/{START_YEAR}-{END_YEAR}/'

WS_MAX = 38

COUNTRY = 'Germany'
LATITUDE_NAME = 'lat'
LONGITUDE_NAME = 'lon'
COORDINATE_NAME = f'{LATITUDE_NAME}_{LONGITUDE_NAME}'
U_COMPONENT_CMIP6 = 'uas'
V_COMPONENT_CMIP6 = 'vas'
SURFACE_WIND_SPEED = 'sfcWind'

WIND_SPEED_LABEL = 'Wind speed in m/s'
WIND_POWER_LABEL = 'Wind power in MW'

ERA5_NAME = 'ERA5'
ERA5_FILE_NAME = f'{SURFACE_WIND_SPEED}_Reanalysis_{ERA5_NAME}_hintcast_$REGION$_{START_YEAR}-{END_YEAR}_$TIME$.nc'

GLOBAL_MODEL_NAME = 'Global'
FILE_NAME = f'{SURFACE_WIND_SPEED}$SIZE$_$CATEGORY$_$MODEL$_r$RUN$_$REGION$_{START_YEAR}-{END_YEAR}.nc'
MPI_HR = 'MPI-HR'
MPI_LR = 'MPI-LR'
JAP = 'JAP'
IPSL = 'IPSL'
CMCC = 'CMCC'
NCC_HR = 'NCC-HR'
NCC_LR = 'NCC-LR'
MOHC_HR = 'MOHC-HR'
MOHC_LR = 'MOHC-LR'
EC_EARTH = 'EC-EARTH'

MODELS_WITH_SEVERAL_RUNS = [MPI_HR, MPI_LR, JAP, CMCC, NCC_HR, NCC_LR]

REGIONAL_MODEL_NAME = 'Regional'
MPI = 'MPI'
CONSTANT_GLOBAL_BOUNDARY_MODEL = MPI
NCC = 'NCC'
MOHC = 'MOHC'
CNRM = 'CNRM'
SMHI = 'SMHI'
CONSTANT_REGIONAL_MODEL = SMHI
DMI = 'DMI'
ETH = 'ETH'
ICTP = 'ICTP'

COLORS = {
    MPI_HR: 'crimson',
    MPI_LR: 'deeppink',
    MOHC_HR: 'blue',
    MOHC_LR: 'deepskyblue',
    NCC_HR: 'green',
    NCC_LR: 'lime',
    CMCC: 'sienna',
    JAP: 'darkgrey',
    IPSL: 'orange',
    EC_EARTH: 'violet',
    ERA5_NAME: 'black',
    'MPI-LR-CMIP6': 'deeppink',
    'MPI-LR-CMIP5': 'pink'
}

COLORS_RCMs = {
    MPI: 'darkred',
    IPSL: 'darkorange',
    EC_EARTH: 'darkviolet',
    NCC: 'darkgreen',
    MOHC: 'darkblue',
    CNRM: 'gold'
}

LINE_STYLES_RCMs = {
    SMHI: '-',
    DMI: '--',
    ETH: '-.',
    MOHC: ':',
    CNRM: (0, (3, 5, 1, 5, 1, 5)),
    ICTP: (5, (10, 3))
}

LINE_STYLES_FOR_RUN_COMPARISON = {
    'r1': 'dashed',
    'r2': 'dashdot',
    'r3': 'dotted',
    'average': 'solid',
}
