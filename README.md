This project compares wind speed data from CMIP6 and EUR-11-CORDEX with the reanalysis ERA5. 

The repository contains the pre-processing, analysis and plotting procedures for
- the published paper: Sofia Morelli, Nina Effenberger, Luca Schmidt, and Nicole Ludwig (2025). Climate data selection for multi-decadal wind power forecasts. https://doi.org/10.1088/1748-9326/adc01f.
- the Master's Thesis: Sofia Morelli (2024). The impact of the spatial resolution of wind data on multi-decadal wind power forecasts in Germany. https://arxiv.org/abs/2410.14681.

The figures in Morelli at al. (2024) can be reproduced with the jupyter notebooks in plots using the 
already pre-processed data of Europe (expect for the wind speeds needed for demonstration plot 1 since the files are too large) saved in plotting/data_pre-processed.

For re-running the full analysis with adaptable region, the following variables need to be set in settings.py:
- BASE_DIR: local base directory
- CDSAPI_KEY: personal UID and API Key from the Copernicus website, to create an account visit https://cds.climate.copernicus.eu/#!/home

The data can then be downloaded from Copernicus and a list of CMIP6 servers by calling the download tasks in 
scr/pre_processing/main_save_wind_speeds.py.
We used 6-hourly wind speed values as suggested by Effenberger et al. (2023) (https://doi.org/10.1088/1748-9326/ad0bd6) 
from historical runs.
If avaiable, we downloaded files containing the surface wind speed, otherwise we calculated the wind speed from the 
orthogonal u and v components at 10 meters above the surface.

Next, the data need to be pre-processed by calling the according tasks in scr/pre_processing/main_save_wind_speeds.py, 
which makes use of the CDO library (https://code.mpimet.mpg.de/projects/cdo).
Files will be created containing the wind speed values between 2005 and 2015 on a regular grid with original spatial 
resolution constrained to the region of Europe and Germany in separate folders.

Additional files containing the wind speeds at the turbine locations in Germany can be created from the Europe files.
In this case, the wind power dataset from https://www.mdpi.com/2306-5729/4/1/29/htm must be downloaded 
and saved in TURBINE_LOCATIONS_DIR (see settings.py).

Please be aware that all these tasks require considerably large storage space!

The analysis can be reproduced by running the tasks in scr/analysis/main_make_plots.py. This involves calculating the 
wind power via the windpowerlib library (https://github.com/wind-python/windpowerlib).
