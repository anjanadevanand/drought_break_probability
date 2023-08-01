# drought_break_probability

Jupyter notebooks used to analyse probability of drought breaking events on Gadi

This repository contains work in progress.

### Code workflow

1. Calculation of soil moisture percentiles (soil_moisture_perc_calc.py, job_sm_perc_calc/, soil_moisture_perc_concat.ipynb)

Use the soil moisture data from the corresponding NCI directory  & calculate the SM percentiles for each day of the year (doy). Concatenate the calculations for each doy. Concatenate the results for each doy in a single file.

AWRA:
input: /g/data/fj8/BoM/AWRA/DATA/SCHEDULED-V6/processed/values/day/sm_[1-2]\*.nc
output: /g/data/w97/ad9701/p_prob_analysis/processed_data/awra/sm_refPeriod_perc/by_doy/sm_perc_\*day\*.nc
        /g/data/w97/ad9701/p_prob_analysis/processed_data/awra/sm_refPeriod_perc/sm_percentiles.nc

2. Identify grid level droughts (identify_gridlevel_drought_events.ipynb)
    
Use the soil moisture data and the climatological percentiles calculated from the same (saved in directory 'sm_refPeriod_perc' in step 1) to identify times of drought at all grid points of interest. The script generates files named 'events_\*.nc' for each year that contain binary information, with '1' indicating that the grid is in drought.

AWRA:
inputs: /g/data/fj8/BoM/AWRA/DATA/SCHEDULED-V6/processed/values/day/sm_[1-2]\*.nc
        /g/data/w97/ad9701/p_prob_analysis/processed_data/awra/sm_refPeriod_perc/sm_percentiles.nc
output: /g/data/w97/ad9701/p_prob_analysis/processed_data/awra/sm_droughts/events\*.nc


3. Calculate soil moisture deficit (drought_events_calc_smDeficit.ipynb)

Calculates SM deficits based on the future day corresponding to the x-week timescale of interest at each day in drought that is identified in the previous step. These values are the threshold values of moisture accumulations that can end the ongoing drought. The SM deficits and P-E-Q thresholds may also be negative in certain conditions, especially in SON where the negative values indicate a thershold on the loss of existing soil moisture going into the warmer months.


4. Calculate soil moisture changes at x-week timescales (process_data_temp.ipynb)

The step calculates and saves the deltaSM data that will be used an input to the GLM model to estimate the historical probabilities of exceedence of SM change thresholds (currently x-week = 4, 8, and 12 weeks)
AWRA:
inputs:/g/data/fj8/BoM/AWRA/DATA/SCHEDULED-V6/processed/values/day/sm_[1-2]\*.nc
output:/g/data/w97/ad9701/p_prob_analysis/processed_data/awra/sm_week\*/sm_diff_week\*.nc

5. The GLM Climate model predictors

The climate mode data downloaded from corresponding centres are saved as netcdf files in this directory: /g/data/w97/ad9701/p_prob_analysis/sst_data/

6. The GLM Model (fit_logiReg_gridded_varyThresh_model4_sm_drought_parallel.py)

Script used to fit the GLM model and estimate the historical probabilities as jobs on Gadi

Bash script used to submit the jobs: sm_submit_jobs_glmfit_concatdaily.sh
