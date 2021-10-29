import xarray as xr
import numpy as np
import pandas as pd
import os
from statsmodels.formula.api import glm
import statsmodels.formula.api as smf
import statsmodels.api as sm
import itertools
import datetime
from dask.distributed import Client, LocalCluster
from dask.distributed import Client,Scheduler
from dask_jobqueue import SLURMCluster
import time
import glob
from statsmodels.tools.sm_exceptions import PerfectSeparationError

def create_filepath_oneTime(ds, prefix='filename', root_path="."):
    """
    Generate a filepath when given an xarray dataset
    """
    time_str = ds.time.dt.strftime("%Y-%m-%d").data
    filepath = f'{root_path}/{prefix}_{time_str}.nc'
    return filepath


# define a function to fit the GLM model
def fit_logistReg_2Pred_oneThres(y, x1, x2, predictors, thres, formula, x1_new, x2_new, n_predictors = 3):
    '''Function to fit a logistic regression model to estimate exceedence probability
    using a thres argument. If the thres argument is nan, the grid is not in drought.
    '''
    GLM_params = np.empty(n_predictors)
    GLM_pvalues = np.empty(n_predictors)
    GLM_probability = np.empty(1)
    GLM_aic = np.empty(1)

    if np.isnan(thres):
        GLM_probability[:] = np.nan
        GLM_params[:] = np.nan
        GLM_pvalues[:] = np.nan
        GLM_aic[:] = np.nan
    else:
        y_binary = np.where(y >= thres, 1, 0)      
        if (sum(y_binary) < 4):                      # too few data points for estimation
            GLM_probability[:] = 0
            GLM_params[:] = 0
            GLM_pvalues[:] = np.nan
            GLM_aic[:] = np.nan
        else:                                        # logistic regression fit
            # create a dataframe of reponse and predictors
            x_dict = {predictors[0]:x1, predictors[1]:x2}
            x = pd.DataFrame(x_dict)
            x['response'] = y_binary

            x_new_dict = {predictors[0]:x1_new, predictors[1]:x2_new}
            x_new = pd.DataFrame(x_new_dict, index = [0])

            model = glm(formula, x, family=sm.families.Binomial())
            try:
                model_GLM = model.fit()
                GLM_probability[:] = model_GLM.predict(x_new)
                GLM_params[:] = model_GLM.params.values
                GLM_pvalues[:] = model_GLM.pvalues.values
                GLM_aic[:] = model_GLM.aic
            except PerfectSeparationError:          # this error occurs at longer timescales with fewer data points
                GLM_probability[:] = np.nan
                GLM_params[:] = np.nan
                GLM_pvalues[:] = np.nan
                GLM_aic[:] = np.nan
    return GLM_params, GLM_pvalues, GLM_probability, GLM_aic

if __name__ == '__main__':

    main_dir = '/g/data/w97/ad9701/p_prob_analysis/temp_files/'

    varname = 'PminusEQ' #'P'   # the name of the directory and file
    vname = 'PminusEQ'   #'precip'  # the name of the variable inside the files
    fname = varname + '_*_*_*.nc'

    iW = int(os.environ['iWeek'])
    print(iW)
    start_yr = int(os.environ['start_yr'])
    end_yr = start_yr #int(os.environ['end_yr'])

    # select thresholds
    # load the threshold data file & select the drought period of interest
    PmEQ_events_file = 'sm_droughts/PmEQ_events_*.nc'
    ds_thresh = xr.open_mfdataset(main_dir + PmEQ_events_file)
    drght_time_slice = slice(str(start_yr)+'-01-01', str(end_yr)+'-12-31')
    drght_name = 'full_record'
    drght_dir = 'GLM_results_' + drght_name

    # select the thresholds for the time periods of the drought
    thresName = 'PminusEQ'
    da_thresh_temp = ds_thresh[thresName].sel(time = drght_time_slice)
    
    # select only a sample of time for calculation - changed this from every 2 weeks to every 4 weeks due to run time
    time_sel = da_thresh_temp.time[0::28]
    da_thresh = da_thresh_temp.sel(time = time_sel)

    ############################################
    # GET THE SST PREDICTORS
    ############################################

    # get the sst data
    sst_dir = '/g/data/w97/ad9701/p_prob_analysis/sst_data/'
    pNames = ['soi', 'sami', 'dmi', 'nino34_anom', 'nino4_anom']
    pFiles = ['soi_monthly.nc', 'newsam.1957.2021.nc', 'dmi.had.long.data.nc', 'nino34.long.anom.data.nc', 'nino4.long.anom.data.nc']
    for p in np.arange(len(pNames)):
        ds_temp = xr.open_dataset(sst_dir+pFiles[p])
        if (p>0):
            ds_p[pNames[p]]=ds_temp[pNames[p]]
        else:
            ds_p = ds_temp
        del ds_temp

    # select the predictors to include in the model
    predSel = ['soi', 'dmi']
    formula = 'response ~ soi + dmi'
    parameter = ['Intercept']
    parameter.extend(predSel)

    # select the sst predictors corresponding to the dates of the thresholds data
    thresh_time_bymon = np.array(pd.to_datetime(da_thresh.time).to_period('M').to_timestamp().floor('D'))
    da_p1_current = ds_p[predSel[0]].rename({'time': 'time'}).sel(time = thresh_time_bymon)
    da_p2_current = ds_p[predSel[1]].rename({'time': 'time'}).sel(time = thresh_time_bymon)

    ############################################
    # START A LOCAL CLUSTER
    ############################################

    # from dask.distributed import Client, LocalCluster
    # cluster = LocalCluster()
    # client = Client(cluster)
    # client

    ############################################
    # PERFORM CALCULATIONS FOR EACH SET OF DATA
    ############################################
    
    nSets = (7*iW)-1    # number of sets in addition to the original aggregation
    
    for i in range(nSets):
        sub_dir = '/set' + str(i+2)
        
        # get data
        data_dir = main_dir + varname + '_week' + str(iW) + '/' + sub_dir + '/'
        print(data_dir)
        ds = xr.open_mfdataset(data_dir + fname, chunks = {'lat':400, 'lon':400})
        da_var_temp = ds[vname].reindex(lat=ds.lat[::-1]).chunk(chunks = {'lat':40,'lon':40,'time':-1}).rename({'time':'hist_time'})
        da_var = da_var_temp.groupby('hist_time.season')

        # select predictors for the same time points as the P-E or P-E-Q data at multi-weekly timescale
        da_time_bymon = np.array(pd.to_datetime(ds.time).to_period('M').to_timestamp().floor('D'))
        ds_p_sel = ds_p.sel(time = da_time_bymon)
        ds_p1_sel_gb = ds_p_sel[predSel[0]].rename({'time':'hist_time'}).groupby('hist_time.season')
        ds_p2_sel_gb = ds_p_sel[predSel[1]].rename({'time':'hist_time'}).groupby('hist_time.season')

        full_dir_path = main_dir + '/' + drght_dir + '/' + varname + '_week' + str(iW) + '/' + sub_dir + '/by_day/'
        if not os.path.exists(full_dir_path):
            os.makedirs(full_dir_path)

        # check for existing half completed files
        # glm_files = glob.glob(full_dir_path + 'GLM_results_*' + str(start_yr) + '-*.nc')
        # nfiles = len(glm_files)
        # if nfiles < 365:   # the required nfiles may be 365 or 366. taking a chance with using 365 here
        #     if nfiles > 4:
        #         start_day = nfiles - 4
        #     else:
        #         start_day = 0
        
        start_day = 0
        dask_gufunc_kwargs = {'output_sizes':{"glm_parameter": len(parameter)}} #, 'time':1}}

        # looping over the current times
        for i_time in range(start_day, len(da_thresh.time)):
            seas = da_thresh['time.season'].values[i_time]
            da_logistReg = xr.apply_ufunc(
                fit_logistReg_2Pred_oneThres,                # first the function, this function returns a tuple (GLM params, GLM pvalues, GLM modelled probabilities)
                da_var[seas],                                # function arg
                ds_p1_sel_gb[seas].values,
                ds_p2_sel_gb[seas].values,
                predSel,                                     #      "
                da_thresh.sel(timescale = iW).isel(time = i_time),                                  #      "
                formula,                                     #      "
                [da_p1_current.values[i_time]],                    #      "
                [da_p2_current.values[i_time]],                    #      "
                input_core_dims=[["hist_time"], ["hist_time"], ["hist_time"], ["predictors"], [], [], [], []], #["sample_time"], ["sample_time"]],   # list with one entry per arg, these are the dimensions not to be broadcast
                output_core_dims=[["glm_parameter"], ["glm_parameter"], [], []],                                # dimensions of the output
                vectorize=True,                                                                                                                    # broadcast over non-core dimensions of the input object?
                dask="parallelized",                                                                                                               # enable dask?
                dask_gufunc_kwargs=dask_gufunc_kwargs,                     
                output_dtypes=[float, float, float, float]
            )

            # assign co-ordinates add metadata
            new_coords_dict = {'glm_parameter':parameter} #, 'current_time':[da_thresh['current_time'][i_time]]}    
            ds_all = da_logistReg[2].rename('glm_probability').to_dataset()
            ds_all['glm_params'] = da_logistReg[0].rename('glm_params').assign_coords(new_coords_dict)
            ds_all['glm_pvalues'] = da_logistReg[1].rename('glm_pvalues').assign_coords(new_coords_dict)
            ds_all['glm_aic'] = da_logistReg[3].rename('glm_aic')
            ds_all[predSel[0]] = da_p1_current.isel(time = i_time)
            ds_all[predSel[1]] = da_p2_current.isel(time = i_time)

            out_file = create_filepath_oneTime(ds_all, prefix = 'GLM_results_' + '_'.join(predSel), root_path = full_dir_path)
            ds_all.to_netcdf(out_file)
    # else:
    #     print("Year " + str(start_yr) + " is already complete")
