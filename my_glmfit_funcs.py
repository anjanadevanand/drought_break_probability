import xarray as xr
import numpy as np
import pandas as pd
from statsmodels.formula.api import glm
import statsmodels.formula.api as smf
import statsmodels.api as sm
import itertools
import datetime
from dask.distributed import Client, LocalCluster
from dask.distributed import Client,Scheduler
from dask_jobqueue import SLURMCluster
import time
from statsmodels.tools.sm_exceptions import PerfectSeparationError
# model = glm(formula, data, family)

# define a function to fit the GLM model
def fit_logistReg_3Pred(y, x1, x2, x3, predictors, thres, formula, x1_new, x2_new, x3_new, n_predictors = 6):
    x_dict = {predictors[0]:x1, predictors[1]:x2, predictors[2]:x3}
    x = pd.DataFrame(x_dict)
    x_new_dict = {predictors[0]:x1_new, predictors[1]:x2_new, predictors[2]:x3_new}
    x_new = pd.DataFrame(x_new_dict)
    
    GLM_params = np.empty((n_predictors, len(thres))) # array([])
    GLM_pvalues = np.empty((n_predictors, len(thres)))
    GLM_probability = np.empty((len(x_new), len(thres)))
    GLM_aic = np.empty(len(thres))
    for ith in np.arange(len(thres)):
        # create a dataframe of reponse and predictors
        y_binary = np.where(y >= thres[ith], 1, 0)
        x['response'] = y_binary

        if (sum(y_binary) < 4):                      # too few data points for estimation
            GLM_probability[:,ith] = 0
            GLM_params[:,ith] = np.nan
            GLM_pvalues[:,ith] = np.nan
            GLM_aic[ith] = np.nan
        else:                                        # logistic regression fit
            model = glm(formula, x, family=sm.families.Binomial())
            model_GLM = model.fit()
            GLM_probability[:, ith] = model_GLM.predict(x_new)
            GLM_params[:, ith] = model_GLM.params.values
            GLM_pvalues[:, ith] = model_GLM.pvalues.values
            GLM_aic[ith] = model_GLM.aic
    return GLM_params, GLM_pvalues, GLM_probability, GLM_aic

# define a function to fit the GLM model
def fit_logistReg_2Pred(y, x1, x2, predictors, thres, formula, x1_new, x2_new, n_predictors = 3):
    x_dict = {predictors[0]:x1, predictors[1]:x2}
    x = pd.DataFrame(x_dict)
    x_new_dict = {predictors[0]:x1_new, predictors[1]:x2_new}
    x_new = pd.DataFrame(x_new_dict)
    
    GLM_params = np.empty((n_predictors, len(thres)))
    GLM_pvalues = np.empty((n_predictors, len(thres)))
    GLM_probability = np.empty((len(x_new), len(thres)))
    GLM_aic = np.empty(len(thres))
    
    for ith in np.arange(len(thres)):
        # create a dataframe of reponse and predictors
        y_binary = np.where(y >= thres[ith], 1, 0)
        x['response'] = y_binary

        if (sum(y_binary) < 4):                      # too few data points for estimation
            GLM_probability[:,ith] = 0
            GLM_params[:,ith] = 0
            GLM_pvalues[:,ith] = np.nan
            GLM_aic[ith] = np.nan
        else:                                        # logistic regression fit
            model = glm(formula, x, family=sm.families.Binomial())
            try:
                model_GLM = model.fit()
                GLM_probability[:, ith] = model_GLM.predict(x_new)
                GLM_params[:, ith] = model_GLM.params.values
                GLM_pvalues[:, ith] = model_GLM.pvalues.values
                GLM_aic[ith] = model_GLM.aic
            except PerfectSeparationError: 
                GLM_probability[:,ith] = np.nan
                GLM_params[:,ith] = np.nan
                GLM_pvalues[:,ith] = np.nan
                GLM_aic[ith] = np.nan

    return GLM_params, GLM_pvalues, GLM_probability, GLM_aic

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

# define a function to fit a one predictor linear regression model
def fit_lm_1Pred(y, x, dryInd, predictand, predictor):
    xy_dict = {predictand:y[dryInd], predictor:x[dryInd]}
    xy_df = pd.DataFrame(x_dict)
    formula = predictand + ' ~ ' + predictor
    lm = smf.ols(formula, xy_df)
    model_lm = lm.fit()
    lm_params = model_lm.params.values
    lm_pvalues = model_lm.pvalues.values
    lm_rsq_adj = model_lm.rsquared_adj
    return lm_params, lm_pvalues, lm_rsq_adj

# function to create a new data frame that will be used to 'predict' probabilities from the fitted model
# the new data points would cover combinations of unique values for categorical predictors and mean/perturbations one sd above the mean for quantitative predictors
def createSampleDf(df, fields):
    '''Function creates a sample dataframe from a larger input dataframe (df).
       The sample points will include all permutations of columns specfied (fields).
       String columns: use unique values. Numeric columns: Mean, Mean-1SD, Mean+1SD
    '''
    dataVal = []
    for f in fields:
        # str data types are assumed to be categorical variables
        if (isinstance(df[f][0], str)):
            dataVal.append(pd.unique(df[f]))
        else:
            temp = [df[f].mean()]
            temp.extend([df[f].mean()+df[f].std(), df[f].mean()-df[f].std()])
            dataVal.append(temp)
            del temp
    # get all combinations of values across the fields
    dataValPermute = list(itertools.product(*dataVal))
    # make it into a data frame
    newDf = pd.DataFrame(dataValPermute, columns = fields)
    return(newDf)

def get_sst_predictors(sst_dir = '/g/data/w97/ad9701/p_prob_analysis/sst_data/',
                pNames = ['soi', 'sami', 'dmi', 'nino34_anom', 'nino4_anom'],
                pFiles = ['soi_monthly.nc', 'newsam.1957.2021.nc', 'dmi.had.long.data.nc', 'nino34.long.anom.data.nc', 'nino4.long.anom.data.nc']):
    for p in np.arange(len(pNames)):
        ds_temp = xr.open_dataset(sst_dir+pFiles[p])
        if (p>0):
            ds_p[pNames[p]]=ds_temp[pNames[p]]
        else:
            ds_p = ds_temp
        del ds_temp
    return ds_p

def fit_gridded_logistReg(main_dir, varname, iWeek, threshold, ds_p, x_new, sub_dir = '',
                     predSel = ['soi', 'dmi'], 
                     formula = 'response ~ soi+dmi', 
                     parameter = ['Intercept', 'soi', 'dmi'], 
                     time_slice = slice('1911-01-01','2020-05-31')):
    
    fname = varname + '_*_*_*.nc'

    # # select the predictors to include in the model
    # extra_param.extend(predSel)
    # parameter = extra_param

    # # create a new df of sample points at which 'predictions' will be made using the fitted model
    # ds_p = get_sst_predictors()
    # print('Full set of sst predictors: ' + str(list(ds_p.keys())))
    # ds_p_subset = ds_p.sel(time = time_slice)
    # pred_dict = {}
    # for p in list(ds_p.keys()): #pNames:
    #     pred_dict.update({p: ds_p_subset[p].values})
    # pred_dict.update({"season": ds_p_subset['time.season'].values})    # add season to the sst predictors    
    # pred_df = pd.DataFrame(pred_dict, index = ds_p_subset['time'])     # make a dataframe of predictors
    # print('Selected sst predictors: ' + str(predSel))
    # pred_df_sel = pred_df[predSel]
    # x_new = createSampleDf(pred_df_sel, list(pred_df_sel.keys()))

    # get data
    data_dir = main_dir + varname + '_week' + str(iWeek) + '/' + sub_dir + '/'
    ds = xr.open_mfdataset(data_dir + fname, chunks = {'lat':400, 'lon':400}, combine='nested', concat_dim='time')
    da_var = ds[varname].sel(time = time_slice).chunk(chunks = {'lat':40,'lon':40,'time':-1}).groupby('time.season')
    # da_var = ds[varname].sel(time = time_slice).groupby('time.season')
    
    # select predictors for the same time points as the P-E or P-E-Q data
    da_time_bymon = np.array(pd.to_datetime(ds.time).to_period('M').to_timestamp().floor('D'))
    ds_p_sel = ds_p.sel(time = da_time_bymon)
    ds_p1_sel_gb = ds_p_sel[predSel[0]].groupby('time.season')
    ds_p2_sel_gb = ds_p_sel[predSel[1]].groupby('time.season')
    
    da_params_list = []
    da_pvalues_list = []
    da_prob_list = []
    da_aic_list = []
    seas_names = ['DJF', 'MAM', 'JJA', 'SON']
    dask_gufunc_kwargs = {'output_sizes':{"parameter": len(parameter), "sample":len(x_new)}} #, 'allow_rechunk':True}
    for seas in seas_names:    
        da_logistReg = xr.apply_ufunc(
            fit_logistReg_2Pred,                      # first the function, this function returns a tuple (GLM params, GLM pvalues, GLM modelled probabilities)
            da_var[seas],                                # function arg
            ds_p1_sel_gb[seas].values,
            ds_p2_sel_gb[seas].values,
            predSel,                                     #      "
            threshold,                                  #      "
            formula,                                     #      "
            x_new[predSel[0]].values,                    #      "
            x_new[predSel[1]].values,                    #      "
            input_core_dims=[["time"], ["time"], ["time"], ["predictors"], ["threshold"], [], ["sample"], ["sample"]],   # list with one entry per arg, these are the dimensions not to be broadcast
            output_core_dims=[["parameter", "threshold"], ["parameter", "threshold"], ["sample", "threshold"], ["threshold"]],                                # dimensions of the output
            vectorize=True,                                                                                                                    # broadcast over non-core dimensions of the input object?
            dask="parallelized",                                                                                                               # enable dask?
            dask_gufunc_kwargs=dask_gufunc_kwargs,                     
            output_dtypes=[float, float, float, float]
        )

        # assign co-ordinates add metadata
        new_coords_dict = {'threshold':threshold, 'parameter':parameter}
        da_params_list.append(da_logistReg[0].rename('glm_params').assign_coords(new_coords_dict))
        da_pvalues_list.append(da_logistReg[1].rename('glm_pvalues').assign_coords(new_coords_dict))
        da_aic_list.append(da_logistReg[3].rename('glm_aic').assign_coords({'threshold':threshold}))
        da_prob_temp = da_logistReg[2].rename('glm_probability').assign_coords({'threshold':threshold}).to_dataset()
        for k in list(x_new.keys()):
            da_prob_temp[k] = x_new[k]
        da_prob_temp = da_prob_temp.rename({'dim_0':'sample'})   
        da_prob_list.append(da_prob_temp)

    ds_all = xr.concat(da_prob_list, dim = 'season').assign_coords({'season':seas_names})
    ds_all['glm_params'] = xr.concat(da_params_list, dim = 'season').assign_coords({'season':seas_names})
    ds_all['glm_pvalues'] = xr.concat(da_pvalues_list, dim = 'season').assign_coords({'season':seas_names})
    ds_all['glm_aic'] = xr.concat(da_aic_list, dim = 'season').assign_coords({'season':seas_names})

    return(ds_all)
    
    # # # save file
    # out_file = data_dir + 'GLM_results_' + '_'.join(predSel) + '_bySeason.nc'
    # print(out_file)
    # a = datetime.datetime.now()
    # # cluster = SLURMCluster(cores=8,memory="15GB",walltime='01:30:00')
    # # client = Client(cluster)
    # # cluster.scale(cores = 8)
    # # # cluster.scale(cores=16)
    # # # cluster = LocalCluster()
    # # # client = Client(cluster)
    # # # with LocalCluster() as cluster, Client(cluster) as client:
    # # time.sleep(20)
    # ds_all.to_netcdf(out_file)
    # # # client.loop.add_callback(client.scheduler.retire_workers, close_workers=True)
    # # # client.loop.add_callback(client.scheduler.terminate)
    # # # client.run_on_scheduler(lambda dask_scheduler: dask_scheduler.loop.stop())
    # b = datetime.datetime.now()
    # print('Time taken to write the file = ' + str(b-a))
    # # cluster.scale(cores = 0)
    # # return None
    
if __name__ == "main":

    main_dir = '/g/data/w97/ad9701/p_prob_analysis/temp_files/'
    varname = 'PminusEQ'

    # select thresholds
    threshold = [20, 50, 100] 
    
    # select timescales for analysis
    iWeek = [2, 6, 12] #[2, 6, 8, 12]