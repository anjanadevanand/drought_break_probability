import xarray as xr
import numpy as np
import pandas as pd
from statsmodels.formula.api import glm
import statsmodels.api as sm
import itertools
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
    
    for ith in np.arange(len(thres)):
        # create a dataframe of reponse and predictors
        y_binary = np.where(y >= thres[ith], 1, 0)
        x['response'] = y_binary

        if (sum(y_binary) < 4):                      # too few data points for estimation
            GLM_probability[:,ith] = 0
            GLM_params[:,ith] = np.nan
            GLM_pvalues[:,ith] = np.nan   
        else:                                        # logistic regression fit
            model = glm(formula, x, family=sm.families.Binomial())
            model_GLM = model.fit()
            GLM_probability[:, ith] = model_GLM.predict(x_new)
            GLM_params[:, ith] = model_GLM.params.values
            GLM_pvalues[:, ith] = model_GLM.pvalues.values
    return GLM_params, GLM_pvalues, GLM_probability

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

if __name__ == "__main__":
    a = 1
    
#     sst_dir = '/g/data/w97/ad9701/p_prob_analysis/sst_data/'
#     pNames = ['soi', 'sami', 'dmi', 'nino34_anom', 'nino4_anom']
#     pFiles = ['soi_monthly.nc', 'newsam.1957.2021.nc', 'dmi.had.long.data.nc', 'nino34.long.anom.data.nc', 'nino4.long.anom.data.nc']
#     for p in np.arange(len(pNames)):
#         ds_temp = xr.open_dataset(sst_dir+pFiles[p])
#         if (p>0):
#             ds_p[pNames[p]]=ds_temp[pNames[p]]
#         else:
#             ds_p = ds_temp
#         del ds_temp

#     # select the predictors to include in the model
#     predSel = ['season', 'soi', 'dmi']
#     formula = 'response ~ C(season)+soi+dmi'

#     # create a new df of sample points at which 'predictions' will be made using the fitted model
#     time_slice = slice('1911-01-01','2020-05-31')
#     ds_p_subset = ds_p.sel(time = time_slice)
#     pred_dict = {}
#     for p in pNames:
#         pred_dict.update({p: ds_p_subset[p].values})
#     pred_dict.update({"season": ds_p_subset['time.season'].values})    # add season to the sst predictors    
#     pred_df = pd.DataFrame(pred_dict, index = ds_p_subset['time'])     # make a dataframe of predictors
#     pred_df_sel = pred_df[predSel]
#     x_new = createSampleDf(pred_df_sel, list(pred_df_sel.keys()))

#     varname = 'PminusE'
#     iW = 2
    
#     data_dir = '/g/data/w97/ad9701/p_prob_analysis/temp_files/'+varname+'_week'+str(iW)+'/'
#     fname = varname + '_*_SEA_*.nc'
    
#     ds = xr.open_mfdataset(data_dir + fname, chunks = {'lat':400, 'lon':400})
#     lat_slice = slice(-36.3, -36.2) #tiny slice for testing
#     lon_slice = slice(148.9, 149)
#     ds_subset = ds[vname].sel(lat = lat_slice, lon = lon_slice).chunk(chunks = {'lat':40,'lon':40,'time':-1})
    
#     dask_gufunc_kwargs = {'output_sizes':{"parameter": 6, "sample":len(x_new)}}

#     da_logistReg = xr.apply_ufunc(
#         fit_logistReg_3Pred,                             # first the function
#         ds_subset,                                 # function arg
#         [str(i) for i in x_df[predSel[0]].values],
#         x_df[predSel[1]].values,
#         x_df[predSel[2]].values,
#         predSel,
#         threshList,
#         formula,
#         [str(i) for i in x_new[predSel[0]].values],
#         x_new[predSel[1]].values,
#         x_new[predSel[2]].values,
#         input_core_dims=[["time"], ["time"], ["time"], ["time"], ["predictors"], ["threshold"], [], ["sample"], ["sample"], ["sample"]],  # list with one entry per arg
#         output_core_dims=[["parameter", "threshold"], ["parameter", "threshold"], ["sample", "threshold"]],           # dimensions of the output
#         vectorize=True,                             # broadcast over non-core dimensions of the input object?
#         dask="parallelized",                        # enable dask?
#         dask_gufunc_kwargs=dask_gufunc_kwargs,
#         output_dtypes=[float, float, float]
#     )