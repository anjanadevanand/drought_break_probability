import xarray as xr
import numpy as np
import pandas as pd
import os
import datetime
from datetime import datetime, timedelta
import time
import glob

# This job will be dependent upon the glm fit job
if __name__ == '__main__':
    # spatial subset
    start_lat = float(os.environ['start_lat'])
    end_lat = float(os.environ['end_lat'])
    
    start_lon = float(os.environ['start_lon'])
    end_lon = float(os.environ['end_lon'])
    
    sub_dir = 'lat' + str(start_lat) + '_' + str(end_lat) + '_lon' + str(start_lon) + '_' + str(end_lon)
    print(sub_dir)

    varname = 'sm' #'P'   # the name of the directory
    vname = 'sm_diff'   #'precip'  # the name of the files and variable inside the files
    fname = vname + '_*_*_*.nc'

    iW = int(os.environ['iWeek'])
    print(iW)
    start_yr = int(os.environ['start_yr'])
    end_yr = int(os.environ['end_yr'])
    
    drght_dir = os.environ['glm_dir'] #specify the full path here 
    full_dir_path = drght_dir + '/' + varname + '_week' + str(iW) + '/' + sub_dir + '/by_day/'
    
    for year in range(start_yr, end_yr+1):
        # checking if calculation for all dates have been completed
        glm_files = glob.glob(full_dir_path + 'GLM_results_*' + str(year) + '-*.nc')
        nfiles = len(glm_files)

        t = np.arange(datetime(year,1,1), datetime(year+1,1,1), timedelta(days=1)).astype(datetime)
        ndates = len(t)
        diff = ndates - nfiles
        if diff==0:
            print(full_dir_path + 'GLM_results_*' + str(year) + '*.nc')
            ds_glm = xr.open_mfdataset(full_dir_path + 'GLM_results_*' + str(year) + '*.nc', combine = "nested", concat_dim = "time")
            for var in ds_glm.data_vars:
                ds_glm[var].encoding['zlib'] = True
                ds_glm[var].encoding['complevel'] = 1
                del(ds_glm[var].encoding['contiguous'])
                del(ds_glm[var].encoding['chunksizes'])
 
            # save ds_glm
            out_file = drght_dir + '/' + varname + '_week' + str(iW) + '/' + 'GLM_results_' + sub_dir + '_' + str(year) + '.nc'
            print(out_file)
            ds_glm.to_netcdf(out_file)

            # delete the daily files 
            fileList = glob.glob(full_dir_path + 'GLM_results_*' + str(year) + '*.nc')
            print(fileList[0])
            for f in fileList:
                os.remove(f)
