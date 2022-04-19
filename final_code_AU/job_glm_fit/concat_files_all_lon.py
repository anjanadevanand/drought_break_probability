import xarray as xr
import numpy as np
import os
import glob

if __name__ == '__main__':

    # get the lat_subset we are working on
    start_lat = float(os.environ['start_lat'])
    lat_start_points = [start_lat]

    ts = int(os.environ['iWeek'])
    print(ts)

    lon_start_points = np.linspace(111.025, 153.025, 22)
    data_dir = os.environ['glm_dir'] #'/g/data/w97/ad9701/p_prob_analysis/model_output/awra/GLM_results_model4/'
    data_dir_ts = data_dir + 'sm_week' + str(ts) + '/'

    for start_lat in lat_start_points:
        # get data corresponding to all lon dimensions in a list
        ds_list = []
        for start_lon in lon_start_points:
            end_lat = start_lat-2
            end_lon = start_lon+2
            
            file_names = data_dir_ts + 'GLM_results_lat' + str(start_lat) + '_' + str(end_lat) + '_lon' + str(start_lon) + '_' + str(end_lon) + '*.nc'
            ds = xr.open_mfdataset(file_names)
            ds_list.append(ds)     
        
        # concatenate all lon dimensions
        ds_concat_lon = xr.concat(ds_list, dim = 'lon')
        for year, sample in ds_concat_lon.groupby('time.year'):
            out_file = data_dir_ts + 'concat_time_lon/' + 'GLM_results_lat' + str(start_lat) + '_' + str(end_lat) + '_' + str(year) + '.nc'
            sample.to_netcdf(out_file)
