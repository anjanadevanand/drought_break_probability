import xarray as xr
import numpy as np
import pandas as pd
import os

if __name__ == '__main__':

    band_no = int(os.environ['band_no'])

    main_dir = '/g/data/w97/ad9701/p_prob_analysis/temp_files/'

    drght_time_slice = slice('1911-01-01', '2020-12-31')
    drght_name = 'full_record' #'recent_drght'
    drght_dir = main_dir + 'GLM_results_' + drght_name + '/validation/'
    fullrec_dir = main_dir + 'GLM_results_full_record/validation/'

    # load the threshold data file & select the drought period of interest
    events_file = 'sm_droughts/events_*.nc'
    ds_events = xr.open_mfdataset(main_dir + events_file)
    ds_events = ds_events.sel(time = drght_time_slice)

    # load the status of drought after 'ts' weeeks
    drought_status = []
    ts_list = [2, 6, 12]
    for ts in ts_list:
        ts_files = fullrec_dir + 'events_after_' + str(ts) + 'weeks_*.nc'
        ds = xr.open_mfdataset(ts_files)
        drought_status.append(ds.sel(time = drght_time_slice).rename({'sm_drought_after_' + str(ts) + 'weeks': 'drought_status'}))
        del ds
    ds_drght_status = xr.concat(drought_status, "timescale").assign_coords({'timescale': ts_list})

    # get GLM results
    glm_dir = main_dir + 'GLM_results_' + drght_name + '/'
    glm_file = 'PminusEQ_results_weeks_2_6_12*.nc'
    ds_glm = xr.open_mfdataset(glm_dir + glm_file)

    # calculations for probability bands
    band_name = np.linspace(0.1, 0.9, 9).round(2)
    low = np.linspace(0.05, 0.85, 9)
    high = np.linspace(0.15, 0.95, 9)

    # from dask.distributed import Client, LocalCluster
    # cluster = LocalCluster()
    # client = Client(cluster)
    # client

    for i in [band_no]: #range(len(band_name)):
        # select the GLM estimated probabilities that fall in the band
        sel_prob_band = ds_glm['glm_probability'].where((ds_glm['glm_probability'] >= low[i]) & (ds_glm['glm_probability'] < high[i]))
        # da_drght_status_prob_band = ds_drght_status.where(sel_prob_band > 0)
        # da_sum_drght_status_prob_band = da_drght_status_prob_band['drought_status'].sum('time')

        # out_file = drght_dir + 'drought_status_band' + str(band_name[i]) + '.nc'
        # da_drght_status_prob_band.to_netcdf(out_file)
        # out_file = drght_dir + 'sum_drought_status_band' + str(band_name[i]) + '.nc'
        # da_sum_drght_status_prob_band.to_netcdf(out_file)

        sum_drght_events = []
        drght_events = []
        for ts in ts_list:
            drght_events.append((ds_events['sm_drought'].where(sel_prob_band.sel(timescale=ts) > 0)))
            sum_drght_events.append((ds_events['sm_drought'].where(sel_prob_band.sel(timescale=ts) > 0)).sum('time'))
        # da_drght_events = xr.concat(drght_events, 'timescale').assign_coords({'timescale': ts_list})
        da_sum_drght_events = xr.concat(sum_drght_events, 'timescale').assign_coords({'timescale': ts_list})

        # out_file = drght_dir + 'drought_events_band' + str(band_name[i]) + '.nc'
        # da_drght_events.to_netcdf(out_file)
        out_file = drght_dir + 'sum_drought_events_band' + str(band_name[i]) + '.nc'
        da_sum_drght_events.to_netcdf(out_file)

        # da_prob_drought_break = 1 - (da_sum_drght_status_prob_band/da_sum_drght_events).rename('hist_prob')
        # out_file = drght_dir + 'hist_prob_drought_break_band' + str(band_name[i]) + '.nc'
        # da_prob_drought_break.to_netcdf(out_file)
