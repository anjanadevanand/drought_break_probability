import numpy as np
import xarray as xr
import climtas
import os
import datetime


def calc_agcd_accum(nWeek, allWeek_dict, out_dir, sub_dir = '',
                 agcd_dir = '/g/data/zv2/agcd/v1/precip/total/r005/01day/', agcd_files = 'agcd_v1_precip_total_r005_daily_*.nc', lat_slice = slice(-44, -10), lon_slice = slice(112, 154)):
    '''
    Function to create nWeekly accumulations of agcd precipitation data. lat_slice and lon_slice are used to match the AWRA dataset. sub_dir should start with a '/' if specified.
    '''
    ds_agcd = xr.open_mfdataset(agcd_dir + agcd_files, chunks = {'lat':400,'lon':400})
    
    time_slice = slice(allWeek_dict['start_day'][nWeek], allWeek_dict['end_day'][nWeek])
    time_chunk_dict = {'time':allWeek_dict['time_chunk'][nWeek]}
    da_P = ds_agcd['precip'].sel(lat = lat_slice, lon = lon_slice, time = time_slice).chunk(chunks = time_chunk_dict)  # time needs to be rechunked for blocked_resample operation
    da_P_accum = climtas.blocked.blocked_resample(da_P, time = 7*nWeek).sum()
    
    full_dir_path = out_dir + 'P_week' + str(nWeek) + sub_dir
    print(full_dir_path)
    if not os.path.exists(full_dir_path):
        os.makedirs(full_dir_path)
    
    for year, data in da_P_accum.groupby('time.year'):
        print('Writing file:' + str(year))
        out_file = full_dir_path + '/P_week' + str(nWeek) + '_SEA_' + str(year) + '.nc'
        climtas.io.to_netcdf_throttled(data, f'{out_file}')
    return None

def calc_awra_accum(nWeek, allWeek_dict, file_names, var_name, out_dir, sub_dir = '',
            awra_dir = '/g/data/fj8/BoM/AWRA/DATA/SCHEDULED-V6/', lat_slice = None, lon_slice = None):
    '''
    Function to create nWeekly accumulations of awra evaporation & runoff data. sub_dir should start with a '/' if specified.
    '''
    ds = xr.open_mfdataset(awra_dir + file_names, chunks = {'lat':400,'lon':400})
    
    time_slice = slice(allWeek_dict['start_day'][nWeek], allWeek_dict['end_day'][nWeek])
    time_chunk_dict = {'time':allWeek_dict['time_chunk'][nWeek]}
    if lat_slice is None:
        da_var = ds[var_name].sel(time = time_slice).chunk(chunks = time_chunk_dict).rename({'latitude':'lat','longitude':'lon'})  # time needs to be rechunked for blocked_resample operation
    else:
        da_var = ds[var_name].sel(time = time_slice, latitude = lat_slice, longitude = lon_slice).chunk(chunks = time_chunk_dict).rename({'latitude':'lat','longitude':'lon'})     
    da_var_accum = climtas.blocked.blocked_resample(da_var, time = 7*nWeek).sum()
    
    if var_name == "etot":
        full_dir_path = out_dir + 'E_week' + str(nWeek) + sub_dir
    elif var_name == "qtot":
        full_dir_path = out_dir + 'Q_week' + str(nWeek) + sub_dir
    if not os.path.exists(full_dir_path):
        os.makedirs(full_dir_path) 
    
    for year, data in da_var_accum.groupby('time.year'):
        print('Writing file:' + str(year))
        if var_name == "etot":
            out_file = out_dir + 'E_week' + str(nWeek) + sub_dir + '/E_week' + str(nWeek) + '_SEA_' + str(year) + '.nc'
        elif var_name == "qtot":
            out_file = out_dir + 'Q_week' + str(nWeek) + sub_dir + '/Q_week' + str(nWeek) + '_SEA_' + str(year) + '.nc'
        climtas.io.to_netcdf_throttled(data, f'{out_file}')
    return None
                                   
def process_awra_sm(nWeek, out_dir, sub_dir = '',
            awra_dir = '/g/data/fj8/BoM/AWRA/DATA/SCHEDULED-V6/processed/values/day/', file_names = 'sm_[1-2]*.nc'):
    '''
    Function to process weekly awra sm data. sub_dir should start with a '/' if specified.
    '''
    ds_temp = xr.open_mfdataset(awra_dir + file_names, chunks = {'lat':400,'lon':400})
    
    # converting the datatypes of SM to match P
    lat_new = np.float32(ds_sm_temp['latitude'])
    lon_new = np.float32(ds_sm_temp['longitude'])
    ds = ds_temp.rename({'latitude':'lat','longitude':'lon'}).assign_coords(lat = lat_new, lon = lon_new)
  
    # this data will be used as a sample to look at the days at which soil moisture values are needed for analyses
    fname_sample = 'E_*_AU_*.nc'
    sample_dir = out_dir + 'E_week' + str(iW) + sub_dir + '/'
    ds_sample = xr.open_mfdataset(sample_dir + fname_sample, chunks = {'lat':400, 'lon':400})

    # Points in time at which sm data is required
    daydiff = np.timedelta64(1, 'D')
    initTime = ds_sample.time - daydiff
    nWeekdiff = np.timedelta64((7*nWeek), 'D')
    endTime = initTime + nWeekdiff
    
    ds_init = ds.sel(time = initTime)
    ds_end = ds.sel(time = endTime)

    # creating a temporary dataset that will be used to calculate the soil moisture differences using the initial and end sm values
    # as I want the time label to correspond to the initial date, I'm reassigning the co-ordinate of the sm_end dataset
    time_init = ds_init['time']
    ds_end_fordiff = ds_sm_end.assign_coords(time = time_init)
    ds_diff = (ds_end_fordiff['sm'] - ds_init['sm']).rename('sm_diff')

    for year, data in ds_init.groupby('time.year'):
        print('Writing init file:' + str(year))
        out_file = out_dir + 'sm_week' + str(nWeek) + sub_dir + '/sm_init_week' + str(nWeek) + '_AU_' + str(year) + '.nc'
        climtas.io.to_netcdf_throttled(data, f'{out_file}')

    for year, data in ds_end.groupby('time.year'):
        print('Writing end file:' + str(year))
        out_file = out_dir + 'sm_week' + str(nWeek) + sub_dir + '/sm_end_week' + str(nWeek) + '_AU_' + str(year) + '.nc'
        climtas.io.to_netcdf_throttled(data, f'{out_file}')

    for year, data in ds_diff.groupby('time.year'):
        print('Writing diff file:' + str(year))
        out_file = out_dir + 'sm_week' + str(nWeek) + sub_dir + '/sm_diff_week' + str(nWeek) + '_AU_' + str(year) + '.nc'
        climtas.io.to_netcdf_throttled(data, f'{out_file}')
    return None

def create_week_sets(nWeek, allWeek_dict, final_end_day = "2020-05-31"):

    date = datetime.datetime.strptime(allWeek_dict['start_day'][nWeek], "%Y-%m-%d")
    timedelta = [datetime.timedelta(days = x) for x in range(1,7*nWeek)]
    start_day_list = [date + x for x in timedelta]

    end_day_sel = datetime.datetime.strptime(allWeek_dict['end_day'][nWeek], "%Y-%m-%d")
    end_day_list = [end_day_sel + x for x in timedelta]
    for i in range(len(end_day_list)):
        if (end_day_list[i] > datetime.datetime.strptime(final_end_day, "%Y-%m-%d")):
            end_day_list[i] = end_day_list[i]-datetime.timedelta(7*nWeek)

    # create dictionaries for each set and store them in a list
    week_sets = []
    for i in range(len(start_day_list)):
        temp_dict = {}
        for key in ['day_len', 'time_chunk']:
            temp_dict.update({key:dict((k, allWeek_dict[key][k]) for k in [nWeek])})
        temp_dict.update({'start_day': {nWeek: start_day_list[i].strftime("%Y-%m-%d")}})
        temp_dict.update({'end_day': {nWeek: end_day_list[i].strftime("%Y-%m-%d")}})
        week_sets.append(temp_dict)
    
    return(week_sets)

if __name__ == "main":
    # creating information required for weekly accumulations
    week_names = list(np.arange(2, 13, 2))
    ndays = list(np.arange(14, 13*14, 14))
    start_day = ['1911-01-02'] * len(week_names)

    end_day = [#'2020-05-31',   #1
               '2020-05-24',   #2
               #'2020-05-31',   #3
               '2020-05-24',   #4
               '2020-05-10',   #6
               '2020-04-26',   #8
               '2020-03-29',   #10
               '2020-03-29'    #12
              ]
    time_chunk = [#364,   #1 
                  378,   #2
                  #378,   #3
                  336,   #4
                  378,   #6
                  336,   #8
                  350,   #10
                  336    #12
                 ]

    day_len_dict = dict(zip(week_names, ndays))
    start_day_dict = dict(zip(week_names, start_day))
    end_day_dict = dict(zip(week_names, end_day))
    time_chunk_dict = dict(zip(week_names, time_chunk))

    data_names = ['day_len', 'start_day', 'end_day', 'time_chunk']
    values = [day_len_dict, start_day_dict, end_day_dict, time_chunk_dict]
    allWeek_dict = dict(zip(data_names, values))
    
    allWeek_allSets = {}
    for i in range(2, 13, 2):
        field = 'week' + str(i) + '_sets'
        allWeek_allSets.update({field: create_week_sets(nWeek = i, allWeek_dict = allWeek_dict)})
