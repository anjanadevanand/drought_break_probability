import xarray as xr
import numpy as np
import climtas

if __name__ == '__main__':
    
    out_dir = '/g/data/w97/ad9701/p_prob_analysis/processed_data/awra/sm_refPeriod_perc/by_doy/'
    awra_dir = '/g/data/fj8/BoM/AWRA/DATA/SCHEDULED-V6/processed/values/day/'
    sm_files = 'sm_[1-2]*.nc'
    doy_sel = int(os.environ['doy_sel'])

    ds_sm_temp = xr.open_mfdataset(awra_dir + sm_files)
    # lat_slice = slice(-32, -39)      #slice(-36.3, -36.2)  #tiny slice for testing
    # lon_slice = slice(139, 152)      #slice(148.9, 149)
    time_slice = slice('1911-01-01', '2021-12-31')

    # converting the datatypes of SM to match P
    lat_new = np.float32(ds_sm_temp['latitude'])
    lon_new = np.float32(ds_sm_temp['longitude'])

    # rename & reassign lat-lon to match the precip data; subset lat-lon
    ds_sm = ds_sm_temp.rename({'latitude':'lat','longitude':'lon'}).assign_coords(lat=lat_new, lon=lon_new).sel(time = time_slice)
    
    for doy, sample in ds_sm.sm.rolling(time=31, center=True).construct(time='window').groupby('time.dayofyear'):
        if doy == doy_sel:
            doy_pct = sample.load().quantile([0.1, 0.2, 0.3, 0.4], dim=['time', 'window'])
            out_file = 'sm_191101_to_202005_perc_' + 'day' + str(doy) + '.nc'
            doy_pct.to_netcdf(out_dir + out_file)