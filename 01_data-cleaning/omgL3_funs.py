## all functions for time-averaged, mooring-specific datasets for Processing Level L3 datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr
import netCDF4 as nc4

# function to calculate daily average for sbe37 data
def sbe37_avg(netcdf, mooring_L2_dir, avg_length):
    ## open dataset
    sbe37_dir = Path(mooring_L2_dir)
    sbe37 = xr.open_dataset(sbe37_dir / netcdf)
    sbe37.close()
    
    ## remove observations where there were depth spikes (i.e., select the good data)
    time_sel = sbe37.time[np.where(sbe37.flag_depth == 0)]
    data = sbe37.sel(time = time_sel)
    
    # Create data arrays of averaged data
    conductivity = data.conductivity.resample(time=avg_length, label="left").mean('time') # average conductivity
    density = data.density.resample(time=avg_length, label="left").mean('time') # average density
    depth = data.depth.resample(time=avg_length, label="left").mean('time') # average depth
    potential_temperature = data.potential_temperature.resample(time=avg_length, label="left").mean('time') # average potential temp
    pressure = data.pressure.resample(time=avg_length, label="left").mean('time') # average pressure
    salinity = data.salinity.resample(time=avg_length, label="left").mean('time') # average salinity
    temperature = data.temperature.resample(time=avg_length, label="left").mean('time') # average temp
    flag = data.flag.resample(time=avg_length, label="left").mean('time') # average flag
    time = data.time.resample(time=avg_length, label="left").mean('time') # time coordinates
    
    # Create DataArray Objects from the measurements, add the measurement time coordinate
    conductivity_da = xr.DataArray(conductivity, dims='time', coords={'time':time})
    density_da = xr.DataArray(density, dims='time', coords={'time':time})
    depth_da = xr.DataArray(depth, dims='time', coords={'time':time})
    potential_temperature_da = xr.DataArray(potential_temperature, dims='time', coords={'time':time})
    pressure_da = xr.DataArray(pressure, dims='time', coords={'time':time})
    salinity_da = xr.DataArray(salinity, dims='time', coords={'time':time})
    temperature_da = xr.DataArray(temperature, dims='time', coords={'time':time})
    flag_da = xr.DataArray(flag, dims='time', coords={'time':time})
    
    # Add metadata to the DataArray objects - sbe37_1
    conductivity_da.name = 'conductivity'
    conductivity_da.attrs = sbe37.conductivity.attrs
    
    density_da.name = 'density'
    density_da.attrs = sbe37.density.attrs
    
    depth_da.name = 'depth'
    depth_da.attrs = sbe37.depth.attrs
    
    potential_temperature_da.name = 'potential_temperature'
    potential_temperature_da.attrs = sbe37.potential_temperature.attrs
    
    pressure_da.name = 'pressure'
    pressure_da.attrs = sbe37.pressure.attrs
    
    salinity_da.name = 'salinity'
    salinity_da.attrs = sbe37.salinity.attrs
    
    flag_da.name = 'flag'
    flag_da.attrs = sbe37.flag.attrs
    
    temperature_da.name = 'temperature'
    temperature_da.attrs = sbe37.temperature.attrs
    
    # merge together the different xarray DataArray objects
    sbe37_ds = xr.merge([conductivity_da,density_da, depth_da,\
                       potential_temperature_da, pressure_da,\
                       salinity_da, temperature_da, flag_da])
    # drop nans
    sbe37_ds = sbe37_ds.dropna(dim="time")
    
    # clear copied attributes from merge
    sbe37_ds.attrs = ''
    ## add attributes to dataset
    sbe37_ds.attrs = sbe37.attrs
    sbe37_ds.attrs['title'] = 'OMG Narwhal mooring time-averaged data'
    
    ## fix the time dimension
    if 'D' in avg_length:
        time_tmp1 = []
        for ti,t in enumerate(sbe37_ds.time.values):
            time_tmp1.append(np.datetime64(str(t)[0:10]))
        sbe37_time_tmp_array1 = np.array(time_tmp1, dtype='datetime64')
        sbe37_ds.time.values[:] = sbe37_time_tmp_array1[:]
    if 'H' in avg_length:
        time_tmp1 = []
        for ti,t in enumerate(sbe37_ds.time.values):
            time_tmp1.append(np.datetime64(str(t)[0:13]))
        sbe37_time_tmp_array1 = np.array(time_tmp1, dtype='datetime64')
        sbe37_ds.time.values[:] = sbe37_time_tmp_array1[:]       
    
    return(sbe37_ds)

## 2019 mooring data

def sbe56_avg(netcdf, mooring_sbe56_dir, netcdf_sbe37, mooring_L2_dir, avg_length, truncate, show_plot=False):
    # open dataset
    sbe56_dir = Path(mooring_sbe56_dir)
    data = xr.open_dataset(sbe56_dir / netcdf)
    data.close()
    
    ## remove depth_flag data from the more shallow CTD (sbe37) data
    ## open dataset
    sbe37_dir = Path(mooring_L2_dir)
    sbe37 = xr.open_dataset(sbe37_dir / netcdf_sbe37)
    sbe37.close()
    
    ## remove depth spikes in sbe56 data that are flagged in sbe37 data --------------------
    # Fix time dimension to remove seconds because they differ between sbe37 and sbe56
    time_sel = sbe37.time[np.where(sbe37.flag_depth == 1)]
    # retain dat and, hour minute for sbe37 data
    # NOTE: when I retained date, hour, AND minute, I was not able to remove all of the flagged sbe56 observations
    time_tmp = []
    for ti,t in enumerate(time_sel.time.values):
        time_tmp.append(np.datetime64(str(t)[0:13]))
    time_sel_array = np.array(time_tmp, dtype='datetime64')
    # same for sbe56 data
    time_tmp = []
    for ti,t in enumerate(data.time.values):
        time_tmp.append(np.datetime64(str(t)[0:13]))
    data_time = np.array(time_tmp, dtype='datetime64')
    
    # determine matches between sbe37 and sbe56 data
    time_flag_tmp = []
    for t in range(len(time_sel_array)):
        if any(data_time == time_sel_array[t]):
            time_flag_tmp.append(str(time_sel_array[t]))
            
    ## select only 'good data' (i.e., remove iceberg events/flag_depth data = 1)
    # this loop creates a DataArray that contains all of the flagged sbe56 timestamps that we want to remove
    for i in range(len(time_flag_tmp)):
        time_tmp = data.time.sel(time = time_flag_tmp[i])
        time_flag_all = time_tmp if i == 0 else xr.concat([time_flag_all, time_tmp], dim="time")
    
    data_new = data.sel(time=data.time[~data.time.isin(time_flag_all)])
    
    print('length of raw sbe56 data: ', len(data.time))
    print('length of sbe56 data with flagged data removed: ',len(data_new.time))
    print('number of sbe56 observations removed: ', len(data.time) - len(data_new.time))
    print('length of raw sbe37 data: ',len(sbe37.time))
    print('length of sbe37 data with flagged data removed: ',len(sbe37.time[np.where(sbe37.flag_depth == 0)]))
    print('number of sbe37 observations removed: ', len(sbe37.time) - len(sbe37.time[np.where(sbe37.flag_depth == 0)]))

    ## sanity check plot comparisons
    if show_plot == True:
        data.temperature.plot()
        plt.title("Temperature including iceberg events")
        plt.show()
        data_new.temperature.plot()
        plt.title("Temperature with iceberg events removed")
        plt.show()
    
    ## average data -------------------------------------------------
    # first, truncate so the averaging begins on the same time stamp
    data_new = data_new.sel(time=truncate)
    
    temperature = data_new.temperature.resample(time=avg_length, skipna=True, base=0).mean('time') # average temp
    flag = data_new.flag.resample(time=avg_length, skipna=True, base=0).mean('time') # average flag
    time = data_new.time.resample(time=avg_length, skipna=True, base=0).mean('time') # time coordinates
    
    # Create DataArray Objects from the measurements, add the measurement time coordinate
    temperature_da = xr.DataArray(temperature, dims='time', coords={'time':time})
    flag_da = xr.DataArray(flag, dims='time', coords={'time':time})
    
    ## plot newly averaged data
    if show_plot == True:
        temperature_da.plot()
        plt.title("Time Averaged Temperature")
        plt.show()
    
    # Add metadata to the DataArray objects
    flag_da.name = 'flag'
    flag_da.attrs = data.flag.attrs
    temperature_da.name = 'temperature'
    temperature_da.attrs = data.temperature.attrs
    
    # merge together the different xarray DataArray objects
    sbe56_ds = xr.merge([temperature_da, flag_da])
    # drop nans
    sbe56_ds = sbe56_ds.dropna(dim="time")
    
    # clear copied attributes from merge
    sbe56_ds.attrs = ''
    # add attributes to dataset
    sbe56_ds.attrs = data.attrs
    sbe56_ds.attrs['title'] = 'OMG Narwhal mooring temperature logger Level 3 Data' # change processing level
    sbe56_ds.attrs['processing_level'] = 'L3' # change processing level
    
    ## fix the time dimension
    if 'D' in avg_length:
        time_tmp1 = []
        for ti,t in enumerate(sbe56_ds.time.values):
            time_tmp1.append(np.datetime64(str(t)[0:10]))
        sbe56_time_tmp_array1 = np.array(time_tmp1, dtype='datetime64')
        sbe56_ds.time.values[:] = sbe56_time_tmp_array1[:]
    if 'H' in avg_length:
        time_tmp1 = []
        for ti,t in enumerate(sbe56_ds.time.values):
            time_tmp1.append(np.datetime64(str(t)[0:13]))
        sbe56_time_tmp_array1 = np.array(time_tmp1, dtype='datetime64')
        sbe56_ds.time.values[:] = sbe56_time_tmp_array1[:] 
        
    return(sbe56_ds)

# # original, single daily average for records
# temperature_1 = data.temperature.resample(time='1D').mean('time') # average temp
# flag_1 = data.flag.resample(time='1D').mean('time') # average flag
# time_1 = data.time.resample(time='1D').mean('time') # time coordinates


## function to add depth dimension and coordinate to sbe37 data
def sbe37_add_dims(sbe37_ds):
    ## get depth
    sbe37_depth = int(sbe37_ds.attrs['actual_sensor_depth'][:-7])
    sbe37_serial = 'SBE37_' + sbe37_ds.attrs['serial_number']
    
    ## temperature variables
    sbe37_T = sbe37_ds.temperature.copy(deep=True)
    sbe37_T = sbe37_T.assign_coords({"Depth_temp": sbe37_depth})
    sbe37_T = sbe37_T.expand_dims(dim={'Depth_temp':1},axis=0)
    sbe37_T = sbe37_T.assign_coords(SN_temp=("Depth_temp", [sbe37_serial]))
    
    ## conductivity
    sbe37_C = sbe37_ds.conductivity.copy(deep=True)
    sbe37_C = sbe37_C.assign_coords({"Depth_CTD": sbe37_depth})
    sbe37_C = sbe37_C.expand_dims(dim={'Depth_CTD':1},axis=0)
    sbe37_C = sbe37_C.assign_coords(SN_CTD=("Depth_CTD", [sbe37_serial]))
    
    ## density
    sbe37_D = sbe37_ds.density.copy(deep=True)
    sbe37_D = sbe37_D.assign_coords({"Depth_CTD": sbe37_depth})
    sbe37_D = sbe37_D.expand_dims(dim={'Depth_CTD':1},axis=0)
    sbe37_D = sbe37_D.assign_coords(SN_CTD=("Depth_CTD", [sbe37_serial]))
    
    ## potential temp
    sbe37_PT = sbe37_ds.potential_temperature.copy(deep=True)
    sbe37_PT = sbe37_PT.assign_coords({"Depth_CTD": sbe37_depth})
    sbe37_PT = sbe37_PT.expand_dims(dim={'Depth_CTD':1},axis=0)
    sbe37_PT = sbe37_PT.assign_coords(SN_CTD=("Depth_CTD", [sbe37_serial]))
    
    ## pressure
    sbe37_P = sbe37_ds.pressure.copy(deep=True)
    sbe37_P = sbe37_P.assign_coords({"Depth_CTD": sbe37_depth})
    sbe37_P = sbe37_P.expand_dims(dim={'Depth_CTD':1},axis=0)
    sbe37_P = sbe37_P.assign_coords(SN_CTD=("Depth_CTD", [sbe37_serial]))
    
    ## salinity
    sbe37_S = sbe37_ds.salinity.copy(deep=True)
    sbe37_S = sbe37_S.assign_coords({"Depth_CTD": sbe37_depth})
    sbe37_S = sbe37_S.expand_dims(dim={'Depth_CTD':1},axis=0)
    sbe37_S = sbe37_S.assign_coords(SN_CTD=("Depth_CTD", [sbe37_serial]))
    
    return(sbe37_T, sbe37_C, sbe37_D, sbe37_PT, sbe37_P, sbe37_S)


## function to add SN dimension and coordinate to sbe56 data
def sbe56_add_dims(probe):
    probe_T = probe.temperature.copy(deep=True)
    sbe56_serial = 'SBE56_' + probe.attrs['serial_number']
    sbe56_depth = int(probe.attrs['actual_sensor_depth'][:-7])
    probe_T = probe_T.assign_coords({"Depth_temp": sbe56_depth})
    probe_T = probe_T.expand_dims(dim={'Depth_temp':1},axis=0)
    probe_T = probe_T.assign_coords(SN_temp=("Depth_temp", [sbe56_serial]))
    return(probe_T)

## retaining old, long-hand version as backup
# sbe56_16_T = sbe56_16.temperature.copy(deep=True)
# sbe56_serial = 'SBE56_' + sbe56_16.attrs['serial_number']
# sbe56_16_T = sbe56_16_T.expand_dims(dim={'profile':1},axis=0)
# sbe56_16_T = sbe56_16_T.assign_coords({"SN_Temperature": ('profile', [sbe56_serial])})


## Define helper routine for proper encoding to save netCDF

# define binary precision 64 bit (double)
array_precision = np.float64
binary_output_dtype = '>f8'
netcdf_fill_value = nc4.default_fillvals['f8']
# netcdf_fill_value = '-9999'

def create_encoding(G):
    # first create encoding for the data variables 
    dv_encoding = dict()
    for dv in G.data_vars:
        dv_encoding[dv] =  {'zlib':True, \
                            'complevel':5,\
                            'shuffle':True,\
                            '_FillValue':netcdf_fill_value}
    coord_encoding = dict()

    # next encodings for the coordinates
    for coord in G.coords:
        # default encoding: no fill value
        coord_encoding[coord] = {'_FillValue':None}

        if (G[coord].values.dtype == np.int32) or \
           (G[coord].values.dtype == np.int64) :
            coord_encoding[coord]['dtype'] ='int32'

        # this is the key line, time coordinates should be saved as int32s (integer number of nanoseconds)
        if coord == 'time' or coord == 'time_bnds':
            coord_encoding[coord]['dtype'] ='int32'

    # MERGE ENCODINGS DICTIONARIES for coordinates and variables
    encoding = {**dv_encoding, **coord_encoding}

    return encoding