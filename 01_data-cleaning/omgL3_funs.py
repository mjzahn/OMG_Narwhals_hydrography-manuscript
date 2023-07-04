## all functions for time-averaged, mooring-specific datasets for Processing Level L3 datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr
import netCDF4 as nc4
from datetime import datetime

# function to calculate daily average for sbe37 data
def sbe37_avg(netcdf, mooring_L2_dir, avg_length):
    ## open dataset
    sbe37_dir = Path(mooring_L2_dir)
    sbe37_ds = xr.open_dataset(sbe37_dir / netcdf)
    sbe37_ds.close()
    
    # select one station
    sbe37 = sbe37_ds.isel(station=0)
    
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
    time = data.time.resample(time=avg_length, label="left").mean('time') # time coordinates
    
    # Create DataArray Objects from the measurements, add the measurement time coordinate
    conductivity_da = xr.DataArray(conductivity, dims='time', coords={'time':time})
    density_da = xr.DataArray(density, dims='time', coords={'time':time})
    depth_da = xr.DataArray(depth, dims='time', coords={'time':time})
    potential_temperature_da = xr.DataArray(potential_temperature, dims='time', coords={'time':time})
    pressure_da = xr.DataArray(pressure, dims='time', coords={'time':time})
    salinity_da = xr.DataArray(salinity, dims='time', coords={'time':time})
    temperature_da = xr.DataArray(temperature, dims='time', coords={'time':time})
    
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

    temperature_da.name = 'temperature'
    temperature_da.attrs = sbe37.temperature.attrs
    
    # merge together the different xarray DataArray objects
    sbe37_ds = xr.merge([conductivity_da,density_da, depth_da,\
                       potential_temperature_da, pressure_da,\
                       salinity_da, temperature_da])
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
    sbe56_data = xr.open_dataset(sbe56_dir / netcdf)
    sbe56_data.close()
    # select one station
    data = sbe56_data.isel(station=0)
    
    ## remove depth_flag data from the more shallow CTD (sbe37) data
    ## open dataset
    sbe37_dir = Path(mooring_L2_dir)
    sbe37_ds = xr.open_dataset(sbe37_dir / netcdf_sbe37)
    sbe37_ds.close()
    # select one station
    sbe37 = sbe37_ds.isel(station=0)
    
    ## remove depth spikes in sbe56 data that are flagged in sbe37 data --------------------
    # Fix time dimension to remove seconds because they differ between sbe37 and sbe56
    time_sel = sbe37.time[np.where(sbe37.flag_depth == 1)]
    # retain date and, hour minute for sbe37 data
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
    
    # print('length of raw sbe56 data: ', len(data.time))
    # print('length of sbe56 data with flagged data removed: ',len(data_new.time))
    # print('number of sbe56 observations removed: ', len(data.time) - len(data_new.time))
    # print('length of raw sbe37 data: ',len(sbe37.time))
    # print('length of sbe37 data with flagged data removed: ',len(sbe37.time[np.where(sbe37.flag_depth == 0)]))
    # print('number of sbe37 observations removed: ', len(sbe37.time) - len(sbe37.time[np.where(sbe37.flag_depth == 0)]))

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
    time = data_new.time.resample(time=avg_length, skipna=True, base=0).mean('time') # time coordinates
    
    # Create DataArray Objects from the measurements, add the measurement time coordinate
    temperature_da = xr.DataArray(temperature, dims='time', coords={'time':time})
    
    ## plot newly averaged data
    if show_plot == True:
        temperature_da.plot()
        plt.title("Time Averaged Temperature")
        plt.show()
    
    # Add metadata to the DataArray objects
    temperature_da.name = 'temperature'
    temperature_da.attrs = data.temperature.attrs
    
    # make DataArray a dataset
    sbe56_dataset = temperature_da.to_dataset()
    # drop nans
    sbe56_ds = sbe56_dataset.dropna(dim="time")
    
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

## function to add depth dimension and coordinate to sbe37 data
def sbe37_add_dims(sbe37_ds):
    ## get depth
    sbe37_depth = int(sbe37_ds.attrs['actual_sensor_depth'][:-7])
    sbe37_serial = 'SBE37_' + sbe37_ds.attrs['serial_number']
    
    ## temperature variables
    sbe37_T = sbe37_ds.temperature.copy(deep=True)
    sbe37_T = sbe37_T.assign_coords({"depth_temperature": sbe37_depth})
    sbe37_T = sbe37_T.expand_dims(dim={'depth_temperature':1},axis=1)
    sbe37_T = sbe37_T.assign_coords(serial_number_temperature=("depth_temperature", [sbe37_serial]))
    
    ## conductivity
    sbe37_C = sbe37_ds.conductivity.copy(deep=True)
    sbe37_C = sbe37_C.assign_coords({"depth_CTD": sbe37_depth})
    sbe37_C = sbe37_C.expand_dims(dim={'depth_CTD':1},axis=1)
    sbe37_C = sbe37_C.assign_coords(serial_number_CTD=("depth_CTD", [sbe37_serial]))
    
    ## density
    sbe37_D = sbe37_ds.density.copy(deep=True)
    sbe37_D = sbe37_D.assign_coords({"depth_CTD": sbe37_depth})
    sbe37_D = sbe37_D.expand_dims(dim={'depth_CTD':1},axis=1)
    sbe37_D = sbe37_D.assign_coords(serial_number_CTD=("depth_CTD", [sbe37_serial]))
    
    ## potential temp
    sbe37_PT = sbe37_ds.potential_temperature.copy(deep=True)
    sbe37_PT = sbe37_PT.assign_coords({"depth_CTD": sbe37_depth})
    sbe37_PT = sbe37_PT.expand_dims(dim={'depth_CTD':1},axis=1)
    sbe37_PT = sbe37_PT.assign_coords(serial_number_CTD=("depth_CTD", [sbe37_serial]))
    
    ## pressure
    sbe37_P = sbe37_ds.pressure.copy(deep=True)
    sbe37_P = sbe37_P.assign_coords({"depth_CTD": sbe37_depth})
    sbe37_P = sbe37_P.expand_dims(dim={'depth_CTD':1},axis=1)
    sbe37_P = sbe37_P.assign_coords(serial_number_CTD=("depth_CTD", [sbe37_serial]))
    
    ## depth
    sbe37_DP = sbe37_ds.depth.copy(deep=True)
    sbe37_DP = sbe37_DP.assign_coords({"depth_CTD": sbe37_depth})
    sbe37_DP = sbe37_DP.expand_dims(dim={'depth_CTD':1},axis=1)
    sbe37_DP = sbe37_DP.assign_coords(serial_number_CTD=("depth_CTD", [sbe37_serial]))
    
    ## salinity
    sbe37_S = sbe37_ds.salinity.copy(deep=True)
    sbe37_S = sbe37_S.assign_coords({"depth_CTD": sbe37_depth})
    sbe37_S = sbe37_S.expand_dims(dim={'depth_CTD':1},axis=1)
    sbe37_S = sbe37_S.assign_coords(serial_number_CTD=("depth_CTD", [sbe37_serial]))
    
    return(sbe37_T, sbe37_C, sbe37_D, sbe37_PT, sbe37_P, sbe37_DP, sbe37_S)


## function to add serial_number dimension and coordinate to sbe56 data
def sbe56_add_dims(probe):
    probe_T = probe.temperature.copy(deep=True)
    sbe56_serial = 'SBE56_' + probe.attrs['serial_number']
    sbe56_depth = int(probe.attrs['actual_sensor_depth'][:-7])
    probe_T = probe_T.assign_coords({"depth_temperature": sbe56_depth})
    probe_T = probe_T.expand_dims(dim={'depth_temperature':1},axis=1)
    probe_T = probe_T.assign_coords(serial_number_temperature=("depth_temperature", [sbe56_serial]))
    
    return(probe_T)


def add_metadata(sbe_merged, tmp_ds, title, netcdf_filename, uuid):
    
    ## add station dimension with latitude and longitude variables back into dataset
    sbe_merged_expand = sbe_merged.expand_dims(dim={'station':1},axis=(0)).assign_coords({'station':tmp_ds.station})
    sbe_merged_expand = sbe_merged.assign(latitude=tmp_ds.latitude).assign(longitude=tmp_ds.longitude)
    
    ## reorder dimensions
    sbe_merged_dims = sbe_merged_expand[["station","time", "depth_temperature", "depth_CTD", "serial_number_temperature", "serial_number_CTD"]]
    sbe_merged = sbe_merged_dims.assign(latitude=sbe_merged_expand.latitude).assign(longitude=sbe_merged_expand.longitude).assign(temperature=sbe_merged_expand.temperature).assign(salinity=sbe_merged_expand.salinity).assign(conductivity=sbe_merged_expand.conductivity).assign(potential_temperature=sbe_merged_expand.potential_temperature).assign(pressure=sbe_merged_expand.pressure).assign(depth=sbe_merged_expand.depth).assign(density=sbe_merged_expand.density)
    
    ## clear copied attributes from merge
    sbe_merged.attrs = ''
    ## copy attributes from previous netCDF
    sbe_merged.attrs = tmp_ds.attrs
    
    ## add comment to data variables
    sbe_merged.conductivity.attrs['comments'] = 'Daily-mean conductivity from CTD instruments.'
    sbe_merged.density.attrs['comments'] = 'Daily-mean density from CTD instruments.'
    sbe_merged.potential_temperature.attrs['comments'] = 'ITS-90; Daily-mean potential temperature from CTD instruments.'
    sbe_merged.pressure.attrs['comments'] = 'Daily-mean pressure from CTD instruments.'
    sbe_merged.salinity.attrs['comments'] = 'Daily-mean salinity from CTD instruments.'
    sbe_merged.temperature.attrs['comments'] = 'Daily-mean temperature from both CTD and temperature logger instruments.'
    
    ## add attributes to coordinates
    sbe_merged.time.attrs['long_name'] = 'time'
    sbe_merged.time.attrs['standard_name'] = 'time'
    sbe_merged.time.attrs['axis'] = 'T'
    sbe_merged.time.attrs['coverage_content_type'] = 'coordinate'
    
    sbe_merged.latitude.name = 'latitude'
    sbe_merged.latitude.attrs['long_name'] = 'station latitude'
    sbe_merged.latitude.attrs['standard_name'] = 'latitude'
    sbe_merged.latitude.attrs['units'] = 'degrees_north'
    sbe_merged.latitude.attrs['coverage_content_type'] = 'coordinate'
    sbe_merged.latitude.attrs['comments'] = 'Latitude of mooring location.'
    
    sbe_merged.longitude.name = 'longitude'
    sbe_merged.longitude.attrs['long_name'] = 'station longitude'
    sbe_merged.longitude.attrs['standard_name'] = 'longitude'
    sbe_merged.longitude.attrs['units'] = 'degrees_east'
    sbe_merged.longitude.attrs['coverage_content_type'] = 'coordinate'
    sbe_merged.longitude.attrs['comments'] = 'Longitude of mooring location.'
    
    sbe_merged.depth_CTD.name = 'CTD depth'
    sbe_merged.depth_CTD.attrs['long_name'] = 'CTD depth'
    sbe_merged.depth_CTD.attrs['standard_name'] = 'depth'
    sbe_merged.depth_CTD.attrs['units'] = 'meters'
    sbe_merged.depth_CTD.attrs['positive'] = 'down'
    sbe_merged.depth_CTD.attrs['axis'] = 'Z'
    sbe_merged.depth_CTD.attrs['coverage_content_type'] = 'coordinate'
    sbe_merged.depth_CTD.attrs['seabird_var_name'] = 'depSM'
    sbe_merged.depth_CTD.attrs['valid_min'] = int(0)
    sbe_merged.depth_CTD.attrs['valid_max'] = int(1000)
    sbe_merged.depth_CTD.attrs['comments'] = "Depths of CTD instruments attached to mooring."
    
    sbe_merged.depth_temperature.name = 'temperature sensor depth'
    sbe_merged.depth_temperature.attrs['long_name'] = 'temperature sensor depth'
    sbe_merged.depth_temperature.attrs['standard_name'] = 'depth'
    sbe_merged.depth_temperature.attrs['units'] = 'meters'
    sbe_merged.depth_temperature.attrs['positive'] = 'down'
    sbe_merged.depth_temperature.attrs['axis'] = 'Z'
    sbe_merged.depth_temperature.attrs['coverage_content_type'] = 'coordinate'
    sbe_merged.depth_temperature.attrs['seabird_var_name'] = 'depSM'
    sbe_merged.depth_temperature.attrs['valid_min'] = int(0)
    sbe_merged.depth_temperature.attrs['valid_max'] = int(1000)
    sbe_merged.depth_temperature.attrs['comments'] = "Depths of temperature sensors attached to mooring, inclusive of CTD and temperature logger instruments."
    
    sbe_merged.serial_number_CTD.name = 'serial_number_CTD'
    sbe_merged.serial_number_CTD.attrs['long_name'] = 'CTD instrument serial number'
    sbe_merged.serial_number_CTD.attrs['coverage_content_type'] = 'auxiliaryInformation'
    
    sbe_merged.serial_number_temperature.name = 'serial_number_temperature'
    sbe_merged.serial_number_temperature.attrs['long_name'] = 'Temperature sensor instrument serial number'
    sbe_merged.serial_number_temperature.attrs['coverage_content_type'] = 'auxiliaryInformation'
    
    ## remove unneeded attrs
    sbe_merged.attrs.pop('serial_number')
    sbe_merged.attrs.pop('instrument')
    sbe_merged.attrs.pop('target_sensor_depth')
    sbe_merged.attrs.pop('actual_sensor_depth')
    sbe_merged.attrs.pop('time_coverage_resolution')
    
    ## add global attributes to dataset
    # get recording duration from start and end times
    start_time = sbe_merged.time[0].values
    end_time = sbe_merged.time[-1].values
    tdelta = pd.Timedelta(end_time - start_time).isoformat()
    
    sbe_merged.attrs['title'] = title
    sbe_merged.attrs['id'] = netcdf_filename[:-3]
    sbe_merged.attrs['uuid'] = uuid
    sbe_merged.attrs['summary'] = "This file contains daily-averaged conductivity, temperature, and depth measurements from CTD and temperature logger instruments attached to the same mooring from one of three sites in Melville Bay, northwest Greenland. This dataset was collected by the Oceans Melting Greenland (OMG) Narwhals program that provides two years of oceanographic measurements. Between August 2018 to August 2020, three bottom-mounted moorings with a suite of instrumentation were deployed in front of three glaciers: Sverdrup Glacier, Kong Oscar Glacier, and Rink Glacier."
    sbe_merged.attrs['source'] = "Temperature and salinity data were collected using Conductivity Temperature Depth (CTD) and temperature logger instruments purchased from Sea-Bird Electronics, Inc. that were attached to the same mooring."
    sbe_merged.attrs['instrument'] = 'SBE37SM-RS232; SBE56'
    sbe_merged.attrs['history'] = "This dataset was created using CTD L2 datasets and temperature logger L1 datasets to combine observations from the same mooring."
    sbe_merged.attrs['filename'] = netcdf_filename
    sbe_merged.attrs['processing_level'] = 'L3'
    sbe_merged.attrs['geospatial_vertical_min'] = sbe_merged.depth_temperature.values.min()
    sbe_merged.attrs['geospatial_vertical_max'] = sbe_merged.depth_temperature.values.max()
    sbe_merged.attrs['time_coverage_start'] = str(start_time)[:-10]
    sbe_merged.attrs['time_coverage_end'] = str(end_time)[:-10]
    sbe_merged.attrs['time_coverage_duration'] = tdelta
    sbe_merged.attrs['time_coverage_resolution'] = 'P1D'
    sbe_merged.attrs['date_created'] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    
    return(sbe_merged)

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