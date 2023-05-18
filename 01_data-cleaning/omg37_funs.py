## all functions for 'L1 and L2 processing OMG Narwhals data - create netCDFs from mooring CTD (sbe37) data'

import string
import numpy as np
import pandas as pd
import csv
import glob, os
import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pprint
import xarray as xr
import netCDF4 as nc4


## Function open_omg_file() to open file and extract header
def open_omg_file(file):
    # open file
    f = open(file, "r")
    
    # define new lists for the header and data lines, respectively
    header_lines = []
    data_lines = []
    
    # loop through each line, l, in the file, f
    for l in f:
        # remove the newline '\n' ending of each line (last two characters)
        l = l[:-1]
        # header lines start with # or *
        if l[0] == '#' or l[0] == '*':
            header_lines.append(l)
        
        # other lines are data
        else:
            data_lines.append(l[:-1])
    f.close()
    print(f'number data lines {len(data_lines)}')
    print(f'number header lines {len(header_lines)} \n')
    
    # extract the measurement start time & sample interval, and the measurement parameter names & units, instrument device type  and serial number 
    var_names = []

    # loop through header lines
    for h in header_lines:
    # measurement start time is keyed by the work 'start_time' appearing in the line
        if 'start_time' in h:
            # ... the third through seventh part of the start_time line has the 'Month Day Year HH:MM:SS' info
            start_time = h.split()[3:7]
            
        # measurement sample interval (time between measuremnets) is keyed by the work 'sample_interval'
        # appearing in the line        
        if 'sample interval' in h:
            # ... the fourth part of the sample interval line has the number we want
            sample_interval = h.split()[4:6]
            
        # measurement parameter name is keyed by the word 'name' appearing in the line
        if 'name' in h:
            # ... remove the '# name' part and the '\n' part of the string and append to list
            var_names.append(h.split('# name ')[1][0:-1])        
            
        if 'HardwareData DeviceType' in h:
            print(h)
            device_type = h.split()[2][12:-1]
            serial_number = h.split('SerialNumber=')[1][1:-2]
            
    # convert start_time and sample_interval to plain language for attributes of Dataset
    start_date = ' '.join(start_time[0:3])
    sample_interval_plain = ' '.join(sample_interval)
    
    # Convert the measurement start time and sample interval into datetime64 objects
    # create a numpy datetime64 object corresponding with the measurement start time
    # -- need to handle the month by hand because the header records as a string (e.g., 'Aug') but we need a number
    if start_time[1] == 'Jul':
        start_time_str = f'{start_time[2]}-07-{start_time[1]}T{start_time[3]}'
    elif start_time[0] == 'Aug':
        start_time_str = f'{start_time[2]}-08-{start_time[1]}T{start_time[3]}'
    elif start_time[0] == 'Sep':
        start_time_str = f'{start_time[2]}-09-{start_time[1]}T{start_time[3]}'
        
    start_time_dt64 = np.datetime64(start_time_str)
    print(start_time_dt64)
    
    # create a numpy time delta 64 object corresponding with the measurement sample interval
    # in the case of this example file it is 180 seconds
    if sample_interval[1] == 'seconds':
        sample_interval_td64 = np.timedelta64(sample_interval[0], 's')
    elif sample_interval[1] == 'minutes':
        sample_interval_td64 = np.timedelta64(sample_interval[0], 'm')
    elif sample_interval[1] == 'hours':
        sample_interval_td64 = np.timedelta64(sample_interval[0], 'h')
            
    return(data_lines, device_type, serial_number, start_time_dt64, start_date, sample_interval_plain, sample_interval_td64, var_names)

## Function create_Dataset()
### extracts data and creates xarray data arrays and then a DataSet Object (collection of all DataArray objects)
def create_Dataset(glacier_front, data_lines, start_time_dt64, sample_interval_td64):
    # extract out numbers from the text data lines
    data_list =[]
    
    for di, d in enumerate(data_lines):
        # take the line, split against whitespace, convert to numpy array,
        # and convert that array data type from string to float
        tmp = np.array(d.split()).astype('float')
        # append the resulting array to a list
        data_list.append(tmp)
    
    # convert the list of arrays to an array (fast)
    data_array = np.array(data_list)
    
    # extract out each variable in the data_array
    conductivity = data_array[:,0]
    density = data_array[:,1]
    depth = data_array[:,2]
    potential_temperature = data_array[:,3]
    pressure = data_array[:,4]
    salinity = data_array[:,5]
    temperature = data_array[:,6]
    time = data_array[:,7]
    
    # Make the useful measurement time array using numpy datetime64 objects
    # the file records the measurement times as fractional Julian years, which is not very useful
    print('before')
    print(time[0:5])
    
    # more useful is to make our own array of measurement times using the start time and sample interval
    measurement_times = []
    for i in range(len(time)):
        measurement_times.append(start_time_dt64 + sample_interval_td64*i)
    
    # convert list of datetime64 objects to an array
    measurement_times_array = np.array(measurement_times)  
    
    # much better
    print('\nafter')
    pprint(measurement_times[0:5])
    
    ## recording start time
    print('recording start time: ', measurement_times[0])
    
    ## recording end time
    end_time = measurement_times[len(measurement_times)-1]
    print('recording end time: ', end_time)
    
    ## Create DataArray Objects from the measurements, add the measurement time coordinate
    conductivity_da = xr.DataArray(conductivity, dims='time', coords={'time':measurement_times})
    density_da = xr.DataArray(density, dims='time', coords={'time':measurement_times})
    depth_da = xr.DataArray(depth, dims='time', coords={'time':measurement_times})
    potential_temperature_da = xr.DataArray(potential_temperature, dims='time', coords={'time':measurement_times})
    pressure_da = xr.DataArray(pressure, dims='time', coords={'time':measurement_times})
    salinity_da = xr.DataArray(salinity, dims='time', coords={'time':measurement_times})
    temperature_da = xr.DataArray(temperature, dims='time', coords={'time':measurement_times})
    
    ## add variable that indexes outliers/extreme data points from depth spikes 
    ## 2019
    if '2018' in str(start_time_dt64) and glacier_front == 'Kong Oscar glacier':
        
        flag_depth = depth_da.copy(deep=True)
        # spikes in depth data (separate from dragging events)
        flag_depth_norm = flag_depth - flag_depth.isel(time = 0) # normalize depth data around first observation
        flag_depth_norm.loc['2018-08-25':'2018-08-31'][flag_depth_norm.loc['2018-08-25':'2018-08-31'] > 1.5] = 1 # mean of -0.24395774
        flag_depth_norm.loc['2018-10-01':'2018-10-11'][flag_depth_norm.loc['2018-10-01':'2018-10-11'] > -1.5] = 1 # mean of -2.62526818
        flag_depth_norm.loc['2019-06-29':'2019-07-28'][flag_depth_norm.loc['2019-06-29':'2019-07-28'] > 8] = 1 # mean of 6.87850924
        
        # dragging periods
        flag_depth_norm.loc['2018-09-02T00:55':'2018-09-02T09:35'] = 1 # drag 1
        flag_depth_norm.loc['2018-09-07T17:40':'2018-09-07T22:56'] = 1 # drag 2
        flag_depth_norm.loc['2018-10-12T18:40':'2018-10-13T13:00'] = 1 # drag 3
        flag_depth_norm.loc['2019-07-28T19:32':'2019-07-29T09:30'] = 1 # drag 4
        
        flag_depth_norm[np.where(flag_depth_norm != 1)] = 0
        flag_depth = flag_depth_norm
        flag_depth_da = xr.DataArray(flag_depth_norm, dims='time', coords={'time':measurement_times})
        
    if '2018' in str(start_time_dt64) and glacier_front != 'Kong Oscar glacier':
        window_low = np.mean(depth_da[30000:-20000]) - 1.5
        window_high = np.mean(depth_da[30000:-20000]) + 1.5
        
        flag_depth = depth_da.copy(deep=True)
        flag_depth[np.where((depth_da < window_low) | (depth_da > window_high))] = 1
        flag_depth[np.where(flag_depth != 1)] = 0
        flag_depth_da = xr.DataArray(flag_depth, dims='time', coords={'time':measurement_times})
        
    ## 2020
    if '2019' in str(start_time_dt64):
        window_low = np.mean(depth_da[30000:-227520]) - 1.5
        window_high = np.mean(depth_da[30000:-227520]) + 1.5
        
        flag_depth = depth_da.copy(deep=True)
        flag_depth[np.where((depth_da < window_low) | (depth_da > window_high))] = 1
        flag_depth[np.where(flag_depth != 1)] = 0
        flag_depth_da = xr.DataArray(flag_depth, dims='time', coords={'time':measurement_times})
    
    ## add metadata to data arrays
    
    conductivity_da.name = 'conductivity'
    conductivity_da.attrs['units'] = 'S/m'
    conductivity_da.attrs['seabird_var_name'] = 'cond0S/m'
    
    density_da.name = 'density'
    density_da.attrs['units'] = 'kg m-3'
    density_da.attrs['seabird_var_name'] = 'density00'
    
    depth_da.name = 'depth'
    depth_da.attrs['units'] = 'm'
    depth_da.attrs['seabird_var_name'] = 'depSM'
    
    potential_temperature_da.name = 'potential_temperature'
    potential_temperature_da.attrs['units'] = 'C'
    potential_temperature_da.attrs['seabird_var_name'] = 'potemp090C'
    potential_temperature_da.attrs['comments'] = 'ITS-90'
    
    pressure_da.name = 'pressure'
    pressure_da.attrs['units'] = 'db'
    pressure_da.attrs['seabird_var_name'] = 'prdM'
    pressure_da.attrs['comments'] = 'strain gauge'
    
    salinity_da.name = 'salinity'
    salinity_da.attrs['units'] = 'PSU'
    salinity_da.attrs['seabird_var_name'] = 'sal00'
    salinity_da.attrs['comments'] = 'Practical'
    
    temperature_da.name = 'temperature'
    temperature_da.attrs['units'] = 'C'
    temperature_da.attrs['seabird_var_name'] = 'tv290C'
    temperature_da.attrs['comments'] = 'ITS-90'
    
    flag_depth_da.name = 'flag_depth'
    flag_depth_da.attrs['units'] = ''
    flag_depth_da.attrs['comments'] = "Index for outliers where 0 corresponds to reasonable observations and 1 indicates extreme values based on spikes in depth observations likely due to iceberg events pushing the mooring instruments downward. Variable measurements corresponding to a 1 in 'flag_depth' can be removed."
    
    
    # merge together the different xarray DataArray objects
    mooring_ds = xr.merge([conductivity_da,density_da, depth_da,\
                       potential_temperature_da, pressure_da,\
                       salinity_da, temperature_da, flag_depth_da],
                      combine_attrs='drop_conflicts')

    return(mooring_ds)


## Function add_metadata() to add global attributes

def add_metadata(mooring_ds, uuid, lat, lon, start_date, glacier_front, bottom_depth, netcdf_filename, serial_number, device_type, depth_target, depth_actual, sample_interval_plain):
    # clear copied attributes from merge
    mooring_ds.attrs = ''
    
    ## add attributes to dataset
    mooring_ds.attrs['title'] = 'OMG Narwhals mooring CTD Level 2 Data'
    mooring_ds.attrs['summary'] = 'This dataset contains conductivity, temperature, and pressure measurements from a CTD instrument that was attached to an ocean mooring. It also contains derived variables: depth, salinity, density, and potential temperature. This dataset was collected by the Oceans Melting Greenland (OMG) Narwhals program that will provide subannual hydrographic variability measurements in three northwest Greenland fjords. Between July 2018 to July 2020, three bottom-mounted moorings with a suite of instrumentation were deployed year-round in three glacial fjord sites in Melville Bay, West Greenland: Sverdrup Glacier, Kong Oscar Glacier, and Fisher Islands/Rink Glacier. Examination of water properties at these sites will demonstrate the presence and potential seasonality of warm, salty Atlantic Water intrusion into these marine-terminating glaciers. Additionally, during summer cruises where moorings were deployed and/or recovered, a CTD was lowered into the water to obtain full water column profiles at various locations near the glacier fronts and offshore.'
    mooring_ds.attrs['keywords'] = 'Conductivity, Salinity, Water Depth, Water Temperature'
    mooring_ds.attrs['keywords_vocabulary'] = 'NASA Global Change Master Directory (GCMD) Science Keywords'
    mooring_ds.attrs['id'] = 'OMG_Narwhals_Mooring_CTD_L2'
    mooring_ds.attrs['uuid'] = uuid
    mooring_ds.attrs['platform'] = 'R/V Sanna'
    mooring_ds.attrs['mooring_deployment'] = '2018-2019'
    mooring_ds.attrs['mooring_latitude'] = lat
    mooring_ds.attrs['mooring_longitude'] = lon
    mooring_ds.attrs['region'] = 'Melville Bay, West Greenland'
    mooring_ds.attrs['start_date'] = start_date
    mooring_ds.attrs['glacier_front'] = glacier_front
    mooring_ds.attrs['bottom_depth'] = bottom_depth
    mooring_ds.attrs['filename'] = netcdf_filename
    mooring_ds.attrs['serial_number'] = serial_number
    mooring_ds.attrs['device_type'] = device_type
    mooring_ds.attrs['target_sensor_depth'] = depth_target
    mooring_ds.attrs['actual_sensor_depth'] = depth_actual
    mooring_ds.attrs['sample_interval'] = sample_interval_plain
    
    mooring_ds.attrs['source'] = 'Temperature and salinity data were collected using Conductivity Temperature Depth (CTD) instruments purchased from Sea-Bird Electronics, Inc. that were attached to moorings.'
    mooring_ds.attrs['processing_level'] = 'L2'
    
    mooring_ds.attrs['acknowledgements'] = "This research was carried out by the Jet Propulsion Laboratory, managed by the California Institute of Technology under a contract with the National Aeronautics and Space Administration, the University of Washington's Applied Physics Laboratory and School of Aquatic and Fishery Sciences, and the Greenland Institute of Natural Resources."
    mooring_ds.attrs['license'] = 'Public Domain'
    mooring_ds.attrs['product_version'] = '1.0'
    # mooring_ds.attrs['references'] = '' # DOI number
    mooring_ds.attrs['creator_name'] = 'Marie J. Zahn, Kristin L. Laidre, Malene J. Simon, and Ian Fenty'
    mooring_ds.attrs['creator_email'] = 'mzahn@uw.edu; klaidre@uw.edu; masi@natur.gl; ian.fenty@jpl.nasa.gov'
    mooring_ds.attrs['creator_url'] = 'https://podaac.jpl.nasa.gov/'
    mooring_ds.attrs['creator_type'] = 'group'
    mooring_ds.attrs['creator_institution'] = 'University of Washington; Greenland Institute of Natural Resources; NASA Jet Propulsion Laboratory'
    mooring_ds.attrs['institution'] = 'University of Washington'
    mooring_ds.attrs['naming_authority'] = 'gov.nasa.jpl'
    mooring_ds.attrs['project'] = 'Oceans Melting Greenland (OMG) Narwhals project'
    mooring_ds.attrs['program'] = 'NASA Physical Oceanography and Office of Naval Research (ONR) Marine Mammals and Biology Program'
    mooring_ds.attrs['contributor_name'] = 'OMG Narwhals Science Team'
    mooring_ds.attrs['contributor_role'] = 'OMG Narwhals Science Team performed mooring deployments and recoveries to collect data and performed initial processing.'
    mooring_ds.attrs['publisher_name'] = 'Physical Oceanography Distributed Active Archive Center (PO.DAAC)'
    mooring_ds.attrs['publisher_institution'] = 'PO.DAAC'
    mooring_ds.attrs['publisher_email'] = 'podaac@podaac.jpl.nasa.gov'
    mooring_ds.attrs['publisher_url'] = 'https://podaac.jpl.nasa.gov/'
    mooring_ds.attrs['publisher_type'] = 'group'
    
    with xr.set_options(display_style="html"):
        display(mooring_ds)

    return(mooring_ds)


## Function to create variable encodings (for NetCDF output file)
# define binary precision 64 bit (double)
array_precision = np.float64
binary_output_dtype = '>f8'
netcdf_fill_value = nc4.default_fillvals['f8']

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