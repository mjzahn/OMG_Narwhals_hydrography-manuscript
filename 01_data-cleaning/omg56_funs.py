## all functions for 'L1 processing OMG Narwhals data - create netCDFs from mooring temp logger (sbe56) data'

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
from datetime import datetime

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
            
        # measurement sample interval (time in seconds between measurements) is keyed by the word 'interval'
        # appearing in the line        
        if 'interval' in h:
            # ... the fourth part of the sample interval line has the number we want
            sample_interval = h.split()[4:5]
            sample_interval.append('seconds')
            
        # measurement parameter name is keyed by the word 'name' appearing in the line
        if 'name' in h:
            # ... remove the '# name' part and the '\n' part of the string and append to list
            var_names.append(h.split('# name ')[1])        
            
        if 'HardwareData DeviceType' in h:
            print(h)
            device_type = h.split()[2][12:-1]
            serial_number = h.split('SerialNumber=')[1][1:-2]
            print(device_type, serial_number)
                
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
    
    # make object for sample interval to use in global attributes of final dataset
    sample_interval_iso = 'P' + sample_interval[0] + 'S'
            
    return(data_lines, device_type, serial_number, start_time_dt64, sample_interval_td64, sample_interval_iso, var_names)

## Function create_Dataset()
### extracts data and creates xarray data arrays and then a DataSet Object (collection of all DataArray objects)
def create_Dataset(data_lines, start_time_dt64, sample_interval_td64, glacier_front, lat, lon):
    # identify which 'station' or site the data are from
    if glacier_front == 'Rink glacier':
        station = ['Rink/Fisher']
    if glacier_front == 'Sverdrup glacier':
        station = ['Sverdrup']
    if glacier_front == 'Kong Oscar glacier':
        station = ['Kong Oscar']
    
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
    time = data_array[:,0]
    temperature = data_array[:,1]
        
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
    
    # Create DataArray Objects for lat/lon variables and add the station coordinate
    lat_da = xr.DataArray(lat, dims=['station'], coords={'station':station})
    lon_da = xr.DataArray(lon, dims=['station'], coords={'station':station})
    
    ## Create DataArray Objects from the measurements, add the measurement time coordinate
    temperature_da = xr.DataArray([temperature], dims=['station','time'], coords={'time':measurement_times, 'station':station})
    
    ## SANITY CHECK PLOTS ---------------------------------------------------------------
    
    # plot but skip last day of measurements
    # duty cycle is 20 seconds (4320 measurements per day)
    # temperature_da.plot()
    # plt.title("all/raw data")
    # plt.show()
    
#     # 2019 data
#     if '2018' in str(start_time_dt64):
#         # plot again but skip last days of measurements
#         # duty cycle is 20 seconds (4320 measurements per day)
#         if glacier_front == 'Sverdrup glacier':
#             temperature_da[0:-4320].plot()
#             plt.title("last day removed")
#             plt.show()
#         elif glacier_front == 'Kong Oscar glacier':
#             temperature_da[4320:-4320].plot()
#             plt.title("first and last day removed")
#             plt.show()
#         elif probe_num == 'PROBE_14':
#             temperature_da[8640:-8640].plot()
#             plt.title("first two and last two days removed")
#             plt.show()
#         else: # Rink
#             temperature_da[0:-8640].plot()
#             plt.title("last two days removed")
#             plt.show()
    
#     # 2020 data
#     if '2019' in str(start_time_dt64):
#         # plot again but skip first day and last days for storage and transit time between 2020-2021 (469 days*4320)
#         temperature_da[8640:-2026080].plot()
#         plt.title("first two days and 2020-21 data removed")
#         plt.show()
    
    ## ----------------------------------------------------------------------------------
    
    ## add metadata to data arrays
    temperature_da.name = 'temperature'
    temperature_da.attrs['long_name'] = 'sea water temperature'
    temperature_da.attrs['standard_name'] = 'sea_water_temperature'
    temperature_da.attrs['units'] = 'degrees_C'
    temperature_da.attrs['coverage_content_type'] = 'physicalMeasurement'
    temperature_da.attrs['seabird_var_name'] = 'tv290C'
    temperature_da.attrs['comments'] = 'ITS-90'
    temperature_da.attrs['valid_min'] = float(-2.2)
    temperature_da.attrs['valid_max'] = float(35)

    lat_da.name = 'latitude'
    lat_da.attrs['long_name'] = 'station latitude'
    lat_da.attrs['standard_name'] = 'latitude'
    lat_da.attrs['units'] = 'degrees_north'
    lat_da.attrs['coverage_content_type'] = 'coordinate'
    lat_da.attrs['comments'] = 'Latitude of mooring location.'
    
    lon_da.name = 'longitude'
    lon_da.attrs['long_name'] = 'station longitude'
    lon_da.attrs['standard_name'] = 'longitude'
    lon_da.attrs['units'] = 'degrees_east'
    lon_da.attrs['coverage_content_type'] = 'coordinate'
    lon_da.attrs['comments'] = 'Longitude of mooring location.'
    
    # merge together the different xarray DataArray objects
    mooring_ds = xr.merge([lat_da, lon_da, temperature_da],
                      combine_attrs='drop_conflicts')
    
    # add time and station dimension attributes
    mooring_ds.station.attrs['long_name'] = 'station'
    mooring_ds.station.attrs['standard_name'] = 'station'
    mooring_ds.station.attrs['coverage_content_type'] = 'coordinate'
    mooring_ds.station.attrs['comments'] = 'One of three ocean mooring sites in Melville Bay: Rink/Fisher, Kong Oscar, and Sverdrup.'
    
    mooring_ds.time.attrs['long_name'] = 'time'
    mooring_ds.time.attrs['standard_name'] = 'time'
    mooring_ds.time.attrs['axis'] = 'T'
    mooring_ds.time.attrs['coverage_content_type'] = 'coordinate'
    
    return(mooring_ds)

## Function add_metadata() to add global attributes

def add_metadata(mooring_ds, uuid, glacier_front, bottom_depth, depth_target, depth_actual, netcdf_filename, serial_number, device_type, probe_num, sample_interval_iso):
    
    # get recording duration from start and end times
    start_time = mooring_ds.time[0].values
    end_time = mooring_ds.time[-1].values
    tdelta = pd.Timedelta(end_time - start_time).isoformat()
    
    # clear copied attributes from merge
    mooring_ds.attrs = ''
    
    ## add attributes to dataset
    mooring_ds.attrs['title'] = 'OMG Narwhals Moored Temperature Logger Level 1 Data'
    mooring_ds.attrs['summary'] = 'This dataset contains temperature measurements from a temperature logger sensor attached to an ocean mooring. This dataset was collected by the Oceans Melting Greenland (OMG) Narwhals program that provides two years of oceanographic measurements from Melville Bay, northwest Greenland. Between August 2018 to August 2020, three bottom-mounted moorings with a suite of instrumentation were deployed in front of three glaciers: Sverdrup Glacier, Kong Oscar Glacier, and Rink Glacier.'
    mooring_ds.attrs['keywords'] = 'Water Temperature'
    mooring_ds.attrs['keywords_vocabulary'] = 'NASA Global Change Master Directory (GCMD) Science Keywords'
    mooring_ds.attrs['Conventions'] = 'CF-1.8, ACDD-1.3'
    mooring_ds.attrs['standard_name_vocabulary'] = 'NetCDF Climate and Forecast (CF) Metadata Convention'
    mooring_ds.attrs['id'] = 'OMG_Narwhals_Mooring_Temp_L1'
    mooring_ds.attrs['uuid'] = uuid
    mooring_ds.attrs['featureType'] = "timeSeries"
    mooring_ds.attrs['cdm_data_type'] = "Station"
    mooring_ds.attrs['platform'] = 'mooring'
    mooring_ds.attrs['region'] = 'Melville Bay, northwest Greenland'
    mooring_ds.attrs['glacier_front'] = glacier_front
    mooring_ds.attrs['bottom_depth'] = bottom_depth
    mooring_ds.attrs['filename'] = netcdf_filename
    mooring_ds.attrs['serial_number'] = serial_number
    mooring_ds.attrs['probe_num'] = probe_num
    mooring_ds.attrs['instrument'] = device_type
    mooring_ds.attrs['target_sensor_depth'] = depth_target
    mooring_ds.attrs['actual_sensor_depth'] = depth_actual
    mooring_ds.attrs['history'] = "The dataset was created from processed *.cnv files from the instrument."
    mooring_ds.attrs['source'] = 'Temperature data were collected using temperature logger sensors purchased from Sea-Bird Electronics, Inc. that were attached to moorings.'
    mooring_ds.attrs['processing_level'] = 'L1'
    mooring_ds.attrs['acknowledgement'] = "This research was carried out by the Jet Propulsion Laboratory, managed by the California Institute of Technology under a contract with the National Aeronautics and Space Administration, the University of Washington's Applied Physics Laboratory and School of Aquatic and Fishery Sciences, and the Greenland Institute of Natural Resources."
    mooring_ds.attrs['license'] = 'Public Domain'
    mooring_ds.attrs['product_version'] = '1.0'
    # mooring_ds.attrs['references'] = '' # DOI number
    mooring_ds.attrs['creator_name'] = 'Marie J. Zahn'
    mooring_ds.attrs['creator_email'] = 'mzahn@uw.edu'
    mooring_ds.attrs['creator_type'] = 'person'
    mooring_ds.attrs['creator_institution'] = 'University of Washington'
    mooring_ds.attrs['institution'] = 'University of Washington'
    mooring_ds.attrs['project'] = 'Oceans Melting Greenland (OMG) Narwhals'
    mooring_ds.attrs['contributor_name'] = 'Marie J. Zahn, Kristin L. Laidre, Malene J. Simon, Ian Fenty'
    mooring_ds.attrs['contributor_role'] = "author, principal investigator, co-investigator, co-investigator" 
    mooring_ds.attrs['contributor_email'] = 'mzahn@uw.edu; klaidre@uw.edu; masi@natur.gl; ian.fenty@jpl.nasa.gov'
    mooring_ds.attrs['naming_authority'] = 'gov.nasa.jpl'
    mooring_ds.attrs['program'] = 'NASA Earth Venture Suborbital-2 (EVS-2) and Office of Naval Research (ONR) Marine Mammals and Biology Program'
    mooring_ds.attrs['publisher_name'] = 'Physical Oceanography Distributed Active Archive Center (PO.DAAC)'
    mooring_ds.attrs['publisher_institution'] = 'NASA Jet Propulsion Laboratory (JPL)'
    mooring_ds.attrs['publisher_email'] = 'podaac@podaac.jpl.nasa.gov'
    mooring_ds.attrs['publisher_url'] = 'https://podaac.jpl.nasa.gov/'
    mooring_ds.attrs['publisher_type'] = 'group'
    mooring_ds.attrs['geospatial_lat_min'] = mooring_ds.latitude.values[0]
    mooring_ds.attrs['geospatial_lat_max'] = mooring_ds.latitude.values[0]
    mooring_ds.attrs['geospatial_lat_units'] = "degrees_north"
    mooring_ds.attrs['geospatial_lon_min'] = mooring_ds.longitude.values[0]
    mooring_ds.attrs['geospatial_lon_max'] = mooring_ds.longitude.values[0]
    mooring_ds.attrs['geospatial_lon_units'] = "degrees_east"
    
    mooring_ds.attrs['geospatial_vertical_min'] = depth_actual[:-7]
    mooring_ds.attrs['geospatial_vertical_max'] = depth_actual[:-7]
    mooring_ds.attrs['geospatial_vertical_units'] = 'meters'
    mooring_ds.attrs['geospatial_vertical_positive'] = 'down'
      
    mooring_ds.attrs['time_coverage_resolution'] = sample_interval_iso  
    mooring_ds.attrs['time_coverage_start'] = str(start_time)[:-10]
    mooring_ds.attrs['time_coverage_end'] = str(end_time)[:-10]
    mooring_ds.attrs['time_coverage_duration'] = tdelta
    mooring_ds.attrs['date_created'] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    
    # with xr.set_options(display_style="html"):
    #     display(mooring_ds)

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