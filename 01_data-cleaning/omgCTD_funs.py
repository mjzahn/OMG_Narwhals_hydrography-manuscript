## all functions for 'L1 and L2 processing OMG Narwhals ship-based CTD data'

import string
import numpy as np
import glob 
import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pprint
import xarray as xr
import netCDF4 as nc4


## Function open_omg_file() to open file and extract header
def open_omg_file(file, start_year, start_day, start_time_log):
    f = open(file, "r")
    
    # define new lists for the header and data lines, respectively
    header_lines = []
    data_lines = []
    
    # loop through each line, l, in the file, f
    for l in f:
        # header lines start with # or *
        if l[0] == '#' or l[0] == '*':
            header_lines.append(l)
        
        # other lines are data
        else:
            data_lines.append(l[:-1])
    
    # because 2020 CTDs have some header lines that are blank, we need to remove those rows
    if len(data_lines[0]) == 0:
        del data_lines[:2]
    
    f.close()
    print(f'number data lines {len(data_lines)}')
    print(f'number header lines {len(header_lines)}')
    
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
        if 'interval' in h:
            # ... the fourth part of the sample interval line has the number we want
            sample_interval = h.split()[4:5]
            sample_interval.append('seconds')
            
        # measurement parameter name is keyed by the word 'name' appearing in the line
        if '# name' in h:
            # ... remove the '# name' part and the '\n' part of the string and append to list
            var_names.append(h.split('# name ')[1][0:-1])        
             
    # convert start_time and sample_interval to plain language for attributes of Dataset
    start_date = ' '.join(start_time[0:3])
    # start_time = ' '.join(start_time[3:4])
    sample_interval_plain = ' '.join(sample_interval)
    
    # Convert the measurement start time and sample interval into datetime64 objects
    # create a numpy datetime64 object corresponding with the measurement start time
    # -- need to handle the month by hand because the header records as a string (e.g., 'Aug') but we need a number
    if start_date[1] == 'Jul':
        start_time_str = f'{start_time[2]}-07-{start_time[1]}T{start_time[3]}'
    elif start_time[0] == 'Aug':
        start_time_str = f'{start_time[2]}-08-{start_time[1]}T{start_time[3]}'
    elif start_time[0] == 'Sep':
        start_time_str = f'{start_time[2]}-09-{start_time[1]}T{start_time[3]}'
    # Some of the 2019 CTD cast timestamps were incorrect and need correction based on logs
    elif start_time[0] == 'Mar' or start_time[0] == 'Apr':
        start_time_str = f'{start_year}-08-0{start_day}T{start_time_log}'
        
    start_time_dt64 = np.datetime64(start_time_str)
    print(start_time_dt64)
    
    ## extract date-time for filename
    start_time_str = str(start_time_dt64)
    print(start_time_str)
    start_date = start_time_str[0:10]
    
    ## format of YYYYMMDDHHMMSS
    start_time_filename = start_time_str[0:4] + '08' + start_time_str[8:10] + start_time_str[11:13] + start_time_str[14:16] + start_time_str[17:]
    netcdf_filename = 'OMG_Narwhals_Ocean_CTD_L2_' + start_time_filename + '.nc'
    print(netcdf_filename)
            
    return(netcdf_filename, start_time_filename, data_lines, start_time_dt64, start_time_str, start_date, sample_interval_plain, var_names)


## Function create_Dataset()
### extracts data and creates xarray data arrays and then a DataSet Object (collection of all DataArray objects)
def create_Dataset(data_lines, start_time_dt64, lat, lon):
    data_list =[]
    
    for di, d in enumerate(data_lines):
        # take the line, split against whitespace, convert to numpy array,
        # and convert that array data type from string to float
        tmp = np.array(d.split()).astype('float')
        # append the resulting array to a list
        data_list.append(tmp)
    
    # convert the list of arrays to an array (fast)
    data_array = np.array(data_list, dtype=object)
    
    # extract out each variable in the data_array
    pressure = data_array[:,0]
    temperature = data_array[:,1]
    conductivity = data_array[:,2]
    depth = data_array[:,3]
    salinity = data_array[:,4]
    sound_velocity = data_array[:,5]
    density = data_array[:,6]
    potential_temperature = data_array[:,7]
    flag = data_array[:,8]
    
    ## Create DataArray Objects from the measurements, add depth coordinate
    pressure_da = xr.DataArray(pressure, dims='depth', coords={'depth':depth})
    temperature_da = xr.DataArray(temperature, dims='depth', coords={'depth':depth})
    conductivity_da = xr.DataArray(conductivity, dims='depth', coords={'depth':depth})
    salinity_da = xr.DataArray(salinity, dims='depth', coords={'depth':depth})
    sound_velocity_da = xr.DataArray(sound_velocity, dims='depth', coords={'depth':depth})
    density_da = xr.DataArray(density, dims='depth', coords={'depth':depth})
    potential_temperature_da = xr.DataArray(potential_temperature, dims='depth', coords={'depth':depth})
    flag_da = xr.DataArray(flag, dims='depth', coords={'depth':depth})
    
    ## ----------------------------------------------------------------------------------
    
    ## add metadata to data arrays
    pressure_da.name = 'pressure'
    pressure_da.attrs['units'] = 'db'
    pressure_da.attrs['seabird_var_name'] = 'prdM'
    salinity_da.attrs['comments'] = 'Strain Gauge'
    
    temperature_da.name = 'temperature'
    temperature_da.attrs['units'] = 'C'
    temperature_da.attrs['seabird_var_name'] = 'tv290C'
    temperature_da.attrs['comments'] = 'ITS-90'
    
    conductivity_da.name = 'conductivity'
    conductivity_da.attrs['units'] = 'S/m'
    conductivity_da.attrs['seabird_var_name'] = 'c0S/m'
    
    # depth_da.name = 'depth'
    # depth_da.attrs['units'] = 'm'
    # depth_da.attrs['seabird_var_name'] = 'depSM'
    # depth_da.attrs['comments'] = 'lat = 75.0'
    
    salinity_da.name = 'salinity'
    salinity_da.attrs['units'] = 'PSU'
    salinity_da.attrs['seabird_var_name'] = 'sal00'
    salinity_da.attrs['comments'] = 'Practical'
    
    sound_velocity_da.name = 'sound_velocity'
    sound_velocity_da.attrs['units'] = 'm/s'
    sound_velocity_da.attrs['seabird_var_name'] = 'svCM'
    sound_velocity_da.attrs['comments'] = 'Chen-Millero'
    
    density_da.name = 'density'
    density_da.attrs['units'] = 'kg/m^3'
    density_da.attrs['seabird_var_name'] = 'density00'
    
    potential_temperature_da.name = 'potential_temperature'
    potential_temperature_da.attrs['units'] = 'C'
    potential_temperature_da.attrs['seabird_var_name'] = 'potemp090C'
    potential_temperature_da.attrs['comments'] = 'ITS-90'
    
    flag_da.name = 'flag'
    flag_da.attrs['units'] = ''
    flag_da.attrs['seabird_var_name'] = 'flag'
    
    # merge together the different xarray DataArray objects
    ctd_ds = xr.merge([pressure_da, temperature_da, conductivity_da, salinity_da, sound_velocity_da, 
                       density_da, potential_temperature_da, flag_da], combine_attrs='drop_conflicts')

    # clear copied attributes from merge
    ctd_ds.attrs = ''
    
    # ## remove data from above ocean surface (i.e., 0 meters)
    # depth_below = np.where(depth >= 0)
    # depth_below_list = list(depth_below)
    # ctd_profile_ds = ctd_ds.isel(depth=xr.DataArray(depth_below_list, dims=['profile','obs']))
    
#     ## add lat/lon and start time coordinates
#     ctd_profile_ds = ctd_profile_ds.assign_coords({'lat': ("profile", [lat])})
#     ctd_profile_ds = ctd_profile_ds.assign_coords({'lon': ("profile", [lon])})
#     ctd_profile_ds = ctd_profile_ds.assign_coords({'start_time': ("profile", [start_time_dt64])})
    
#     return(ctd_profile_ds)

    ## add lat/lon and start time coordinates
    ctd_ds = ctd_ds.assign_coords({'lat': ("profile", [lat])})
    ctd_ds = ctd_ds.assign_coords({'lon': ("profile", [lon])})
    ctd_ds = ctd_ds.assign_coords({'start_time': ("profile", [start_time_dt64])})
    
    return(ctd_ds)

## Function add_metadata() to add global attributes

def add_metadata(ctd_profile_ds, uuid, lat, lon, seafloor_depth, cast_depth, cast_id, start_date, netcdf_filename, sample_interval_plain):
    
    ## add metadata to coords
    ctd_profile_ds.lat.attrs['long_name'] = 'latitude'
    ctd_profile_ds.lat.attrs['unit'] = 'degrees_north'
    ctd_profile_ds.lat.attrs['axis'] = 'Y'
    ctd_profile_ds.lat.attrs['valid_max'] = '90.0'
    ctd_profile_ds.lat.attrs['valid_min'] = '-90.0'
    ctd_profile_ds.lat.attrs['comment'] = "Represents the latitudinal coordinate for where the CTD was deployed at the water's surface."
    
    ctd_profile_ds.lon.attrs['long_name'] = 'longitude'
    ctd_profile_ds.lon.attrs['unit'] = 'degrees_west'
    ctd_profile_ds.lon.attrs['axis'] = 'X'
    ctd_profile_ds.lon.attrs['valid_max'] = '180.0'
    ctd_profile_ds.lon.attrs['valid_min'] = '-180.0'
    ctd_profile_ds.lon.attrs['comment'] = "Represents the longitudinal coordinate for where the CTD was deployed at the water's surface."
    
    ctd_profile_ds.start_time.attrs['comment'] = 'Represents the time at which the CTD was deployed.'
    
    ctd_profile_ds.depth.attrs['units'] = 'm'
    ctd_profile_ds.depth.attrs['positive'] = 'down'
    ctd_profile_ds.depth.attrs['axis'] = 'Z'
    ctd_profile_ds.depth.attrs['valid_min'] = '0.0'
    ctd_profile_ds.depth.attrs['valid_max'] = '5000.0'
    
    ## add attributes to dataset
    ctd_profile_ds.attrs['title'] = 'OMG Narwhals Ocean CTD Level 2 Data'
    ctd_profile_ds.attrs['summary'] = 'This dataset contains conductivity, temperature, and pressure measurements from a ship-deployed CTD instrument. It also contains derived variables: salinity, sound velocity, density, and potential temperature. This profile is one of a series of CTD casts as part of the Oceans Melting Greenland (OMG) Narwhals program. OMG Narwhals will provide subannual hydrographic variability measurements in three northwest Greenland fjords. Between July 2018 to July 2020, three bottom-mounted moorings with a suite of instrumentation were deployed year-round in three glacial front sites in Melville Bay: Sverdrup Glacier, Kong Oscar Glacier, and Fisher Islands/Rink Glacier. Examination of water properties at these sites will demonstrate the presence and potential seasonality of warm, salty Atlantic Water intrusion into these coastal glaciers. During summer cruises where moorings were deployed and/or recovered, a CTD was lowered into the water to obtain full water column profiles are various locations at the glacier fronts and offshore.'
    ctd_profile_ds.attrs['keywords'] = 'Conductivity, Salinity, Water Depth, Water Temperature'
    ctd_profile_ds.attrs['keywords_vocabulary'] = 'NASA Global Change Master Directory (GCMD) Science Keywords'
    ctd_profile_ds.attrs['id'] = 'OMG_Narwhals_Ocean_CTD_L2'
    ctd_profile_ds.attrs['featureType'] = 'profile'
    ctd_profile_ds.attrs['cast_id'] = cast_id
    ctd_profile_ds.attrs['uuid'] = uuid
    ctd_profile_ds.attrs['platform'] = 'R/V Sanna'
    ctd_profile_ds.attrs['mooring_deployment'] = '2018-2019'
    ctd_profile_ds.attrs['latitude'] = lat
    ctd_profile_ds.attrs['longitude'] = lon
    ctd_profile_ds.attrs['region'] = 'Melville Bay, West Greenland'
    ctd_profile_ds.attrs['date'] = start_date
    ctd_profile_ds.attrs['seafloor_depth'] = seafloor_depth
    ctd_profile_ds.attrs['cast_depth'] = cast_depth
    ctd_profile_ds.attrs['filename'] = netcdf_filename
    ctd_profile_ds.attrs['serial_number'] = '1906981'
    ctd_profile_ds.attrs['device_type'] = 'SBE19plus'
    
    ctd_profile_ds.attrs['source'] = 'Conductivity, Temperature and Depth (CTD) data collected from a ship-deployed CTD instrument.'
    ctd_profile_ds.attrs['processing_level'] = 'L2'
    
    ctd_profile_ds.attrs['acknowledgements'] = "This research was carried out by the Jet Propulsion Laboratory, managed by the California Institute of Technology under a contract with the National Aeronautics and Space Administration, the University of Washington's Applied Physics Laboratory and School of Aquatic and Fishery Sciences, and the Greenland Institute of Natural Resources."
    ctd_profile_ds.attrs['license'] = 'Public Domain'
    ctd_profile_ds.attrs['product_version'] = '1.0'
    # ctd_profile_ds.attrs['references'] = '' # DOI number
    ctd_profile_ds.attrs['creator_name'] = 'Marie J. Zahn, Kristin L. Laidre, Malene J. Simon, and Ian Fenty'
    ctd_profile_ds.attrs['creator_email'] = 'mzahn@uw.edu; klaidre@uw.edu; masi@natur.gl; ian.fenty@jpl.nasa.gov'
    ctd_profile_ds.attrs['creator_url'] = 'https://podaac.jpl.nasa.gov/'
    ctd_profile_ds.attrs['creator_type'] = 'group'
    ctd_profile_ds.attrs['creator_institution'] = 'University of Washington; Greenland Institute of Natural Resources; NASA Jet Propulsion Laboratory'
    ctd_profile_ds.attrs['institution'] = 'University of Washington'
    ctd_profile_ds.attrs['naming_authority'] = 'gov.nasa.jpl'
    ctd_profile_ds.attrs['project'] = 'Oceans Melting Greenland (OMG) Narwhals project'
    ctd_profile_ds.attrs['program'] = 'NASA Physical Oceanography and Office of Naval Research (ONR) Marine Mammals and Biology Program'
    ctd_profile_ds.attrs['contributor_name'] = 'OMG Narwhals Science Team'
    ctd_profile_ds.attrs['contributor_role'] = 'OMG Narwhals Science Team performed mooring deployments and recoveries to collect data and performed initial processing.'
    ctd_profile_ds.attrs['publisher_name'] = 'Physical Oceanography Distributed Active Archive Center (PO.DAAC)'
    ctd_profile_ds.attrs['publisher_institution'] = 'PO.DAAC'
    ctd_profile_ds.attrs['publisher_email'] = 'podaac@podaac.jpl.nasa.gov'
    ctd_profile_ds.attrs['publisher_url'] = 'https://podaac.jpl.nasa.gov/'
    ctd_profile_ds.attrs['publisher_type'] = 'group'
    
    
    with xr.set_options(display_style="html"):
        display(ctd_profile_ds)

    return(ctd_profile_ds)


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