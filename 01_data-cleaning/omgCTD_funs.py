## all functions for 'L1 and L2 processing OMG Narwhals ship-based CTD data'

import string
import numpy as np
import glob 
import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pprint
import xarray as xr
import netCDF4 as nc4
from datetime import datetime
import pandas as pd

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
    # start_date = ' '.join(start_time[0:3])
    # start_time = ' '.join(start_time[3:4])
    # sample_interval_plain = ' '.join(sample_interval)
    
    # make object for sample interval to use in global attributes of final dataset
    sample_interval_iso = 'P' + sample_interval[0] + 'S'
    
    # Convert the measurement start time and sample interval into datetime64 objects
    # create a numpy datetime64 object corresponding with the measurement start time
    # -- need to handle the month by hand because the header records as a string (e.g., 'Aug') but we need a number
    if start_time[0] == 'Jul':
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
    
    # get recording end time using start time and sampling duration
    sample_interval_td64 = np.timedelta64(250,'ms') # interval is 0.25 s
    measurement_times = []
    for i in range(len(data_lines)):
        measurement_times.append(start_time_dt64 + sample_interval_td64*i)
    end_time = measurement_times[-1]
    
    ## extract date-time for filename
    start_time_str = str(start_time_dt64)
    print(start_time_str)
    start_date = start_time_str[0:10]
    
    ## format of YYYYMMDDHHMMSS
    start_time_filename = start_time_str[0:4] + '08' + start_time_str[8:10] + start_time_str[11:13] + start_time_str[14:16] + start_time_str[17:]
    netcdf_filename = 'OMG_Narwhals_Ocean_CTD_L2_' + start_time_filename + '.nc'
    print(netcdf_filename)
            
    return(netcdf_filename, start_time_filename, data_lines, start_time_dt64, end_time, sample_interval_iso, var_names)


## Function create_Dataset()
### extracts data and creates xarray data arrays and then a DataSet Object (collection of all DataArray objects)
def create_Dataset(data_lines, start_time_dt64, lat, lon, netcdf_filename):
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
    
    ## Create DataArray Objects from the measurements, add depth coordinate
    pressure_da = xr.DataArray(pressure, dims='depth', coords={'depth':depth})
    temperature_da = xr.DataArray(temperature, dims='depth', coords={'depth':depth})
    conductivity_da = xr.DataArray(conductivity, dims='depth', coords={'depth':depth})
    salinity_da = xr.DataArray(salinity, dims='depth', coords={'depth':depth})
    sound_velocity_da = xr.DataArray(sound_velocity, dims='depth', coords={'depth':depth})
    density_da = xr.DataArray(density, dims='depth', coords={'depth':depth})
    potential_temperature_da = xr.DataArray(potential_temperature, dims='depth', coords={'depth':depth})

    ## ----------------------------------------------------------------------------------
    
    ## add metadata to data arrays
    pressure_da.name = 'pressure'
    pressure_da.attrs['long_name'] = 'sea water pressure'
    pressure_da.attrs['standard_name'] = 'sea_water_pressure'
    pressure_da.attrs['units'] = 'dBar'
    pressure_da.attrs['coverage_content_type'] = 'physicalMeasurement'
    pressure_da.attrs['seabird_var_name'] = 'prdM'
    pressure_da.attrs['valid_min'] = float(0)
    pressure_da.attrs['valid_max'] = float(1000)
    pressure_da.attrs['comments'] = 'strain gauge'
    
    salinity_da.name = 'salinity'
    salinity_da.attrs['long_name'] = 'sea water practical salinity'
    salinity_da.attrs['standard_name'] = 'sea_water_practical_salinity'
    salinity_da.attrs['units'] = '1'
    salinity_da.attrs['coverage_content_type'] = 'physicalMeasurement'
    salinity_da.attrs['seabird_var_name'] = 'sal00'
    salinity_da.attrs['valid_min'] = float(0)
    salinity_da.attrs['valid_max'] = float(45)
    
    temperature_da.name = 'temperature'
    temperature_da.attrs['long_name'] = 'sea water temperature'
    temperature_da.attrs['standard_name'] = 'sea_water_temperature'
    temperature_da.attrs['units'] = 'degrees_C'
    temperature_da.attrs['coverage_content_type'] = 'physicalMeasurement'
    temperature_da.attrs['seabird_var_name'] = 'tv290C'
    temperature_da.attrs['comments'] = 'ITS-90'
    temperature_da.attrs['valid_min'] = float(-2.2)
    temperature_da.attrs['valid_max'] = float(35)
    
    conductivity_da.name = 'conductivity'
    conductivity_da.attrs['long_name'] = 'sea water electrical conductivity'
    conductivity_da.attrs['standard_name'] = 'sea_water_electrical_conductivity'
    conductivity_da.attrs['units'] = 'S/m'
    conductivity_da.attrs['coverage_content_type'] = 'physicalMeasurement'
    conductivity_da.attrs['seabird_var_name'] = 'cond0S/m'
    conductivity_da.attrs['valid_min'] = float(0)
    conductivity_da.attrs['valid_max'] = float(6)
    
    sound_velocity_da.name = 'sound_velocity'
    sound_velocity_da.attrs['long_name'] = 'speed of sound in sea water'
    sound_velocity_da.attrs['standard_name'] = 'speed_of_sound_in_sea_water'
    sound_velocity_da.attrs['units'] = 'm s-1'
    sound_velocity_da.attrs['coverage_content_type'] = 'physicalMeasurement'
    sound_velocity_da.attrs['seabird_var_name'] = 'svCM'
    sound_velocity_da.attrs['comments'] = 'Chen-Millero'
    density_da.attrs['valid_min'] = float(1405)
    density_da.attrs['valid_max'] = float(1560)
    
    density_da.name = 'density'
    density_da.attrs['long_name'] = 'sea water density'
    density_da.attrs['standard_name'] = 'sea_water_density'
    density_da.attrs['units'] = 'kg m-3'
    density_da.attrs['coverage_content_type'] = 'physicalMeasurement'
    density_da.attrs['seabird_var_name'] = 'density00'
    density_da.attrs['valid_min'] = float(999)
    density_da.attrs['valid_max'] = float(1045)
    
    potential_temperature_da.name = 'potential_temperature'
    potential_temperature_da.attrs['long_name'] = 'sea water potential temperature'
    potential_temperature_da.attrs['standard_name'] = 'sea_water_potential_temperature'
    potential_temperature_da.attrs['units'] = 'degrees_C'
    potential_temperature_da.attrs['coverage_content_type'] = 'physicalMeasurement'
    potential_temperature_da.attrs['seabird_var_name'] = 'potemp090C'
    potential_temperature_da.attrs['comments'] = 'ITS-90'
    potential_temperature_da.attrs['valid_min'] = float(-2.2)
    potential_temperature_da.attrs['valid_max'] = float(35)
    
    # merge together the different xarray DataArray objects
    ctd_ds = xr.merge([pressure_da, temperature_da, conductivity_da, salinity_da, sound_velocity_da, 
                       density_da, potential_temperature_da], combine_attrs='drop_conflicts')

    # clear copied attributes from merge
    ctd_ds.attrs = ''
    
    # ## remove data from above ocean surface (i.e., 0 meters)
    # depth_below = np.where(depth >= 0)
    # depth_below_list = list(depth_below)
    # ctd_profile_ds = ctd_ds.isel(depth=xr.DataArray(depth_below_list, dims=['profile','obs']))
    
    ## add metadata to coords   
    ctd_ds.depth.name = 'depth'
    ctd_ds.depth.attrs['long_name'] = 'depth'
    ctd_ds.depth.attrs['standard_name'] = 'depth'
    ctd_ds.depth.attrs['units'] = 'meters'
    ctd_ds.depth.attrs['positive'] = 'down'
    ctd_ds.depth.attrs['axis'] = 'Z'
    ctd_ds.depth.attrs['coverage_content_type'] = 'coordinate'
    ctd_ds.depth.attrs['seabird_var_name'] = 'depSM'
    ctd_ds.depth.attrs['valid_min'] = float(0)
    ctd_ds.depth.attrs['valid_max'] = float(3000)
    
    ## add coordinate for depth correction for years 2018 and 2019
    if '2020' not in netcdf_filename: # correct depth for 2018 and 2019
        ctd_ds = ctd_ds.assign_coords({'depth_correction': ("depth", ctd_ds.depth.values+33)})
        ## add attributes to describe new coordinate
        ctd_ds.depth_correction.name = 'depth_correction'
        ctd_ds.depth_correction.attrs['long_name'] = 'corrected depth'
        ctd_ds.depth_correction.attrs['standard_name'] = 'depth'
        ctd_ds.depth_correction.attrs['units'] = 'meters'
        ctd_ds.depth_correction.attrs['positive'] = 'down'
        ctd_ds.depth_correction.attrs['coverage_content_type'] = 'coordinate'
        ctd_ds.depth_correction.attrs['seabird_var_name'] = 'depSM'
        ctd_ds.depth_correction.attrs['valid_min'] = float(0)
        ctd_ds.depth_correction.attrs['valid_max'] = float(3000)
        ctd_ds.depth_correction.attrs['comment'] = 'Additional depth coordinate that includes a correction of +33 meters.'

    ## add lat/lon and start time coordinates
    ctd_ds = ctd_ds.assign_coords({'latitude': ("profile", [lat])})
    ctd_ds = ctd_ds.assign_coords({'longitude': ("profile", [lon])})
    ctd_ds = ctd_ds.assign_coords({'time': ("profile", [start_time_dt64])})
    
    ## add metadata
    ctd_ds.latitude.name = 'latitude'
    ctd_ds.latitude.attrs['long_name'] = 'latitude'
    ctd_ds.latitude.attrs['standard_name'] = 'latitude'
    ctd_ds.latitude.attrs['units'] = 'degrees_north'
    ctd_ds.latitude.attrs['coverage_content_type'] = 'coordinate'
    ctd_ds.latitude.attrs['axis'] = 'Y'
    ctd_ds.latitude.attrs['valid_max'] = float(90.0)
    ctd_ds.latitude.attrs['valid_min'] = float(-90.0)
    ctd_ds.latitude.attrs['comments'] = 'Latitude of CTD location.'
    
    ctd_ds.longitude.name = 'longitude'
    ctd_ds.longitude.attrs['long_name'] = 'longitude'
    ctd_ds.longitude.attrs['standard_name'] = 'longitude'
    ctd_ds.longitude.attrs['units'] = 'degrees_east'
    ctd_ds.longitude.attrs['coverage_content_type'] = 'coordinate'
    ctd_ds.longitude.attrs['axis'] = 'X'
    ctd_ds.longitude.attrs['valid_max'] = float(180.0)
    ctd_ds.longitude.attrs['valid_min'] = float(-180.0)
    ctd_ds.longitude.attrs['comments'] = 'Longitude of CTD location.'
    
    ctd_ds.time.name = 'time'
    ctd_ds.time.attrs['long_name'] = 'time'
    ctd_ds.time.attrs['standard_name'] = 'time'
    ctd_ds.time.attrs['axis'] = 'T'
    ctd_ds.time.attrs['coverage_content_type'] = 'coordinate'
    ctd_ds.time.attrs['comment'] = 'Time at which the CTD was deployed.'
    
    return(ctd_ds)

## Function add_metadata() to add global attributes

def add_metadata(ctd_profile_ds, uuid, seafloor_depth, cast_depth, cast_id, netcdf_filename, sample_interval_iso, end_time):
    # get sampling duration
    start_time = ctd_profile_ds.time[0].values
    tdelta = pd.Timedelta(end_time - start_time).isoformat()
    
    ## add attributes to dataset
    ctd_profile_ds.attrs['title'] = 'OMG Narwhals Ocean CTD Level 2 Data'
    ctd_profile_ds.attrs['summary'] = 'This dataset contains conductivity, temperature, and pressure measurements from a ship-deployed CTD instrument. It also contains derived variables: salinity, sound velocity, density, and potential temperature. This profile is one of a series of CTD casts from the Oceans Melting Greenland (OMG) Narwhals program. Between August 2018 to August 2020, three bottom-mounted ocean moorings with a suite of instrumentation were deployed at three glacial fronts in Melville Bay: Sverdrup Glacier, Kong Oscar Glacier, and Rink Glacier. During summer cruises where moorings were deployed and/or recovered, full water column CTD profiles were obtained at the glacier fronts and offshore.'
    ctd_profile_ds.attrs['Conventions'] = 'CF-1.8, ACDD-1.3'
    ctd_profile_ds.attrs['keywords'] = 'Conductivity, Salinity, Water Depth, Water Temperature'
    ctd_profile_ds.attrs['keywords_vocabulary'] = 'NASA Global Change Master Directory (GCMD) Science Keywords'
    ctd_profile_ds.attrs['standard_name_vocabulary'] = 'NetCDF Climate and Forecast (CF) Metadata Convention'
    ctd_profile_ds.attrs['id'] = 'OMG_Narwhals_Ocean_CTD_L2'
    ctd_profile_ds.attrs['featureType'] = 'profile'
    ctd_profile_ds.attrs['cdm_data_type'] = "Station"
    ctd_profile_ds.attrs['cast_id'] = cast_id
    ctd_profile_ds.attrs['uuid'] = uuid
    ctd_profile_ds.attrs['platform'] = 'R/V Sanna'
    ctd_profile_ds.attrs['region'] = 'Melville Bay, northwest Greenland'
    ctd_profile_ds.attrs['seafloor_depth'] = seafloor_depth
    ctd_profile_ds.attrs['cast_depth'] = cast_depth
    ctd_profile_ds.attrs['filename'] = netcdf_filename
    ctd_profile_ds.attrs['serial_number'] = '1906981'
    ctd_profile_ds.attrs['instrument'] = 'SBE19plus'
    ctd_profile_ds.attrs['history'] = "Transformed processed *.cnv files that were converted from the instrument's output *.hex file."
    ctd_profile_ds.attrs['source'] = 'Conductivity, Temperature, and Depth (CTD) data collected from a ship-deployed CTD instrument.'
    ctd_profile_ds.attrs['processing_level'] = 'L2'
    ctd_profile_ds.attrs['acknowledgement'] = "This research was carried out by the Jet Propulsion Laboratory, managed by the California Institute of Technology under a contract with the National Aeronautics and Space Administration, the University of Washington's Applied Physics Laboratory and School of Aquatic and Fishery Sciences, and the Greenland Institute of Natural Resources."
    ctd_profile_ds.attrs['license'] = 'Public Domain'
    ctd_profile_ds.attrs['product_version'] = '1.0'
    # ctd_profile_ds.attrs['references'] = '' # DOI number
    ctd_profile_ds.attrs['creator_name'] = 'Marie J. Zahn'
    ctd_profile_ds.attrs['creator_email'] = 'mzahn@uw.edu'
    ctd_profile_ds.attrs['creator_type'] = 'person'
    ctd_profile_ds.attrs['creator_institution'] = 'University of Washington'
    ctd_profile_ds.attrs['institution'] = 'University of Washington'
    ctd_profile_ds.attrs['project'] = 'Oceans Melting Greenland (OMG) Narwhals'
    ctd_profile_ds.attrs['contributor_name'] = 'Marie J. Zahn, Kristin L. Laidre, Malene J. Simon, Ian Fenty'
    ctd_profile_ds.attrs['contributor_role'] = "author, principal investigator, co-investigator, co-investigator" 
    ctd_profile_ds.attrs['contributor_email'] = 'mzahn@uw.edu; klaidre@uw.edu; masi@natur.gl; ian.fenty@jpl.nasa.gov'
    ctd_profile_ds.attrs['naming_authority'] = 'gov.nasa.jpl'
    ctd_profile_ds.attrs['program'] = 'NASA Earth Venture Suborbital-2 (EVS-2) and Office of Naval Research (ONR) Marine Mammals and Biology Program'
    ctd_profile_ds.attrs['publisher_name'] = 'Physical Oceanography Distributed Active Archive Center (PO.DAAC)'
    ctd_profile_ds.attrs['publisher_institution'] = 'NASA Jet Propulsion Laboratory (JPL)'
    ctd_profile_ds.attrs['publisher_email'] = 'podaac@podaac.jpl.nasa.gov'
    ctd_profile_ds.attrs['publisher_url'] = 'https://podaac.jpl.nasa.gov/'
    ctd_profile_ds.attrs['publisher_type'] = 'group'
    ctd_profile_ds.attrs['geospatial_lat_min'] = ctd_profile_ds.latitude.values[0]
    ctd_profile_ds.attrs['geospatial_lat_max'] = ctd_profile_ds.latitude.values[0]
    ctd_profile_ds.attrs['geospatial_lat_units'] = "degrees_north"
    ctd_profile_ds.attrs['geospatial_lon_min'] = ctd_profile_ds.longitude.values[0]
    ctd_profile_ds.attrs['geospatial_lon_max'] = ctd_profile_ds.longitude.values[0]
    ctd_profile_ds.attrs['geospatial_lon_units'] = "degrees_east"
    ctd_profile_ds.attrs['geospatial_vertical_min'] = ctd_profile_ds.depth_correction.values.min()
    ctd_profile_ds.attrs['geospatial_vertical_max'] = ctd_profile_ds.depth_correction.values.max()
    ctd_profile_ds.attrs['geospatial_vertical_units'] = 'meters'
    ctd_profile_ds.attrs['geospatial_vertical_positive'] = 'down'
    ctd_profile_ds.attrs['time_coverage_resolution'] = sample_interval_iso  
    ctd_profile_ds.attrs['time_coverage_start'] = str(start_time)[:-10]
    ctd_profile_ds.attrs['time_coverage_end'] = str(end_time)[:-10]
    ctd_profile_ds.attrs['time_coverage_duration'] = tdelta
    ctd_profile_ds.attrs['date_created'] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    
    # with xr.set_options(display_style="html"):
    #     display(ctd_profile_ds)

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