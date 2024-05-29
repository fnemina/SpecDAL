# readers.py provides functions to read .asd spectrum files for data and
# metadata.


import pandas as pd
import numpy as np
from os.path import abspath, expanduser, splitext, basename, join, split
import glob
from collections import OrderedDict
import json
import struct
import datetime

ASD_VERSIONS = ['ASD', 'asd', 'as6', 'as7', 'as8']
ASD_HAS_REF = {'ASD': False, 'asd': False, 'as6': True, 'as7': True,
               'as8': True}
ASD_DATA_TYPES = OrderedDict([("RAW_TYPE", "tgt_count"),
                              ("REF_TYPE", "tgt_reflect"),
                              ("RAD_TYPE", "tgt_radiance"),
                              ("NOUNITS_TYPE", None),
                              ("IRRAD_TYPE", "tgt_irradiance"),
                              ("QI_TYPE", None),
                              ("TRANS_TYPE", None),
                              ("UNKNOWN_TYPE", None),
                              ("ABS_TYPE", None)])

ASD_DATA_TYPES_MANUAL = {0:"RAW_TYPE", 1:"REF_TYPE", 2:"RAD_TYPE", 3:"NOUNITS_TYPE", 4:"IRRAD_TYPE", 5:"QI_TYPE",
                              6:"TRANS_TYPE", 7:"UNKNOWN_TYPE", 8:"ABS_TYPE"}

ASD_GPS_DATA = struct.Struct("= 5d 2b cl 2b 5B 2c")

ASD_DATA_FORMAT = {0:"FLOAT_FORMAT", 1:"INTEGER_FORMAT", 2:"DOUBLE_FORMAT",3:"UNKNOWN_FORMAT"}

ASD_INSTRUMENT_TYPE = {0:"UNKNOWN_INSTRUMENT", 1:"PSII_INSTRUMENT", 2:"LSVNIR_INSTRUMENT", 3:"FSVNIR_INSTRUMENT",
                       4:"FSFR_INSTRUMENT", 5:"FSNIR_INSTRUMENT", 6:"CHEM_INSTRUMENT", 7:"FSFR_UNATTENDED_INSTRUMENT",}

# ASD Metadata structre for ASD version 8 first 484 bytes
METADATA_ASD = struct.Struct("< 3s 157s 9h 4c L 1c L 2f 5c H 128s 5d 2b cl 2b 5B 2c L 2h 2H 4f H c 4s 3H c L 4H 2f i 3f h c 2f 5c")

def read_asd(filepath, read_data=True, read_metadata=True, verbose=False):
    """
    Read asd file for data and metadata
    
    Return
    ------
    2-tuple of (pd.DataFrame, OrderedDict) for data, metadata
    """
    data = None
    metadata = None
    if read_metadata:
        metadata = OrderedDict()
    raw_metadata = {}
    with open(abspath(expanduser(filepath)), 'rb') as f:
        if verbose:
            print('reading {}'.format(filepath))
        binconts = f.read()
        version = binconts[0:3].decode('utf-8')
        assert(version in ASD_VERSIONS) # TODO: define ASD_VERSIONS
        # read spectrum type
        spectrum_type_index = struct.unpack('B', binconts[186:(186 + 1)])[0]
        spectrum_type = list(ASD_DATA_TYPES.keys())[spectrum_type_index]
        # read wavelength info
        wavestart = struct.unpack('f', binconts[191:(191 + 4)])[0]
        wavestep = struct.unpack('f', binconts[195:(195 + 4)])[0] # in nm
        num_channels = struct.unpack('h', binconts[204:(204 + 2)])[0]
        wavestop = wavestart + num_channels*wavestep - 1
        if read_data:
            # read data
            tgt_column = ASD_DATA_TYPES[spectrum_type]
            ref_column = tgt_column.replace('tgt', 'ref')
            data_format = struct.unpack('B', binconts[199:(199 + 1)])[0]
            fmt = 'f'*num_channels
            if data_format == 2:
                fmt = 'd'*num_channels
            if data_format == 0:
                fmt = 'f'*num_channels
            # data to DataFrame
            size = num_channels*8
            # Read the spectrum block data
            waves = np.linspace(wavestart, wavestop, num_channels)
            spectrum = np.array(struct.unpack(fmt, binconts[484:(484 + size)]))
            reference = None
            if ASD_HAS_REF[version]:
                # read reference
                start = 484 + size
                ref_flag = struct.unpack('??', binconts[start: start + 2])[0]
                first, last = start + 18, start + 20
                ref_desc_length = struct.unpack('H', binconts[first:last])[0]
                first = start + 20 + ref_desc_length
                last = first + size
                reference = np.array(struct.unpack(fmt, binconts[first:last]))
            data = pd.DataFrame({tgt_column : spectrum,
                                 ref_column: reference}, index=waves)
            data.index.name = 'wavelength'
            data.dropna(axis=1, how='all')
        if read_metadata:
            metadata_unpacked = METADATA_ASD.unpack(binconts[:484])
            # First
            metadata['file'] = f.name
            metadata['instrument_manufacturer'] = metadata_unpacked[0].decode('ascii')
            metadata['comments'] = metadata_unpacked[1].replace(b'\x00', b'').decode('ascii')
            # date
            time_lock = 160
            second, minute, hour, day, month, year, wday, yday, isdst = METADATA_ASD.unpack(binconts[:484])[2:2+9]
            month = month+1
            year = year+1900
            metadata['measurement_date'] = f"{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02}"
            # version
            metadata['version'] = metadata_unpacked[11]
            metadata['file_version'] = metadata_unpacked[12]
            # Dark current
            metadata['dc_corr'] = metadata_unpacked[14]
            metadata['dc_time'] = metadata_unpacked[15]
            metadata['dc_time'] = datetime.datetime.fromtimestamp(metadata['dc_time']).strftime('%Y-%m-%d %H:%M:%S')
            # Data type
            metadata['data_type'] = ASD_DATA_TYPES_MANUAL[int.from_bytes(metadata_unpacked[16])]
            # WR
            metadata['wr_time'] = metadata_unpacked[17]
            metadata['wr_time'] = datetime.datetime.fromtimestamp(metadata['wr_time']).strftime('%Y-%m-%d %H:%M:%S')
            # Channels
            metadata["ch1_wave"] = metadata_unpacked[18]
            metadata["wave_step"] = metadata_unpacked[19]
            metadata["data_format"] = ASD_DATA_FORMAT[int.from_bytes(metadata_unpacked[20], byteorder='little')]
            metadata["channels"] = metadata_unpacked[25]
            # GPS info
            gps_struct = metadata_unpacked[27:27+18]
            gps_true_heading, gps_speed, gps_latitude, gps_longitude, gps_altitude = gps_struct[:5]
            gps_flags = gps_struct[5:7] 
            gps_hardware_mode = gps_struct[7]
            gps_timestamp = gps_struct[8]
            gps_flags2 = gps_struct[9:11] # unpack this into bits
            gps_satellites = gps_struct[11:16]
            gps_filler = gps_struct[16:18]
            
            metadata['gps_true_heading'] = gps_true_heading
            metadata['gps_speed'] = gps_speed
            metadata['gps_latitude'] = gps_latitude
            metadata['gps_longitude'] = gps_longitude
            metadata['gps_altitude'] = gps_altitude
            metadata['gps_flags'] = gps_flags
            metadata['gps_hardware_mode'] = gps_hardware_mode
            metadata['gps_timestamp'] = gps_timestamp
            metadata['gps_flags2'] = gps_flags2
            metadata['gps_satellites'] = gps_satellites
            metadata['gps_filler'] = gps_filler
            
            # Intrument config
            metadata['integration_time'] = metadata_unpacked[45]
            metadata['fore_optics'] = metadata_unpacked[46]
            metadata['dark_current_correction'] = metadata_unpacked[47]

            metadata['calibration_series'] = metadata_unpacked[48]
            metadata['instrument_number'] = metadata_unpacked[49]

            metadata['y-range'] = (metadata_unpacked[50], metadata_unpacked[51])
            metadata['x-range'] = (metadata_unpacked[52], metadata_unpacked[53])   

            metadata['dynamic_range'] = metadata_unpacked[54]
            metadata['xmode'] = metadata_unpacked[55]

            metadata['flags'] = metadata_unpacked[56]

            metadata['dc_count'] = metadata_unpacked[57]
            metadata['wr_count'] = metadata_unpacked[58]
            metadata['sample_count'] = metadata_unpacked[59]

            metadata['instrument_type'] = ASD_INSTRUMENT_TYPE[int.from_bytes(metadata_unpacked[60], byteorder='little')]

            metadata['bulb_id'] = metadata_unpacked[61]

            metadata['swir1_gain'] = metadata_unpacked[62]
            metadata['swir2_gain'] = metadata_unpacked[63]
            metadata['swir1_offset'] = metadata_unpacked[64]
            metadata['swir2_offset'] = metadata_unpacked[65]

            metadata['splice1_wavelength'] = metadata_unpacked[66]
            metadata['splice2_wavelength'] = metadata_unpacked[67]
            
    return data, metadata
