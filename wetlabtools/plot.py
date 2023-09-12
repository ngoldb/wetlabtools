"""
Tools to easily plot experimental data:
- CD spectra
- SEC-MALS
- FPLC chromatograms
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# =======================================
# CD DATA
# =======================================
def load_CD_melt(data_file: str):
    """
    :param: fata_file: str, path to the ProData csv file

    Function to read ProData csv file and process the data. Will return a dictonary containing CD, HV and Absorbance data. Each data block is stored in dicts.
    If a path to a buffer csv file is provided, the background will automatically be subtracted from the CD data
    """
    
    # initialize variables
    properties = ['CircularDichroism' , 'HV', 'Absorbance']
    data_dict = {}
    samples = {}
    data = False
    wv = False
    prop = ''

    # open file and read content
    with open(data_file, 'r') as fobj:
        for line in fobj:

            # start in the header and collect meta data
            # collect samples
            if '#Sample' in line.strip() and not data:
                sample = line.strip().split(' ')[-1]
                sample_id = int(line.strip().split(' ')[0][-2])
                samples[sample_id] = sample

            # collect wavelengths measured
            if 'Wavelength,Wavelength:' in line.strip() and not wv and not data:
                wv_high = line.strip().split(',')[1].split(' ')[3].strip('nm')
                wv_low  = line.strip().split(',')[1].split(' ')[1].strip('nm')
                step    = int(line.strip().split(',')[2].split(' ')[-1].strip('nm'))

                # create a list of all wavelengths measured
                wv_len = np.linspace(int(wv_low), 
                                     int(wv_high),
                                     num=(int(wv_high) - int(wv_low)) // step + 1,
                                     dtype=int
                                    )
                wv = True

            # now entering data part of ProData file
            if line.strip() == 'Data:':
                data = True

            # now reading data from block
            if len(samples) == 1 and data and wv:
                # we know that there is only one sample
                tmp_data = {}
                sample = samples[1]
                
                # collect the data from the different properties
                if len(properties) != 0 and properties[0] in line.strip() and data:
                    # initialize dictonary for property and samples
                    prop = properties[0]
                    properties.pop(0)
                    data_dict[prop] = {}
                    line = next(fobj)
                    line = next(fobj)
                    
                    # get column names (temperatures)
                    columns = line.strip().strip(',').split(',')
                    
                    # iterate over the available wave lengths
                    for _, _ in enumerate(wv_len[::-1]):
                        line = next(fobj)
                        m_wv = line.strip().split(',')[0]
                        dataline = line.strip().split(',')[1:]
                        tmp_data[m_wv] = dataline
                    
                    # now construct Data Frame
                    df_temp = pd.DataFrame.from_dict(tmp_data, 
                                orient='index',
                                columns=columns
                               )

                    data_dict[prop][sample] = df_temp

            elif len(samples) > 1 and data and wv:
                # collect the data from the different properties
                if len(properties) != 0 and properties[0] in line.strip() and data:
                    # initialize dictonary for property and samples
                    prop = properties[0]
                    properties.pop(0)
                    data_dict[prop] = {}
                
                if 'Cell:,' in line.strip() and prop != '':

                    # clear data
                    tmp_data = {}
                    sample = samples[int(line.strip().split(',')[-1])]

                    line = next(fobj)
                    line = next(fobj)

                    # get column names (temperatures)
                    columns = line.strip().strip(',').split(',')

                    # iterate over the available wave lengths
                    for _, _ in enumerate(wv_len[::-1]):
                        line = next(fobj)
                        m_wv = line.strip().split(',')[0]
                        dataline = line.strip().split(',')[1:]
                        tmp_data[m_wv] = dataline

                    # now construct Data Frame
                    df_temp = pd.DataFrame.from_dict(tmp_data, 
                                orient='index',
                                columns=columns
                               )

                    data_dict[prop][sample] = df_temp
    
    return data_dict



def get_buffer_CD(buffer_data: str):
    """
    :param: buffer_data: str, path to the csv with buffer CD data
    Function to read in CD signal from buffer background. Returns a Data Frame.
    """
    # Initialize variables
    wv = False
    data = False

    with open(buffer_data, 'r') as fobj:
        for line in fobj:

            # collect wavelengths measured
            if 'Wavelength,Wavelength:' in line.strip() and not wv and not data:
                wv_high = line.strip().split(',')[1].split(' ')[3].strip('nm')
                wv_low  = line.strip().split(',')[1].split(' ')[1].strip('nm')
                step    = int(line.strip().split(',')[2].split(' ')[-1].strip('nm'))

                # create a list of all wavelengths measured
                wv_len = np.linspace(int(wv_low), 
                                     int(wv_high),
                                     num=(int(wv_high) - int(wv_low)) // step + 1,
                                     dtype=int
                                    )
                wv = True

            if line.strip() == 'Data:':
                data = True

            if 'CircularDichroism' in line.strip() and wv and data:
                tmp_data = {}

                for _ in wv_len:
                    line = next(fobj)
                    m_wv = line.strip().split(',')[0]
                    dataline = line.strip().split(',')[1:]
                    tmp_data[m_wv] = dataline

                # now construct Data Frame
                df_temp = pd.DataFrame.from_dict(tmp_data, 
                            orient='index'
                           )
    return df_temp