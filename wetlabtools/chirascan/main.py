"""
Main submodule for processing CD data from Chirascan instruments
"""

import pandas as pd
from wetlabtools.chirascan.parse import parse_header, parse_data, get_sample_params
from wetlabtools.chirascan.process import data2df

class cd_experiment(object):
    """
    Base class for cd experiments on chirascan
    """

    def __init__(self, result_file, sample_data: dict=None):
        self.data_file = result_file
        
        self.cells, self.dimensions, self.properties = parse_header(self.data_file)
        self.data = parse_data(self.data_file, 
                               dimensions=self.dimensions,
                               properties=self.properties)
        self.data = data2df(self.data, self.dimensions, self.cells)

        self.sample_data = pd.DataFrame.from_dict(self.cells, orient='index').reset_index()
        self.sample_data['Cell'] = self.sample_data['index'].apply(lambda x: x[-1])
        self.sample_data.drop('index', axis=1, inplace=True)

        if sample_data:
            user_sample_data = pd.DataFrame.from_dict(sample_data, orient='index').reset_index()
            user_sample_data['Cell'] = user_sample_data['index'].apply(lambda x: x[-1])
            user_sample_data.drop('index', inplace=True, axis=1)

            # merge sample data 
            self.sample_data = pd.merge(user_sample_data,
                                        self.sample_data,
                                        on='Cell',
                                        how='outer',
                                        suffixes=('', '_drop'))
            
            # keep input from user
            self.sample_data = self.sample_data.loc[:, ~self.sample_data.columns.str.endswith('_drop')]

        # now add sample data to data
        for data in self.data:
            self.data[data] = pd.merge(self.data[data], self.sample_data, on='Cell')

        # make sure wavelength is of type int
        self.data["CircularDichroism"]['Wavelength'] = self.data["CircularDichroism"]['Wavelength'].astype(int)

        # record data format of cd signal
        self.cd_unit = 'mdeg'

    def convert(self, unit: str='mre'):
        IMPLEMENTED = ['mre', 'mrex103']

        if unit.casefold() not in IMPLEMENTED:
            raise NotImplementedError(IMPLEMENTED)
        
        if unit.casefold() in ['MRE', 'MREx103']:
            self.data['CircularDichroism']
            self.data['CircularDichroism']['value'] = self.data['CircularDichroism'].value / (self.data['CircularDichroism'].pathlength * self.data['CircularDichroism'].conc * self.data['CircularDichroism'].n_pep * 10 ** -6)
            
            if unit.casefold() == 'MREx103':
                self.data['CircularDichroism']['value'] = self.data['CircularDichroism']['value'] * 1e-3
                self.cd_unit = 'MREx103' 
            else:   
                self.cd_unit = 'MRE'

        print(f'converted to {unit}')

    def subtract_blank(self, blank_file: str):
        """
        buffer_file: str, path to the buffer data file

        Subtract CD signal of buffer from sample.
        """
        c, d, p = parse_header(blank_file)
        self.blank_data = parse_data(blank_file, d, p)
        self.blank_data = data2df(self.blank_data, d, c)

        # calculate mean
        if 'Repeat' in self.data['CircularDichroism'].columns:
            self.blank_data['CircularDichroism'] = self.blank_data['CircularDichroism'].groupby(['Wavelength', 'Cell'], as_index=False)['value'].mean()

        # subtract buffer blank
        df = pd.merge(self.data['CircularDichroism'], self.blank_data['CircularDichroism'], on='Wavelength', suffixes=('', '_buffer'))
        df['value'] = df['value'] - df['value_buffer']
        df.drop([column for column in df.columns if '_buffer' in column], axis=1, inplace=True)
        self.data['CircularDichroism'] = df
        print('subtracted blank')


    # getters
    def get_cells(self):
        return self.cells
    
    def get_dimensions(self):
        return self.dimensions
    
    def get_properties(self):
        return self.properties
    
    def get_data(self, cell: str=None):
        if cell:
            data = dict()
            for prop in self.data:
                df = self.data[prop]
                data[prop] = df[df['Cell'] == str(cell)]
        else:
            data = self.data

        return data
    
    def get_samples(self):
        return self.sample_data
    
    def get_blank_data(self):
        try:
            return self.blank_data
        except AttributeError as err:
            raise Exception('blank data not available - use subtract_blank() to add blank data') from err
        
    def get_cd_unit(self):
        return self.cd_unit