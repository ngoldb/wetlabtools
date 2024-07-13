"""
Tecan module
"""

import openpyxl
from wetlabtools.tecan.parse import parse_header, parse_data
from wetlabtools.tecan.plotting import plot_96_well_plate

class experiment(object):
    """Base class for tecan experiments"""

    def __init__(self, name: str, data: str):
        """
        name: str, name of the experiment
        data: str, path to the tecan excel file
        """
        self.name = name
        self.data_file = data
        self.metadata, self.protocol = parse_header(self.data_file)
        self.data = self.__parse_all_data()
    
    def __parse_all_data(self):
        all_data = {}
        implemented_types = ['Absorbance']

        if 'Kinetic' in self.protocol: 
            kinetic = True
        else:
            kinetic = False

        for action in self.protocol:
            if action in implemented_types:
                x = self.protocol[action]

                if "Reference wavelength [nm]" in x:
                    identifier = 'Difference'
                else:
                    identifier = '<>'

                subset= self.protocol[action]['Part of Plate']

                data = parse_data(self.data_file, identifier=identifier, subset=subset)
                all_data[action] = data
        
        return all_data
    

    def show_plate_data(self, data_id: str=None):
        '''
        data_id: str, identifier of the data to display

        Function to display a 96 well plate with the wells colored according to their measured value
        '''

        df = self.data[data_id]
        fig, ax = plot_96_well_plate(df, title=data_id)

        return fig, ax

    # getters
    def get_name(self):
        return self.name
    
    def get_metadata(self):
        return self.metadata
    
    def get_protocol(self):
        return self.protocol
    
    def get_data_file(self):
        return self.data_file
    
    def get_data(self):
        return self.data