"""
Tecan module
"""
import openpyxl
from wetlabtools.tecan.parse import parse_header, parse_data

class experiment(object):
    """Base class for tecan experiments"""

    def __init__(self, name: str, data: str, plate: str):
        """
        name: str, name of the experiment
        data: str, path to the tecan excel file
        plate: str, path to the plate map of the experiment
        """
        self.name = name
        self.data = data
        self.plate = plate
        self.metadata, self.protocol = parse_header(self.data)
        self.data = self.__parse_all_data()
    
    def __parse_all_data(self):
        all_data = {}
        implemented_types = ['Absorbance']
        protocol = self.protocol

        if 'Kinetic' in protocol: 
            kinetic = True
        else:
            kinetic = False

        for action in protocol:
            if action in implemented_types:
                x = protocol[action]

                if "Reference wavelength [nm]" in x:
                    reference = True
                else:
                    reference = False

                data = parse_data()
                all_data[action] = data
        
        return all_data
    

    # getters
    def get_name(self):
        return self.name
    
    def get_metadata(self):
        return self.metadata
    
    def get_protocol(self):
        return self.protocol
    
    def get_data(self):
        return self.data