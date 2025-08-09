"""
Tecan module
"""

from wetlabtools.tecan.parse import parse_header, parse_data
from wetlabtools.tecan.plotting import plot_96_well_plate

class experiment(object):
    """Base class for tecan experiments"""

    def __init__(self, name: str, data: str, plate_format: int=96):
        """
        name: str, name of the experiment
        data: str, path to the tecan excel file
        plate_format: int, number of wells of the microplate
        """
        self.name = name
        self.data_file = data
        self.plate_format = plate_format
        self.metadata, self.protocol = parse_header(self.data_file, plate_format)
        self.data = self.__parse_all_data()
    
    def __parse_all_data(self):
        all_data = {}
        implemented_types = ['Absorbance', 'Fluorescence Top Reading', 'Luminescence']

        for action in self.protocol:
            skip = 0
            if action not in implemented_types:
                if action == 'Plate': pass
                else: print(f"data type not implemented: {action}")\
                
            elif action == 'Absorbance' or action == 'Fluorescence Top Reading' or action == 'Luminescence':

                if "Reference wavelength [nm]" in self.protocol[action]:
                    identifier = 'Difference'
                else:
                    identifier = '<>'
                    if 'plate_layout' in self.protocol['Plate']: skip = 1

                subset= self.protocol[action]['Part of Plate']

                try:
                    label = self.protocol[action]['label']
                except:
                    label = 'value'

                data = parse_data(
                    self.data_file, 
                    identifier=identifier, 
                    subset=subset, 
                    label=label, 
                    skip=skip, 
                    plate=self.plate_format
                )
                all_data[action] = data
        
        return all_data
    

    def show_plate_data(self, data_id: str=None, **kwargs):
        '''
        data_id: str, identifier of the data to display

        Function to display a 96 well plate with the wells colored according to their measured value
        '''

        df = self.data[data_id]
        fig, ax = plot_96_well_plate(df, title=data_id, **kwargs)

        return fig, ax

    # getters
    def get_name(self):
        return self.name

    def get_format(self):
        return self.plate_format
    
    def get_metadata(self):
        return self.metadata
    
    def get_protocol(self):
        return self.protocol
    
    def get_data_file(self):
        return self.data_file
    
    def get_data(self):
        return self.data
    

class elisa(experiment):
    """
    Class for elisa measurements on the Tecan
    """

    def __init__(self, name: str, data: str, exp_layout: str):
        super().__init__(name, data)
        self.exp_layout = exp_layout
        self.coating, self.samples, self.concentrations = self.parse_layout()

    def parse_layout(self):
        action = self.protocol[0]
        subset= self.protocol[action]['Part of Plate']
        coating = parse_data(self.exp_layout, "Coating", subset=subset)
        samples = parse_data(self.exp_layout, "Samples", subset=subset)
        concentrations = parse_data(self.exp_layout, "Concentrations", subset=subset)

        return coating, samples, concentrations