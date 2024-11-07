"""
This module contains code to parse data from csv files exported 
from Chirascan CD spectrometers. 
"""

import Bio


def parse_header(file):
    """
    file: str, path to the csv file

    Function to parse header and return meta data from csv file
    """
    cells = dict()
    dimensions = dict()
    properties = list()

    # parse header of data
    with open(file, "r") as fobj:
        iter_fobj = iter(fobj)
        for line in iter_fobj:
            line = line.strip()

            if 'Data:' in line:
                break
            
            # reading in cell and sample data
            if '#Cell' in line:
                cell_id = line.strip().split(' ')[0].strip('#')
                if cell_id not in cells.keys():
                    cells[cell_id] = dict()
                
                #if 'type' in line:
                #    cells[cell_id]['type'] = line.strip().split(':')[-1]
                
                if 'Pathlength' in line:
                    cells[cell_id]['pathlength'] = int(line.strip().split(' ')[-2])

            if '#Sample' in line:
                sample_id = line.strip().split(' ')[0][-2]
                cell_id = 'Cell' + sample_id
                if cell_id not in cells.keys():
                    cells[cell_id] = dict()
                cells[cell_id]['sample_id'] = line.strip().split(' ')[-1]

            # reading in dimensions
            if 'Available Dimensions' in line:
                n_dim = int(line.split(',')[-1])
                
                for x in range(n_dim):
                    line = next(iter_fobj).strip()
                    if 'Warning' in line:
                        for i in range(3):
                            line = next(iter_fobj)

                    dimension = line.split(',')[0]
                    if dimension == 'Wavelength':
                        single_wvl = False
                        wvl, step, bandwith = line.split(',')[1:]
                        try: step = int(step.split(' ')[-1].strip('nm'))
                        except ValueError: 
                            step = 0
                            single_wvl = True
                        bandwith = int(bandwith.split(' ')[-1].strip('nm'))
                        dimensions[dimension] = {'step': step,
                                                 'bandwith': bandwith,
                                                 'single_wvl': single_wvl}
                        
                        if single_wvl:
                            dimensions[dimension]['wavelength'] = int(wvl.split(' ')[-1].strip('nm'))
                        else:
                            dimensions[dimension]['start'] = wvl.split(' ')[1].strip().strip('nm')
                            dimensions[dimension]['end'] = wvl.split(' ')[3].strip().strip('nm')

                    elif dimension == 'Repeat' or dimension == 'Cell':
                        dimensions[dimension] = int(line.split(',')[1][0])
                    
                    elif dimension == 'Temperature':
                        dimensions[dimension] = int(line.split(',')[1].split(' ')[0])
            
            # reading properties
            if 'Available Properties' in line:
                n_properties = int(line.split(',')[-1])
                for i in range(n_properties):
                    line = next(iter_fobj)
                    properties.append(line.split(',')[0])

    return cells, dimensions, properties


#######################
# reading fasta file data
def get_sample_params(fasta_file, concentrations):
    '''
    fasta_file: str, path to the fasta file
    concentrations: dict, sample names (as in fasta) as keys and concentrations (in µM) as values
    Note: concentrations can be a single value too if all samples have same concentration

    Function to parse sample data from fasta file
    '''

    sample_data = dict()
    if type(concentrations) != dict:
        conc = concentrations
        concentrations = dict()
        for record in Bio.SeqIO.parse(fasta_file, 'fasta'):
            concentrations[record.id] = conc

    for record in Bio.SeqIO.parse(fasta_file, 'fasta'):
            prot_param = Bio.SeqUtils.ProtParam.ProteinAnalysis(record.seq)
            sample_data[record.id] = {
                'n_pep': len(record.seq) - 1,
                'conc': concentrations[record.id],
                'mw': round(prot_param.molecular_weight(), 1)
                }
            
    return sample_data


#######################
# retrieve data for further processing
def parse_data(data_file: str, dimensions: dict, properties: list, data_names: list=['CircularDichroism', 'HV', 'Absorbance']):
    """
    data_file: str, path to the csv file
    dimensions: dict, from parse_header()
    properties: list, from parse_header()
    data_names: list, list of strings describing which data to fetch

    Function to parse data from a csv file from Chirascan CD spectrometer.
    Will return the data for further processing. Not converted to df yet!
    """
    data = dict()

    if len(dimensions) > 1:
        with open(data_file, 'r') as fobj:
            data_section = False
            iter_fobj = iter(fobj)
            
            for line in iter_fobj:
                line = line.strip()

                # skip anything until we reach the data section
                if 'Data:' not in line:
                    pass
                else:
                    data_section = True
                
                # start to fetch data
                if line in data_names and data_section:
                    data_name = line
                    data[data_name] = list()
                    line = next(iter_fobj)
                    line = line.strip()
                    
                    while line not in properties:

                        # get the dimensions of current data block
                        curr_data_dim = dict()
                        while line.split(',')[0].strip(':') in dimensions.keys():
                            if all([x in dimensions.keys() for x in line.split(',')]):
                                curr_data_dim['ax'] = line.split(',')
                            else:
                                curr_data_dim[line.split(',')[0]] = line.split(',')[1]
                            line = next(iter_fobj)
                            line = line.strip()

                        # collect data lines
                        lines = ''
                        while line.split(':')[0].strip(':') not in dimensions.keys():
                            lines += line + '\n'
                            line = next(iter_fobj)
                            if line == '\n':
                                break
                            line = line.strip()

                        data[data_name].append([curr_data_dim, lines])
                        if line == '\n':
                            break

    else:
        with open(data_file, 'r') as fobj:
            data_section = False
            iter_fobj = iter(fobj)
            
            for line in iter_fobj:
                line = line.strip()

                # skip anything until we reach the data section
                if 'Data:' not in line:
                    pass
                else:
                    data_section = True

                # start to fetch data
                if line.strip(',') in dimensions.keys() and data_section:
                    ax = line.strip(',')
                    line = next(iter_fobj)
                    line = line.strip()

                    data_name = line
                    if data_name not in data_names:
                        break
                    curr_data_dim = {'ax': [ax]}
                    data[data_name] = list()

                    line = next(iter_fobj)
                    lines = ''
                    while line != '\n':
                        lines += line
                        line = next(iter_fobj)
                    
                    data[data_name].append([curr_data_dim, lines])
                    # break

    return data