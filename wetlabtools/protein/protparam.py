"""Module to calculate biochemical numbers from amino acid sequence"""

# general imports
import numpy as np
import pandas as pd

# BioPython
import Bio.SeqIO
import Bio.SeqUtils.ProtParam

# wetlabtools import
from wetlabtools import utils

# TODO  - cloning module: adapter sequences for cloning


class Protein(object):
    """Protein object"""

    def __init__(self, seq: str, letter_code: int=1):
        """
        Instantiate Protein from amino acid sequence. Sequence must contain amino acids only.
        letter_code can be 1 or 3 depending on the sequence being in 1 or 3 letter code.
        """

        # sanity check
        if letter_code not in [1, 3]:
            raise ValueError(f'letter code must be either 1 or 3. You provided: {letter_code}')
        
        if letter_code == 3:
            if len(seq) % 3 != 0:
                raise ValueError(f'Your amino acid sequence is corrupt. Please check that the entire sequence is in 3 letter code!')
            
            else:
                # convert from 3 letter code to 1 letter code
                try:
                    seq = [utils.aa_321[seq[i:i+3].capitalize()] for i in range(0, len(seq), 3)]

                except KeyError as error:
                    print('It looks like there is an unknown amino acid in your sequence:', error)
                    return None

                seq = ''.join(seq)
                seq = seq.upper()

        if letter_code == 1:
            seq = seq.upper()

            for aa in seq:
                if aa.upper() not in utils.aa_1_letter:
                    raise ValueError(f'There is an unknown amino acid in your sequence!')
 
        # instantiate protein object
        self.seq = seq
        self.len = len(self.seq)
        self.prot_param = Bio.SeqUtils.ProtParam.ProteinAnalysis(self.seq)
        self.aa_frequencies = self.prot_param.get_amino_acids_percent()
        self.isoelectric_point = self.prot_param.isoelectric_point()
        self.molecular_weight = self.prot_param.molecular_weight()
        self.ext_coef_red = self.prot_param.molar_extinction_coefficient()[0]
        self.ext_coef_ox = self.prot_param.molar_extinction_coefficient()[1]
        self.e1_percent_red = round(self.ext_coef_red / self.molecular_weight * 10, 2)
        self.e1_percent_ox = round(self.ext_coef_ox / self.molecular_weight * 10, 2)

    def charge_at_pH(self, pH):
        """calculates charge of the protein at a given pH"""
        return self.prot_param.charge_at_pH(pH)
    
    def aa_frequency(self, aa):
        """calculate frequency of given amino acid in protein"""
        return self.aa_frequencies[aa]
    
    def charge_table(self, low_pH: float=3.5, high_pH: int=9.5, step: float=0.5) -> dict:
        """
        low_pH: float, pH to start table from (inclusive)
        high_pH: float, pH to stop table at (inclusive)
        step: float, pH step

        prints and returns a table with charges at pH in the specified pH range
        """

        # init variables
        pH_range = np.arange(low_pH, high_pH, step)
        pH_table = {}

        print('pH\tcharge\n')

        # calculate charge at pH
        for pH in pH_range:
            pH_table[pH] = round(self.charge_at_pH(pH), 2)
            print(f'{pH}:\t{pH_table[pH]}')

        return pH_table
    
    def print_protparam(self, quiet: bool=False) -> dict:
        """Creates an output similar to the ExPasys ProtParam tool"""
        
        param_dict = {}

        # collecting all the prot param scores
        param_dict['Number of amino acids'] = self.len
        param_dict['Molecular Weight (Da)'] = round(self.molecular_weight, 2)
        param_dict['Theoretical pI'] = round(self.isoelectric_point, 2)
        param_dict['Charge at pH 7.5'] = round(self.charge_at_pH(7.5), 2)
        param_dict['Molar extinction coefficient (ox) [1/M*cm @ 280nm]'] = self.ext_coef_ox
        param_dict['Molar extinction coefficient (red) [1/M*cm @ 280nm]'] = self.ext_coef_red
        param_dict['Absorbance 1% (ox) [A280 @ 10g/l]'] = round(self.e1_percent_ox, 2)
        param_dict['Absorbance 1% (red) [A280 @ 10g/l]'] = round(self.e1_percent_red, 2)
        
        if not quiet:
            for property in param_dict:
                print(property + '\t' + str(param_dict[property]))

            print('\namino acid frequencies:')
            for aa in self.aa_frequencies:
                print(aa + '\t' + str(round(self.aa_frequencies[aa] * 100, 1)) + '%')
                param_dict[aa + ' [%]'] = round(self.aa_frequencies[aa] * 100, 1)

            print('\nCharge Table:')
            self.charge_table()

        return param_dict



def batch_prot_param(fasta_file: str, csv: str='') -> pd.DataFrame:
    """
    fasta_file: str, path to the fasta file with the protein sequences (1-letter code)
    csv: str, will write a csv file to this file if provided

    Function to batch-process a list of sequences in a fasta file and return a 
    dataframe with ProtParam output. It will save a csv file if requested.
    """

    # need to change the labels to shorter for csv file
    LABEL_CONVERSION = {'Number of amino acids': 'aa count',
                        'Molecular Weight (Da)': 'MW [Da]',
                        'Theoretical pI': 'pI',
                        'Charge at pH 7.5': 'Charge (pH 7.5)',
                        'Molar extinction coefficient (ox) [1/M*cm @ 280nm]': 'E280_ox [1/M*cm]',
                        'Molar extinction coefficient (red) [1/M*cm @ 280nm]': 'E280_red [1/M*cm]',
                        'Absorbance 1% (ox) [A280 @ 10g/l]': 'A280_ox @ 10g/l',
                        'Absorbance 1% (red) [A280 @ 10g/l]': 'A280_red @ 10g/l'
                        }
    
    data = pd.DataFrame()

    for record in Bio.SeqIO.parse(fasta_file, 'fasta'):
        name = record.id

        # create protein from sequence
        protein = Protein(record.seq)
        prot_param = protein.print_protparam(quiet=True)
        tmp_df = pd.DataFrame(prot_param, index=[name])

        data = pd.concat([data, tmp_df])

    data.rename(columns=LABEL_CONVERSION, inplace=True)
    
    if csv != '':
        data.to_csv(csv, index=True)
        print(f'wrote data to {csv}')

    return data