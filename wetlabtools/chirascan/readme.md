# Chirascan CD data analysis
This module can import and process data from CD experiments conducted on chirascan instruments. The data must have been converted to csv. The module contains a central class cd_experiment, which should be able to process (almost) any data format recorded on chirascan instruments. Currently it does not support thermal melts with final measurements. It supports full spectrum thermal melts, single wavelength thermal melts. Thermal ramps can also contain recordings of the return ramp.

## Example usage
You can import the data manually using the central cd_experiment class and process / plot the data on your own. The data will be formatted into long format allowing for easy plotting.
```
cd = wetlabtools.chirascan.cd_experiment(result_file=cd_data)
cd.subtract_blank(blank_data)
data = cd.get_data()
```

When you provide the sample information, the script can convert the units from mdeg (measured by the instrument) into MRE. The protein concentration must be provided in µM and the pathlenght must be provided in mm:
```python
import wetlabtools

sample_dict = {
    'Cell1': {
        'sample_id': 'sample_1',
        'conc': 10,                     # µM
        'n_pep': 116,
        'pathlength': 1                 # mm
    },
    'Cell2': {
        'sample_id': 'sample_2',
        'conc': 10,
        'n_pep': 116,
        'pathlength': 1
    },
    'Cell3': {
        'sample_id': 'sample_3',
        'conc': 10,
        'n_pep': 113,
        'pathlength': 1
    },
    'Cell4': {
        'sample_id': 'sample_4',
        'conc': 10,
        'n_pep': 116,
        'pathlength': 1
    },
    'Cell5': {
        'sample_id': 'sample_5',
        'conc': 10,
        'n_pep': 113,
        'pathlength': 1
    },
    'Cell6': {
        'sample_id': 'sample_6',
        'conc': 10,
        'n_pep': 113,
        'pathlength': 1
    }
}

cd = wetlabtools.chirascan.cd_experiment(result_file=cd_data, sample_data=sample_dict)
cd.subtract_blank(blank_data)
cd.convert(unit='mre')
data = cd.get_data()
```
Note that user provided sample data will overwrite the data read from the data file!

## Layouts
Layouts take specific data inputs and generate figures from these. 
### Single wavelength melt with pre- and post-melt full spectra
This layout plots the full spectra before and after a melt on the left and plots the melting curve on the right. It will generate one figure for each cell and convert the cd signal to MRE.
```python
premelt_file = '/path/to/pre_melt_spectra.csv'
postmelt_file = '/path/to/post_melt_spectra.csv'
melt_file = '/path/to/single_wavelength_melt.csv'
blank_file = '/path/to/buffer.csv'

wetlabtools.chirascan.layouts.single_wvl_melt(premelt_file=premelt_file, 
                                              postmelt_file=postmelt_file, 
                                              melt_file=melt_file, 
                                              blank_file=blank_file, 
                                              sample_data=sample_dict, 
                                              wvl_lim=(200,260), 
                                              tmp_lim=(20,90), 
                                              save_png=True, 
                                              save_svg=True)
```