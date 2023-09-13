"""
Tools to easily plot experimental data:
- CD spectra
"""

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

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
                    columns = [int(name) for name in columns]
                    
                    # iterate over the available wave lengths
                    for _, _ in enumerate(wv_len[::-1]):
                        line = next(fobj)
                        m_wv = int(line.strip().split(',')[0])
                        dataline = line.strip().split(',')[1:]
                        tmp_data[m_wv] = dataline
                    
                    # now construct Data Frame and convert to numeric
                    df_temp = pd.DataFrame.from_dict(tmp_data, 
                                orient='index',
                                columns=columns
                               ).apply(pd.to_numeric)

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
                    columns = [int(name) for name in columns]

                    # iterate over the available wave lengths
                    for _, _ in enumerate(wv_len[::-1]):
                        line = next(fobj)
                        m_wv = int(line.strip().split(',')[0])
                        dataline = line.strip().split(',')[1:]
                        tmp_data[m_wv] = dataline

                    # now construct Data Frame and convert to numeric
                    df_temp = pd.DataFrame.from_dict(tmp_data, 
                                orient='index',
                                columns=columns
                               ).apply(pd.to_numeric)

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

                # now construct Data Frame and convert to numeric
                df_temp = pd.DataFrame.from_dict(tmp_data, 
                            orient='index'
                           ).apply(pd.to_numeric)
    return df_temp



def load_CD_data(data_csv: str, buffer_csv: str=None):
    """
    :param: data_csv: str, path to the CD data csv file
    :param: buffer_csv: str, path to the buffer csv file
    
    Function to load and process CD data. The function will load CD, HV, and Absorbance data from the
    data csv file and return a dictonary with all data for all samples. Additionally, it will subtract
    the buffer CD signal in case a path to the buffer csv file is provided.
    """
    
    data = load_CD_melt(data_file=data_csv)
    
    if buffer_csv != None:
        buffer_CD = get_buffer_CD(buffer_csv)
        
        # correct for buffer baseline
        samples = [sample for sample in data['CircularDichroism']]
        
        for sample in samples:
            column_names = data['CircularDichroism'][sample].columns.astype(int)
            index_names = data['CircularDichroism'][sample].index.astype(int)
            data['CircularDichroism'][sample] = pd.DataFrame(data['CircularDichroism'][sample].to_numpy() - buffer_CD.to_numpy(),
                                                             columns=column_names,
                                                             index=index_names)
        
    return data



def plot_CD(data:dict, zooms:list, out_path:str='.', cutoff:float=2.0, mode:str='fade', min_x:float=195, max_x:float=260, save_pdf:bool=False, save_png:bool=False):
    '''
    :param: data: dict, data dictonary containing CD, HV, and Absorbance data
    :param: zooms: list, list of zoomed regions, False for no zoom, len of list must match number of samples
    :param: out_path: str, path to the output directory
    :param: cutoff: float, cutoff for absorbance
    :param: mode: str, cutoff mode for CD plot ['None','fade','cut']
    :param: min_x: float, min wavelength to plot
    :param: max_x: float, max wavelength to plot
    :param: save_pdf: bool, whether to save plot as pdf
    :param: save_png: bool, whether to save plot as png
    Function to plot data from CD melting ramps. Requires processing of the data in advance. Will plot CD spectrum, HV, and Absorbance
    '''
    i = 0
    
    sample_names = data['CircularDichroism'].keys()
    
    for sample_name in sample_names:
        zoom = zooms[i]
        mode = mode.upper()
        out_file = os.path.join(out_path, sample_name)

        # CD signal
        df = data['CircularDichroism'][sample_name]
        df = df.dropna(axis=1, how="all")
        df = df[::-1]
        
        df_a = data['Absorbance'][sample_name]
        df_a = df_a.dropna(axis=1, how="all")
        df_a = df_a[::-1]
        
        # plot
        fig, ((ax1,ax3), (ax4, ax5)) = plt.subplots(2,2, figsize=(15, 10)) # ax2 is zoom if used
        
        if mode == 'NONE':
            sns.lineplot(data=df, legend="full", ax=ax1, dashes=False, palette="ch:s=-.2,r=.6").set(title=sample_name)
        else:
            dfn = df.mask(df_a >= cutoff)
            sns.lineplot(data=dfn, legend="full", ax=ax1, dashes=False, palette="ch:s=-.2,r=.6").set(title=sample_name)
            
            if mode == 'FADE':
                dff = df.mask(df_a <= cutoff+cutoff*0.1)
                sns.lineplot(data=dff, legend=False, ax=ax1, dashes=False, palette="ch:s=-.2,r=.6", alpha=0.2)
                
        sns.move_legend(ax1, "upper left", bbox_to_anchor=(1,1), prop={'size': 8}, ncol=2, title='Temperature [°C]', frameon=False)
        
        ax1.set(ylabel = "CD [mdeg]", xlabel = "Wavelength [nm]")
        ax1.set_xlim([min_x,max_x])
        selected_rows = df.loc[min_x+5:max_x-5]
        
        y_min, y_max = selected_rows.min().min()-10, selected_rows.max().max()
        ax1.set_ylim(y_min, y_max)
        
        # plot zoom in plot
        if zoom:
            ax2 = plt.axes([0.2857, 0.7, .2, .2])
            sns.lineplot(data=df, ax=ax2, dashes=False, legend=False, palette="ch:s=-.2,r=.6")
            ax2.set_box_aspect(0.7)
            ax2.set_title('Zoom')
            ax2.set_xlim([205,230])
            ax2.set_ylim(zoom)
            ax2.set(xlabel = None)
            ax2.set(xticklabels=[])
            ax2.set(ylabel = None)
            ax2.set(yticklabels = [])
            
        # plot melting correlation at 220 nm
        df2 = df.copy()
        row_220 = df.loc[220].transpose()
        plot = sns.scatterplot(data = row_220, ax=ax3, legend = False)
        plot.set(title=sample_name)
        sns.lineplot(data=row_220, ax=ax3)
        plot.set(ylabel = "CD [mdeg] @220 nm", xlabel = "Temperature [°C]")
        plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
        y2_min, y2_max = row_220.min()-10, row_220.max()+10
        ax3.set_ylim(y2_min, y2_max)
        
        plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
        
        # Absorbance data
        sns.lineplot( data=df_a, legend='full', ax=ax4, dashes=False, palette="ch:s=-.2,r=.6").set(title=sample_name )
        sns.move_legend( ax4, "upper left", bbox_to_anchor=(1,1), prop={'size': 8}, ncol=2, title='Temperature [°C]', frameon=False )
        ax4.set( ylabel = "Absorbance [a.u.]", xlabel = "Wavelength [nm]" )
        ax4.set_xlim( [min_x,max_x] )
        ax4.axhline( y=cutoff, color='red' )
        
        # HV data
        df = data['HV'][sample_name]
        df = df.dropna( axis=1, how="all" )
        df = df[::-1]
        sns.lineplot( data=df, legend=False, ax=ax5, dashes=False, palette="ch:s=-.2,r=.6" ).set(title=sample_name)
        ax5.set( ylabel = "HV [V]", xlabel = "Wavelength [nm]" )
        ax5.set_xlim( [min_x,max_x] )
        # ax5.axhline( y=800, color='red' )
        
        if save_pdf: 
            plt.savefig(out_file+'.pdf')
            print(f'saving plot to {out_file}.pdf')
        if save_png: 
            plt.savefig(out_file+'.png', dpi=300)
            print(f'saving plot to {out_file}.png')
        
        plt.show()
        plt.close()
        i = i + 1
    
    return None



# =======================================
# SEC-MALS
# =======================================
def plot_MALS(path:str, flow_rate:float, min_x:float=0, max_x:float=999, save_pdf:bool=False, save_png:bool=False):
    '''
    :param: path: str, path to the directory with csv files
    :param: flow_rate: float, flow rate in ml/min to convert min to ml (x-axis)
    :param: min_x: float, minimum retention volume to plot
    :param: max_x: float, maximum retention volume to plot
    :param: save_png: bool, whether to save plots as pdf
    :param: save_pdf: bool, whether to save plots as png
    
    Function to plot data from SEC-MALS. It will parse the directory for all csv files and plot them as SEC-MALS data.
    '''

    # collect all csv files in path
    paths = []
    for file in os.listdir(path):
        if file.endswith('.csv'):
            paths.append(os.path.join(path,file))
    
    # plot data for all csv files
    for csv_path in paths:
        sample_name = os.path.basename(csv_path).split('_')[-1].split('.')[0]

        df = pd.read_csv(csv_path)
        df.rename(columns = {df.columns[7]:'MW [Da]', df.columns[5]: "UV [Relative scale]"}, inplace = True)
        
        fig,ax = plt.subplots()
        
        # Set y-axis variable for left y-axis
        ax.plot(df["time (min).1"] * flow_rate,
                df ["UV [Relative scale]"],
                color='#1f77b4',
                linewidth = 0.8)
        ax.set_ylabel(ylabel = "UV [Relative scale]",
                color='#1f77b4',
                fontsize=12)
        
        ax2=ax.twinx()
        
        # Set x-axis variable for right y-axis
        ax2.scatter(df["time (min).3"]*0.5,
                df ["MW [Da]"],
                color="black",
                s = 0.2)
        ax2.set_ylabel(ylabel = "MW [Da]",
                color="black",
                fontsize=12)
        
        # Make second axis log-scaled
        ax2.set_yscale("log")
        ax2.set_ylim (1e4, 1e6)
        
        # x-axis
        ax.set_xlabel("Volume [ml]", fontsize = 12)
        if min_x and max_x:
            ax.set_xlim(min_x, max_x)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_major_formatter('{x:.0f}')
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        
        plt.title(sample_name)

        if save_pdf:
            plt.savefig(csv_path[:-4]+'.pdf')
            print(f'saving plot to {csv_path[:-4]}.pdf')
        
        if save_png: 
            plt.savefig(csv_path[:-4]+'.png', dpi=300)
            print(f'saving plot to {csv_path[:-4]}.png')
        
        plt.show()
        plt.close('all')

    return None