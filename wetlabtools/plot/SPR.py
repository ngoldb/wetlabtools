"""
Module to load, plot, and fit data from SPR experiments. 
"""

# data handling
import os
import numpy as np
import pandas as pd
import warnings

# plotting
import seaborn as sns
import matplotlib.pyplot as plt

# non-linear regression
import scipy

# wetlabtool utils
from wetlabtools import utils



# ============================
# Steady-state affinity data
# ============================

def load_affinity_data(file: str, omit_concentrations: list=[]):
    '''
    file: str, path to the file.txt with the affinity data
    omit_concentration: list, list of concentrations to omit

    This function loads the measured datapoints of the SPR experiment. It returns 
    a data frame with the x/y points with x being the concentration of ligand and
    y being the measured RU_max values for each ligand concentration.
    '''
    
    df = {'x':[],'y':[]}

    with open(file, 'r') as fobj:
        lines = fobj.readlines() 

    for line in lines:
        line = line.strip()
        if line.startswith('Run'):
            df['x'].append(line.split('\t')[-2].strip())
            df['y'].append(line.split('\t')[-1].strip())

    df = pd.DataFrame(df)
    df['x'] = pd.to_numeric(df['x'])
    df['x'] = df['x'] * 1_000_000
    df['y'] = pd.to_numeric(df['y'])
    df.sort_values('x', ascending=False, inplace=True)

    if omit_concentrations != []:
        df = df[~df['x'].isin(omit_concentrations)]
    
    return df


def load_affinity_fit(file: str):
    '''
    file: str, path to the file.txt with the affinity data
    
    This function loads the datapoints of the non-linear fit created in the Biacore
    Insight software. It returns a data frame with the x/y points to plot the fit.
    '''
    df_fit = {'x':[],'y':[]}

    with open(file, 'r') as fobj:
        lines = fobj.readlines()

    for line in lines[1:]:
        if line.strip() == '':
            break

        df_fit['x'].append(line.strip().split('\t')[0])
        df_fit['y'].append(line.strip().split('\t')[1])

    df_fit = pd.DataFrame(df_fit)
    df_fit['x'] = pd.to_numeric(df_fit['x'])
    df_fit['x'] = df_fit['x'] * 1_000_000
    df_fit['y'] = pd.to_numeric(df_fit['y'])
    
    return df_fit


def spr_affinity(measured: pd.DataFrame, fitted: pd.DataFrame=pd.DataFrame(), 
                 log: bool=True, save_fig: bool=False, height: int=4, width: int=6):
    '''
    measured: pd.DataFrame, data frame with the measured datapoints (expected columns x and y)
    fitted: pd.DataFrame, data frame with the data point of the calculated fit (expected columns x and y)
    log: bool, whether to plot on a logarithmic x-axis
    save_fig: bool, whether to save the plot to a file
    height: int, height of the plot
    width: int, width of the plot

    Function to plot affinity data from Biacore Insight software
    '''
    
    # initialize plot
    fig, ax = plt.subplots(figsize=(width, height))

    # plot the measured datapoints
    sns.scatterplot(data=measured,
                    x='x',
                    y='y',
                    ax=ax
                   )
    
    # plot the fit
    if not fitted.empty:
        sns.lineplot(data=fitted, x='x', y='y', ax=ax)
    
    # log x axis and labels
    sns.despine()
    if log == True:
        plt.xscale('log')
    plt.xlabel('Concentration (µM)')
    plt.ylabel('RU')

    if save_fig == True:
        plt.savefig("affinity.png", dpi=300)

    plt.show()

    return fig
    

def fit_sigmoid_function(data, mock_scale='log', method: str='dogbox', **curve_fit_kwargs):
    '''
    data: pd.DataFrame, data frame with the measured data points (expects columns x and y)
    mock_scale: str, either 'log' or 'lin' - the scale to generate mock data in
    method: str, optimization for scipy.optimize.curve_fit()
    curve_fit_kwargs: kwargs passed to scipy.optimize.curve_fit() e.g. maxfev=5000 to increase iterations
    
    This function will fit a sigmoid function to the data and return a data frame with
    x and y values for the fitted function and the ec50 of the fit.
    '''
    
    def sigmoid(x, slope, ec50, top, bottom):
        '''https://www.graphpad.com/guides/prism/latest/curve-fitting/reg_dr_stim_variable_2.htm'''
        return (x ** slope * (top - bottom)) / (x ** slope + ec50 ** slope) 

    # initial guess
    p0 = [1,
          np.mean(data['x']),
          max(data['y']),
          min(data['y'])
         ]

    popt, _ = scipy.optimize.curve_fit(sigmoid,
                                          data['x'],
                                          data['y'],
                                          p0,
                                          method=method,
                                          **curve_fit_kwargs
                                         )

    ec50 = popt[1]
    
    # generate mock data for plotting
    if mock_scale == 'log':
        start = min(data['x'])
        stop = max(data['x'])
        x = np.logspace(start=np.log10(start) - 0.5,
                        stop=np.log10(stop) + 0.5,
                        num=1000)
        
    if mock_scale == 'lin':
        x = np.linspace(min(data['x']), max(data['y']), 1000)
    
    y = sigmoid(x, *popt)
    df_fit = pd.DataFrame({'x':x, 'y':y})
    
    return df_fit, ec50


def multi_affinity(data_dir: str='', 
                   path_list: list=[],
                   normalize: bool=False, 
                   rel_scale: bool=False,
                   fit_sigmoid: bool=True,
                   report_kd: bool=True,
                   save_fig: bool=False, 
                   log: bool=True, 
                   height:int=4, 
                   width: int=6,
                   omit_concentrations: list=[],
                   omit_samples: list=[],
                   **curve_fit_kwargs):
    """
    data_dir: str, path to the directory containing txt files
    path_list: list, list of paths to txt files to plot (instead of data_dir)
    normalize: bool, whether to rescale all data from 0 to 100% of respective RU max
    rel_scale: bool, whether to scale all plots relative to the max RU (y axis will be 0 - 1)
    save_fig: bool, whether to save the figure
    fit_sigmoid: bool, whether to fit a sigmoid function
    report_kd: bool, whether to print Kd in the legend of the plot
    log: bool, whether or not to plot on logarithmic x-axis
    heigh: int, height of the plot
    wdith: int, width of the plot
    omit_concentrations: list, list of concentrations to omit from data
    omit_samples: list, list of samples to omit
    curve_fit_kwargs: kwargs passed to scipy.optimize.curve_fit() e.g. maxfev=5000 to increase iterations

    Function to plot affinity data from multiple experiments and overlay the curves in a single plot.
    It will parse the directory and collect all txt files containing "affinity" in the file name. There
    are two normalization options available: normalize and rel_scale. Normalize will rescale all plots
    to a y-axis of 0% to 100% of RUmax of the sample; rel_scale will scale all plots relative to the
    overall rel_max (on a scale from 0 to 1) - this is preferred if you like to show data together
    with their respective negative controls.
    """

    # sanity checks
    if normalize and rel_scale:
        warnings.warn('You are about to do something stupid: you try to use normalize and rel_scale simultaneously! Choose one of those options!')

    try:
        if data_dir == '' and path_list == []:
            raise IOError('I/O error: You have to provide input data - you can use either of the parameters data_dir or path_list')

        if data_dir != '' and path_list != []:
            raise IOError('You provided two input formats - use either data_dir or path_list')
    
    except IOError as e:
        print(e)
        return None
    
    # defining list of input files
    if path_list:
        files = path_list

    else:
        files = [os.path.join(data_dir, file) 
                for file in os.listdir(data_dir) 
                if file.endswith('.txt') and 'affinity' in file]

    # iterate over all files and load data
    data = {}

    for file in files:
        sample = os.path.basename(file).split('_')[0]
        data[sample] = load_affinity_data(file=file, omit_concentrations=omit_concentrations)

    # get global y max
    if rel_scale:
        rel_max = max([data[sample]['y'].max() for sample in data])

    # initialize the plot
    fig, ax = plt.subplots(figsize=(width, height))

    for sample in data:

        # skipping samples
        if sample in omit_samples:
            continue

        # fit sigmoid function if requested
        if fit_sigmoid:
            if log:
                scale = 'log'
            else:
                scale = 'lin'
            
            fitted, kd = fit_sigmoid_function(data=data[sample], mock_scale=scale, **curve_fit_kwargs)

            if normalize:
                y_max = fitted['y'].max()
                y_min = fitted['y'].min()
                fitted['y'] = fitted['y'].apply(lambda y: utils.normalize_percent_max(y, y_max, y_min))
            
            elif rel_scale:
                fitted['y'] = fitted['y'].apply(lambda y: utils.rel_scale(y, rel_max))

            # plot fitted function
            sns.lineplot(data=fitted, x='x', y='y')

        # normalize if requested
        if normalize:
            norm_data = data[sample].copy()
            y_max = data[sample]['y'].max()
            y_min = data[sample]['y'].min()
            norm_data['y'] = norm_data['y'].apply(lambda y: utils.normalize_percent_max(y, y_max, y_min))
            ylabel = 'Percent RUmax (%)'

        elif rel_scale:
            norm_data = data[sample].copy()
            norm_data['y'] = norm_data['y'].apply(lambda y: utils.rel_scale(y, rel_max))
            ylabel = 'relative RU'

        else:
            norm_data = data[sample]
            ylabel = 'response units (RU)'

        # plotting experimental data points
        if report_kd:
            label = f'{sample} (Kd = {round(kd*1_000, 2)} nM)'
        else:
            label = sample
        
        sns.scatterplot(data=norm_data,
                        x='x',
                        y='y',
                        ax=ax,
                        label=label
                        )

    # style plot
    sns.despine()
    if log == True:
        plt.xscale('log')

    plt.xlabel('Concentration (µM)')
    plt.ylabel(ylabel)

    if save_fig == True:
        plt.savefig("multi_affinity.png", dpi=300)

    plt.show()

    return fig


# =============
# Kinetics
# =============
def spr_kinetics(file: str, save_fig: bool=False, height: int=4, width: int=7, show_figure: bool=True):
    '''
    file: str, path to the .txt file containing the recorded sensorgram
    save_fig: bool, whether to save the plot to a file
    height: int, height of the plot
    width: int, width of the plot
    show_figure: bool, whether to show the figure
    
    This function will read the data from the provided data file and will plot
    the sensorgram of the respective sample.
    '''
    
    # initialize plot
    fig, ax = plt.subplots(figsize=(width, height))
    sns.set(palette='colorblind', style='ticks')
    
    # read data from file
    df = pd.read_csv(file, sep='\t', skipinitialspace=True)
    to_drop = [col for col in df.columns if 'Fitted' in col]
    df.drop(columns=to_drop, inplace=True)
    
    # drop fitted data if available

    columns = df.columns
    done = []

    mapping = {'X':'Y', 'Y':'X'}

    for column in columns:
        if column in done:
            pass
        else:
            try:
                cols = {'X':'','Y':''}

                sample = column.split(';')[4].strip()
                conc = column.split(';')[5].strip().strip('Conc').split('_')[0].strip()
                series = column.split(';')[-1].split('_')[-1].strip()

                cols[series] = column
                cols[mapping[series]] = column.replace(series, mapping[series])

                done.append(column)
                done.append(column.replace(series, mapping[series]))

                g = sns.lineplot(df, x=cols['X'], y=cols['Y'], label=conc, ax=ax)
            except: pass

    # adjusting plot
    plt.xlabel('time (s)')
    plt.ylabel('Response units')
    plt.title(sample)
    sns.despine()
    plt.legend(frameon=False)
    
    if save_fig:
        plt.savefig(f"{sample}_sensorgram.png", dpi=300)

    if show_figure:    
        plt.show()    
    
    plt.close('all')
        
    return fig



# =============
# Summary
# =============
def draw_sensorgram(file: str, ax, legend: bool) -> None:
    """
    file: path to the data file
    ax: matplotlib axes object
    legend: bool, whether to show legend

    Plot sensorgram to ax
    """
    sns.set(palette='colorblind', style='ticks')
    
    # read data from file
    df = pd.read_csv(file, sep='\t', skipinitialspace=True)
    to_drop = [col for col in df.columns if 'Fitted' in col]
    df.drop(columns=to_drop, inplace=True)
    
    if legend == True:
        legend = 'auto'

    columns = df.columns
    done = []

    mapping = {'X':'Y', 'Y':'X'}

    for column in columns:
        if column in done:
            pass
        else:
            try:
                cols = {'X':'','Y':''}

                conc = column.split(';')[5].strip().strip('Conc').split('_')[0].strip()
                series = column.split(';')[-1].split('_')[-1].strip()

                cols[series] = column
                cols[mapping[series]] = column.replace(series, mapping[series])

                done.append(column)
                done.append(column.replace(series, mapping[series]))

                g = sns.lineplot(df, x=cols['X'], y=cols['Y'], label=conc, ax=ax, legend=legend)
            except: pass

    ax.set_xlabel('time (s)', fontsize=18, fontname='Helvetica')
    ax.set_ylabel('RU', fontsize=18, fontname='Helvetica')

    return None



def draw_steady_state(exp_data: pd.DataFrame, fitted: pd.DataFrame, ax, log: bool=True) -> None:
    """
    exp_data: pandas DataFrame, experimental data
    fitted: pandas DataFrame, fit to experimental data
    ax: axes to plot on
    log: bool, whether to plot on log y axis

    Function to plot steady state affinity data
    """

    # plot the measured datapoints
    sns.scatterplot(data=exp_data,
                    x='x',
                    y='y',
                    ax=ax
                   )
    
    # plot the fit
    if not fitted.empty:
        sns.lineplot(data=fitted, x='x', y='y', ax=ax)
    
    # log x axis and labels
    sns.despine()
    if log == True:
        ax.set_xscale('log')
    
    ax.set_xlabel('Concentration (µM)', fontsize=18, fontname='Helvetica')
    ax.set_ylabel('RU', fontsize=18, fontname='Helvetica')

    return None



def spr_summary(data_dir: str, save_fig: bool=False, sensorgram_legend: bool=False, aff_scale: str='log'):
    """
    data_dir: str, path to the directory containing the data
    save_fig: bool, whether to save the figure or not
    sensorgram_legend: bool, whether to show legend in the sensorgram
    aff_scale: str, scale of the affinity plot x-axis (log or lin)

    Function to plot kinetics and affinity plots from SPR data collected on 
    Biacore 8. It will parse the data directory and plot kinetics and affinity
    for each sample next to each other. The files need to be named like this:
    sample-name_<affinity/kinetics>.txt
    """

    # collect all files
    files = [os.path.join(data_dir, file) 
             for file in os.listdir(data_dir) 
             if file.endswith('.txt')]
    
    # get (unique) sample names
    samples = set(os.path.basename(file).split('_')[0] for file in files)

    # create subplots: 1 row for each sample, 2 columns for imac and sec
    fig, ax = plt.subplots(len(samples), 2, figsize=(20,5*len(samples)))

    for i, sample in enumerate(samples):
        
        # setting axes
        if len(samples) == 1:
            ax_kinetics = ax[0]
            ax_affinity = ax[1]
            ax_kinetics.set_title('Sensorgram', fontsize=24, fontname='Helvetica')
            ax_affinity.set_title('Affinity', fontsize=24, fontname='Helvetica')
        
        else:
            ax_kinetics = ax[i, 0]
            ax_affinity = ax[i, 1]
            if i == 0:
                ax[i, 0].set_title('Sensorgram', fontsize=24, fontname='Helvetica')
                ax[i, 1].set_title('Affinity', fontsize=24, fontname='Helvetica')
            
        # find data files
        try:
            kinetics = [file for file in files if sample in file and 'kinetics'.casefold() in file.casefold()][0]
        except IndexError:
            print(f'did not find a IMAC chromatogram file for {sample}')
            kinetics = None
        
        try:
            affinity =  [file for file in files if sample in file and 'affinity'.casefold() in file.casefold()][0]
        except IndexError:
            print(f'did not find a SEC chromatogram file for {sample}')
            affinity = None

        # importing data and plotting
        if kinetics == None:
            # removing plot
            ax_kinetics.spines['top'].set_visible(False)
            ax_kinetics.spines['right'].set_visible(False)
            ax_kinetics.spines['bottom'].set_color('none')
            ax_kinetics.spines['left'].set_color('none')
            ax_kinetics.xaxis.set_ticks_position('none')
            ax_kinetics.yaxis.set_ticks_position('none')
            ax_kinetics.set_xticks([])
            ax_kinetics.set_yticks([])
            
        else:
            draw_sensorgram(file=kinetics, ax=ax_kinetics, legend=sensorgram_legend)
            
            # adding sample description
            ax_kinetics.text(0.02, 0.9, sample, 
                        transform=ax_kinetics.transAxes, 
                        fontsize=16, 
                        fontname='Helvetica')
        
        if affinity == None:
            ax_affinity.spines['top'].set_visible(False)
            ax_affinity.spines['right'].set_visible(False)
            ax_affinity.spines['bottom'].set_color('none')
            ax_affinity.spines['left'].set_color('none')
            ax_affinity.xaxis.set_ticks_position('none')
            ax_affinity.yaxis.set_ticks_position('none')
            ax_affinity.set_xticks([])
            ax_affinity.set_yticks([])
            
        else:
            affinity_data = load_affinity_data(file=affinity)
            fitted, kd = fit_sigmoid_function(affinity_data, mock_scale=aff_scale)
            
            draw_steady_state(exp_data=affinity_data, fitted=fitted, ax=ax_affinity)

            # adding sample description
            ax_affinity.text(0.02, 0.9, sample+f'\nKd = {round(kd, 3)} µM', 
                             transform=ax_affinity.transAxes, 
                             fontsize=16, 
                             fontname='Helvetica')
            
    if save_fig:
        pass
    
    plt.show()
    plt.close('all')

    return None