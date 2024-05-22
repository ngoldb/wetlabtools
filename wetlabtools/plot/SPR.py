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
                 log: bool=True, save_fig: bool=False, height: int=4, width: int=6, ylim: list=[],
                 **save_kwargs):
    '''
    measured: pd.DataFrame, data frame with the measured datapoints (expected columns x and y)
    fitted: pd.DataFrame, data frame with the data point of the calculated fit (expected columns x and y)
    log: bool, whether to plot on a logarithmic x-axis
    save_fig: bool, whether to save the plot to a file
    height: int, height of the plot
    width: int, width of the plot
    ylim: list, limits of y axis (automatic if empty)
    **save_kwargs: keyword arguments passed to plt.savefig()

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

    if ylim:
        plt.ylim(ylim)
    
    # log x axis and labels
    sns.despine()
    if log == True:
        plt.xscale('log')
    plt.xlabel('Concentration (µM)')
    plt.ylabel('RU')

    if save_fig == True:
        # there must be a nore elegant way
        try:
            fname = f'affinity.{save_kwargs["format"]}'
        except KeyError:
            fname = 'affinity.png'

        plt.savefig(fname, **save_kwargs)
        print(f'saved as {fname}')

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
                   marker: str='o',
                   normalize: bool=False, 
                   rel_scale: bool=False,
                   fit_sigmoid: bool=True,
                   report_kd: bool=True,
                   save_fig: bool=False, 
                   log: bool=True, 
                   height: int=4, 
                   width: int=6,
                   omit_concentrations: list=[],
                   omit_samples: list=[],
                   svg: bool=False,
                   **curve_fit_kwargs
                   ):
    """
    data_dir: str, path to the directory containing txt files
    path_list: list, list of paths to txt files to plot (instead of data_dir)
    marker: str, marker to use (seaborn, e.g. 'o' or '^')
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
    svg: bool, whether to save svg file or png (default)
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
        
        if report_kd and not fit_sigmoid:
            raise IOError('You can not report a Kd if you did not fit a sigmoid curve to the data! The Kd is derived from the sigmoid curve fit.')
    
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
        if report_kd and fit_sigmoid:
            label = f'{sample} (Kd = {round(kd*1_000, 2)} nM)'
        else:
            label = sample
        
        sns.scatterplot(data=norm_data,
                        x='x',
                        y='y',
                        ax=ax,
                        label=label,
                        marker=marker
                        )

    # style plot
    sns.despine()
    if log == True:
        plt.xscale('log')

    plt.xlabel('Concentration (µM)')
    plt.ylabel(ylabel)

    if save_fig == True:
        out_file = os.path.join(data_dir, 'multi-affinity')

        # there must be a nore elegant way
        if svg:
            fname = f'{out_file}.svg'
            plt.savefig(fname, format='svg')

        else:
            # png is default if no format specified
            fname = f'{out_file}.png'
            plt.savefig(fname, dpi=300)

        print(f'saved as {fname}')

    plt.show()

    return fig


# =============
# Kinetics
# =============
def load_sensorgram(file: str) -> pd.DataFrame:
    '''
    file: str, path to the .txt file containing the recorded sensorgram

    This function will read the data from the provided data file and
    return a data frame in long format.
    '''
    
    # read data from file
    df = pd.read_csv(file, sep='\t', skipinitialspace=True)

    columns = df.columns
    done = []

    mapping = {'X':'Y', 'Y':'X'}
    df_long = pd.DataFrame(columns=['concentration','x','y','type'])

    for column in columns:
        if column in done:
            pass
        else:
            try:
                # find x and y value pairs from cryptic column names
                cols = {'X':'','Y':''}

                sample = column.split(';')[4].strip()
                conc = column.split(';')[5].strip().strip('Conc').split('_')[0].strip()
                series = column.split(';')[-1].split('_')[-1].strip()

                cols[series] = column
                cols[mapping[series]] = column.replace(series, mapping[series])

                # convert into long format
                df_temp = pd.DataFrame()
                df_temp['x'] = df[cols['X']]
                df_temp['y'] = df[cols['Y']]
                df_temp['concentration'] = conc
                
                if 'Fitted' in column:
                    df_temp['type'] = 'fit'
                else:
                    df_temp['type'] = 'data'
                    
                # add to df_long
                df_long = pd.concat([df_long, df_temp.dropna()], ignore_index=True, sort=False)
                    
                # record that we are done with this pair
                done.append(column)
                done.append(column.replace(series, mapping[series]))

            except: pass
        
    return df_long, sample


def kinetics(file: str, save_fig: bool=False, height: int=4, width: int=7, 
             show_figure: bool=True, ylim: list=[], plot_fit: bool=False, 
             single_cycle: bool=False, annotate_cycles: bool=False, 
             **save_kwargs):
    '''
    file: str, path to the .txt file containing the recorded sensorgram
    save_fig: bool, whether to save the plot to a file
    height: int, height of the plot
    width: int, width of the plot
    show_figure: bool, whether to show the figure
    ylim: list, y-axis limits (automatic if empty)
    plot_fit: bool, whether to plot the fitted function
    single_cycle: bool, whether data is single cycles kinetics measurement
    annotate_cycles: bool, print concentration for single cycle kinetics
    save_kwargs: kwargs passed to plt.savefig
    
    This function will read the data from the provided data file and will plot
    the sensorgram of the respective sample.
    '''
    
    # sanity check
    if annotate_cycles and not single_cycle:
        warnings.warn('annotate_cycles is an option specific to single cylce kinetics. You did not set single_cycle=True')

    # initialize plot
    fig, ax = plt.subplots(figsize=(width, height))
    sns.set_theme(palette='colorblind', style='ticks')
    
    # read data from file
    df, sample = load_sensorgram(file)

    # plotting
    if plot_fit:
        g = sns.lineplot(data=df, x='x', y='y', hue='concentration', style='type')
        handles, labels = ax.get_legend_handles_labels()
    else:
        g = sns.lineplot(data=df[df.type=='data'], x='x', y='y', hue='concentration')
        handles, labels = ax.get_legend_handles_labels()
        
    # Specific adjustments 
    if plot_fit:
        # clean up legend entries
        to_remove = [labels.index(x) for x in ['concentration', 'type', 'data']]

        for x in reversed(to_remove):
            handles.pop(x)
            labels.pop(x)

        if single_cycle:
            handles.pop(0)
            handles[0].set_c('b')
            sc_conc = labels.pop(0)
            
        plt.legend(handles, labels, frameon=False)

    else:
        if single_cycle:
            sc_conc = labels.pop(0)
            plt.legend('', frameon=False)
        else:
            plt.legend(frameon=False)

    # adding description of cycles in single cycle kinetics
    if annotate_cycles and single_cycle:
        # create a list of concentrations
        sc_conc = sc_conc.split()
        sc_conc = [' '.join([x, sc_conc[-1]]) for x in sc_conc[0:-1]]
        
        # estimate time of one cycle
        cycle_time = df[df.y == df.y.max()].x.values[0] / len(sc_conc)
        
        # add text to plot
        for i, conc in enumerate(sc_conc):
            x_pos = cycle_time*(i+1)-cycle_time/2
            y_max = g.get_ylim()[1]
            plt.text(x=x_pos,
                        y=0.98*y_max,
                        s=conc,
                        horizontalalignment='center',
                        verticalalignment='top',
                        fontsize=6)

    # adjusting plot
    plt.xlabel('time (s)')
    plt.ylabel('Response units')
    plt.title(sample)
    sns.despine()
    
    if ylim:
        plt.ylim(ylim)

    if save_fig == True:
        # there must be a nore elegant way
        try:
            fname = f'kinetics.{save_kwargs["format"]}'
        except KeyError:
            fname = 'kinetics.png'

        plt.savefig(fname, **save_kwargs)
        print(f'saved as {fname}')

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
    sns.set(palette='colorblind', style='ticks', rc={'axes.linewidth': 2})

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
    sns.set(palette='colorblind', style='ticks', rc={'axes.linewidth': 2})

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
    
    ax.set_xlabel('Concentration (µM)', fontsize=16, fontname='Helvetica')
    ax.set_ylabel('RU', fontsize=16, fontname='Helvetica')

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

    # TODO: fix styling of the plots (x- and y-axis linewidth, fontsize of the xtick labels)
    #       currently these settings only affect the very last plot of the figure (bottom left)

    # collect all files
    files = [os.path.join(data_dir, file) 
             for file in os.listdir(data_dir) 
             if file.endswith('.txt')]
    
    # get (unique) sample names
    samples = set(os.path.basename(file).split('_')[0] for file in files)

    # create subplots: 1 row for each sample, 2 columns for imac and sec
    fig, ax = plt.subplots(len(samples), 2, figsize=(20,5*len(samples)))
    
    # adjusting padding
    plt.subplots_adjust(left=0.05,
                        bottom=0.05, 
                        right=1, 
                        top=0.9, 
                        wspace=0.2, 
                        hspace=0.2)
    
    # fontsize
    """
    for axs in ax:
        axs.tick_params(axis='x', labelsize=12)
        axs.tick_params(axis='y', labelsize=12)
    """

    for i, sample in enumerate(samples):

        # setting axes
        if len(samples) == 1:
            ax_kinetics = ax[0]
            ax_affinity = ax[1]
            ax_kinetics.set_title('Sensorgram', fontsize=20, fontname='Helvetica')
            ax_affinity.set_title('Affinity', fontsize=20, fontname='Helvetica')
        
        else:
            ax_kinetics = ax[i, 0]
            ax_affinity = ax[i, 1]
            if i == 0:
                ax[i, 0].set_title('Sensorgram', fontsize=20, fontname='Helvetica')
                ax[i, 1].set_title('Affinity', fontsize=20, fontname='Helvetica')
            
        # find data files
        try:
            kinetics = [file for file in files if sample in file and 'kinetics'.casefold() in file.casefold()][0]
        except IndexError:
            print(f'did not find kinetics file for {sample}')
            kinetics = None
        
        try:
            affinity =  [file for file in files if sample in file and 'affinity'.casefold() in file.casefold()][0]
        except IndexError:
            print(f'did not find affinity file for {sample}')
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
                        fontsize=12, 
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

            try:
                fitted, kd = fit_sigmoid_function(affinity_data, mock_scale=aff_scale)
                kd = f'{round(kd, 3)} µM'
            except RuntimeError:
                # handling in case fitting fails (e.g. for negative controls)
                fitted = pd.DataFrame
                kd = 'NA'

            draw_steady_state(exp_data=affinity_data, fitted=fitted, ax=ax_affinity)

            # adding sample description
            ax_affinity.text(0.02, 0.9, sample+f'\nKd = {kd}', 
                             transform=ax_affinity.transAxes, 
                             fontsize=12, 
                             fontname='Helvetica')
            
    if save_fig:
        plt.savefig(os.path.join(data_dir, 'spr_summary.png'), dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close('all')

    return None