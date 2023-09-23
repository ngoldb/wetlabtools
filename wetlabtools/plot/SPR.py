"""
Module to load, plot, and fit data from SPR experiments. 
"""

# data handling
import numpy as np
import pandas as pd

# plotting
import seaborn as sns
import matplotlib.pyplot as plt

# non-linear regression
import scipy



# ============================
# Steady-state affinity data
# ============================
def load_affinity_data(file: str):
    '''
    file: str, path to the file.txt with the affinity data
    
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


def spr_affinity(measured: pd.DataFrame, fitted: pd.DataFrame, 
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
                    y='y'
                   )
    
    # plot the fit
    sns.lineplot(data=fitted, x='x', y='y')
    
    # log x axis and labels
    sns.despine()
    if log == True:
        plt.xscale('log')
    plt.xlabel('Concentration (µM)')
    plt.ylabel('RU')

    if save_fig == True:
        plt.savefig(f"{sample}_affinity.png", dpi=300)

    plt.show()
    

def fit_sigmoid_function(data, mock_scale='log'):
    '''
    data: pd.DataFrame, data frame with the measured data points (expects columns x and y)
    mock_scale: str, either 'log' or 'lin' - the scale to generate mock data in
    
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
                                          method='dogbox'
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



# =============
# Kinetics
# =============
def spr_kinetics(file: str, save_fig: bool=False, height: int=4, width: int=7):
    '''
    file: str, path to the .txt file containing the recorded sensorgram
    save_fig: bool, whether to save the plot to a file
    height: int, height of the plot
    width: int, width of the plot
    
    This function will read the data from the provided data file and will plot
    the sensorgram of the respective sample.
    '''
    
    # initialize plot
    fig, ax = plt.subplots(figsize=(width, height))
    
    # read data from file
    df = pd.read_csv(file, sep='\t', skipinitialspace=True)
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

                g = sns.lineplot(df, x=cols['X'], y=cols['Y'], label=conc)
            except: pass

    # adjusting plot
    sns.set(palette='colorblind', style='ticks')
    
    plt.xlabel('time (s)')
    plt.ylabel('Response units')
    plt.title(sample)
    sns.despine()
    plt.legend(frameon=False)
    
    if save_fig:
        plt.savefig(f"{sample}_sensorgram.png", dpi=300)
        
    plt.show()
    
    plt.close('all')
    
    return None