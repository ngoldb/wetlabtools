"""
Module to plot SEC-MALS data
"""

import os
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



def secmals(path:str, flow_rate:float, min_x:float=0, max_x:float=999, save_pdf:bool=False, save_png:bool=False):
    '''
    path: str, path to the directory with csv files
    flow_rate: float, flow rate in ml/min to convert min to ml (x-axis)
    min_x: float, minimum retention volume to plot
    max_x: float, maximum retention volume to plot
    save_png: bool, whether to save plots as pdf
    save_pdf: bool, whether to save plots as png
    
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