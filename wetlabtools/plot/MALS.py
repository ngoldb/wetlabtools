"""
Module to plot SEC-MALS data
"""

# general & data handling
import os
import pandas as pd

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# wetlabtool utils
from wetlabtools import utils



def secmals(path:str, convert_x: bool=False, flow_rate:float=0.5, min_x:float=0, max_x:float=999, MW_lim: set=(1e4, 1e6), 
            display_MW_mean: bool=True, save_svg:bool=False, save_png:bool=False):
    '''
    path: str, path to the directory with csv files
    convert_x: bool, whether to convert x axis from time (min) to volume (ml) - adjust flow_rate (default: 0.5 ml/min)
    flow_rate: float, flow rate in ml/min to convert time (min) to volume (ml)
    min_x: float, minimum retention volume to plot
    max_x: float, maximum retention volume to plot
    MW_lim: set, limits of the y axis for MW axis: [lower, upper]
    display_MW_mean: bool, calculate mean of MW and display on the plot
    save_png: bool, whether to save plots as pdf
    save_svg: bool, whether to save plots as svg
    
    Function to plot data from SEC-MALS. It will parse the directory for all csv files and plot UV signal and MW.
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

        # find column names
        uv_column = [(column, i) for i, column in enumerate(df.columns) if "UV" in column]
        mw_column = [(column, i) for i, column in enumerate(df.columns) if "Molar mass" in column]
        assert len(uv_column) == 1, f"UV column could not be identified. Columns: {df.columns.values}"
        assert len(mw_column) == 1, f"Molar mass column could not be identified. Columns: {df.columns.values}"

        uv_y_column = uv_column[0][0]
        uv_x_column = df.columns[uv_column[0][1]-1]
        mw_y_column = mw_column[0][0]
        mw_x_column = df.columns[mw_column[0][1]-1]

        # determine x axis unit
        if "mL" in uv_x_column:
            x_unit = "Volume [ml]"
        else:
            x_unit = "Time [min]"

        # convert x axis if requested
        if convert_x and x_unit == "Time [min]":
            df[uv_x_column] = df[uv_x_column] * flow_rate
            df[mw_x_column] = df[mw_x_column] * flow_rate
            x_unit = "Volume [ml]"
        
        fig,ax = plt.subplots()
        
        # Plotting UV
        ax.plot(df[uv_x_column],
                df [uv_y_column],
                color='#1f77b4',
                linewidth = 0.8)
        ax.set_ylabel(ylabel = "UV [Relative scale]",
                color='#1f77b4',
                fontsize=12)
        
        ax2=ax.twinx()
        
        # Plotting molecular weight
        ax2.scatter(df[mw_x_column],
                df [mw_y_column],
                color="black",
                s = 0.2)
        ax2.set_ylabel(ylabel = "MW [Da]",
                color="black",
                fontsize=12)
        
        # Make second axis log-scaled
        ax2.set_yscale("log")
        ax2.set_ylim (MW_lim[0], MW_lim[1])

        # x-axis
        ax.set_xlabel(x_unit, fontsize = 12)
        if min_x and max_x:
            ax.set_xlim(min_x, max_x)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_major_formatter('{x:.0f}')
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        x_lim = ax.get_xlim()

        # calculating mean MW for peaks
        # TODO: implement left / right annotaion
        if display_MW_mean:
            blocks = utils.find_consecutive_blocks(df[mw_x_column].dropna())    
            
            for low, high in blocks:

                # +1 here because upper limit is exclusive
                mean_MW = df.iloc[low : high + 1][mw_y_column].mean()
                
                x_coor = df.iloc[high][mw_x_column]

                # only add the text if in the plot limits. Otherwise it will create infinite big plots
                if MW_lim[0] < mean_MW < MW_lim[1] and x_lim[0] < x_coor < x_lim[1]:
                
                    ax2.text(s=f'{round(mean_MW / 1000, 2)} kDa',
                            x=round(x_coor, 2),
                            y=mean_MW
                            )
        
        plt.title(sample_name)

        if save_svg:
            plt.savefig(csv_path[:-4]+'.svg', format='svg')
            print(f'saving plot to {csv_path[:-4]}.svg')
        
        if save_png: 
            plt.savefig(csv_path[:-4]+'.png', dpi=300)
            print(f'saving plot to {csv_path[:-4]}.png')
        
        plt.show()
        plt.close('all')

    return None