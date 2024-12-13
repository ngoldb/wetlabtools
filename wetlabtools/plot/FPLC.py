"""
Module to plot any kind of chromatogram from FPLC (IMAC, SEC, ...)
"""

# general
import os
import math
import warnings

# Data handling and plotting
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Bokeh
import bokeh.io
import bokeh.layouts
import bokeh.models
import bokeh.plotting

notebook_url = 'localhost:8888'
bokeh.io.output_notebook()



def import_fplc(fplc_data: str) -> pd.DataFrame:
    """
    fplc_data: str, path to the csv file with FPLC data
    
    Function to load FPLC data and fix the column headings. Will return a Data Frame.
    """
    
    # open csv file and fix header formatting
    # fixed csv is written to temporary file
    tmp_file = 'wetlabtools_tmp.csv'

    with open(fplc_data, 'r', encoding='utf16') as read_file, open(tmp_file, 'w') as write_file:
        for line in read_file:
            
            if 'UV' in line:
                columns = []
                for entry in line.strip().split('\t'):  
                    if entry == '' or entry == '\n':
                        pass
                    else:
                        columns.append(entry)
                        columns.append(entry)
                
                write_file.writelines('\t'.join(columns) + '\n')
                
            else:
                write_file.writelines(line)
                
    # load data into data frame
    data = pd.read_csv(tmp_file,
                    sep='\t',
                    encoding='utf8',
                    skiprows=1,
                    header=[0,1]
                    )

    # remove temporary file
    os.remove(tmp_file)

    return data



def interactive_fplc(csv_path: str, height:int=600, width:int=1_000) -> None:
    """
    csv_path: str, path to the csv file containing FPLC data
    height: int, height of the plot
    width: int, width of the plot
    
    Function to make an interactive plot of FPLC data
    """

    # import csv
    data = import_fplc(csv_path)

    # split and source data
    uv_data = data['UV']
    cond_data = data['Cond']
    cB_data = data['Conc B']
    fractions = data['Fraction']
    frac_data = fractions.dropna()

    uv_source = bokeh.models.ColumnDataSource(uv_data)
    cond_source = bokeh.models.ColumnDataSource(cond_data)
    cB_source = bokeh.models.ColumnDataSource(cB_data)
    frac_source = bokeh.models.ColumnDataSource(frac_data)

    ## Figure setup
    p = bokeh.plotting.figure(
        frame_width=width,
        frame_height=height,
        x_axis_label='Volume (ml)',
        y_axis_label='Absorbance 280 nm (a.u)',
        toolbar_location='above'
    )

    # change color for primary y-axis
    p.yaxis.axis_line_color = "dodgerblue"
    p.yaxis.axis_label_text_color = "dodgerblue"

    # second y axis for conductivity
    cond_low = min(cond_data['mS/cm']) - 0.1 * min(cond_data['mS/cm'])
    cond_high = max(cond_data['mS/cm']) + 0.1 * max(cond_data['mS/cm'])
    p.extra_y_ranges['cond'] = bokeh.models.Range1d(cond_low, cond_high)
    p.add_layout(bokeh.models.LinearAxis(axis_label='Conductivity (mS/cm)', 
                                        y_range_name='cond',
                                        axis_label_text_color='orange',
                                        axis_line_color='orange'
                                        ), 'right')

    # third y axis for concentration of B
    p.extra_y_ranges['conc B'] = bokeh.models.Range1d(-2, 105)
    p.add_layout(bokeh.models.LinearAxis(axis_label='Gradient (% B)', 
                                        y_range_name='conc B',
                                        axis_label_text_color='green',
                                        axis_line_color='green'
                                        ), 'right')

    # plotting lines
    cond = p.line(source=cond_source, x='ml', y='mS/cm', line_width=2, color='orange', y_range_name='cond')
    uv = p.line(source=uv_source, x='ml', y='mAU', line_width=2, color='dodgerblue')
    cB = p.line(source=cB_source, x='ml', y='%', line_width=2, color='green', y_range_name='conc B')

    ## tooltips
    tooltips_1=[('UV280', '@{mAU} mAU')]
    tooltips_2=[('Cond', '@{mS/cm} mS/cm')]
    tooltips_3=[('% B', '@{%}%')]

    p.add_tools(bokeh.models.HoverTool(tooltips=tooltips_3, 
                                    renderers=[cB], 
                                    mode='vline', 
                                    description='Conc B'))

    p.add_tools(bokeh.models.HoverTool(tooltips=tooltips_2, 
                                    renderers=[cond], 
                                    mode='vline', 
                                    description='Conductivity'))

    p.add_tools(bokeh.models.HoverTool(tooltips=tooltips_1, 
                                    renderers=[uv], 
                                    mode='vline', 
                                    description='UV280'))

    ## legend
    legend_items = [('UV280', [uv]), ('Conductivity', [cond]), ('% B', [cB])]
    legend = bokeh.models.Legend(items=legend_items, click_policy='hide',border_line_alpha=0)
    p.add_layout(legend, 'right')

    ## Fractions
    for fraction in frac_data.iterrows():
        ml = fraction[1].values[0]
        frac = fraction[1].values[1]
        y_frac = 0.1*max(uv_data['mAU'])
        
        p.line(x=ml,
            y=[0, y_frac],
            color='black'
            )

    # adding text to fractions
    labels = bokeh.models.LabelSet(x='ml', 
                                y=y_frac, 
                                text='Fraction',
                                x_offset=0,
                                y_offset=0, 
                                source=frac_source, 
                                angle_units='deg',
                                angle=45,
                                text_font_size='10px')

    p.add_layout(labels)

    bokeh.io.show(p)

    return None


def plot_subplots(data: pd.DataFrame,
                  ax1,
                  cond: bool=False,
                  concB: bool=False,
                  fractions: bool=False,
                  min_x: float=None,
                  max_x: float=None,
                  elution: bool=False,
                  ylim: list=[],
                  sample: str=''
                  ):
    """
    data: pd.DataFrame, data frame containing the data exported from Unicorn
    ax1: axes object to plot on
    cond: bool, whether to plot conductivity
    concB: bool, whether to plot the concentration of B
    fractions: bool, whether to show the fractions
    min_x: float, start of x axis
    max_x: float, end of x axis
    elution: bool, whether to show only elution phase (will overwrite min_x and max_x) 
    ylim: list, manual limits for UV signal on y axis
    sample: str, sample name, will be set as figure title if provided
    
    Function to plot a chromatogram on a given figure
    """
    
    # Warning in case the user tries something stupid
    if (min_x != None or max_x != None) and elution:
        warnings.warn('Elution and min_x / max_x used! Elution will overwrite manual x-axis range!')
    
    if min_x != None and max_x != None:
        if min_x > max_x:
            warnings.warn('min_x > max_x - are you sure you want to flip the x-axis?')
        
    # using the elution entry in the logbook for x axis range
    if elution:
        log_df = data['Run Log']
        min_x = int(log_df.loc[log_df['Logbook']=='Elution']['ml'].values[0])
        try:
            idx = log_df.loc[log_df['Logbook']=='Elution']['ml'].index[0]
            max_x = math.ceil(log_df['ml'].loc[idx + 1])
        except:
            pass
    
    # get name of uv data
    uv_names = ['UV', 'UV 1_280']
    uv_avail = [x for x in uv_names if x in data.columns]
    uv_name = uv_avail[0]
    print(f'using UV channel {uv_name} of available channels {uv_avail}')

    # assigning min and max x if not set
    if min_x == None and max_x == None:
        min_x = data[uv_name]['ml'].min()
        max_x = data[uv_name]['ml'].max()
        
    elif min_x == None and max_x != None:
        min_x = data[uv_name]['ml'].min()
        
    elif max_x == None and min_x != None:
        max_x = data[uv_name]['ml'].max()
    
    # using min_x * 0.05 for both paddings to 
    # add identical space left and right
    min_x -= min_x * 0.05
    max_x += min_x * 0.05
    
    # setting up y axis limits
    uv_max = data[data[uv_name]['ml'].between(min_x,max_x)][uv_name]['mAU'].max()
    cond_max = data[data['Cond']['ml'].between(min_x,max_x)]['Cond']['mS/cm'].max()

    ax1.set_xlabel('Volume (ml)', fontsize=18, fontname='Helvetica')
    ax1.tick_params(axis='x', labelsize=15)

    # plotting UV280
    uv = sns.lineplot(data=data[data[uv_name]['ml'].between(min_x,max_x)][uv_name], 
                      x='ml', 
                      y='mAU', 
                      color='tab:blue', 
                      ax=ax1
                     )
    if ylim == []:
        uv.set_ylim(0-uv_max*0.05, uv_max+uv_max*0.05)
    else:
        uv.set_ylim(*ylim)

    ax1.set_ylabel('Absorbance 280nm (mAU)', fontsize=18, fontname='Helvetica', color='tab:blue')
    ax1.yaxis.set_tick_params(labelsize=15)

    if sum([cond, concB]) == 2:
        # plotting conductivity
        ax2 = ax1.twinx()
        cond = sns.lineplot(data=data['Cond'], x='ml', y='mS/cm', color='tab:orange', ax=ax2)
        ax2.set_ylim(0-cond_max*0.05, cond_max+cond_max*0.05)
        ax2.set_ylabel('Conductivity (mS/cm)', fontsize=18, fontname='Helvetica', color='tab:orange')
        ax2.yaxis.set_tick_params(labelsize=15)

        # plotting %B
        ax3 = ax1.twinx()
        ax3.spines.right.set_position(("axes", 1.075))
        concB = sns.lineplot(data=data['Conc B'],x='ml',y='%',color='green', ax=ax3)
        ax3.set_ylim(-5, 105)
        ax3.set_ylabel('Concentration B (%)', fontsize=18, fontname='Helvetica', color='green')
        ax3.yaxis.set_tick_params(labelsize=15)
    
    elif sum([cond, concB]) == 1:
        ax2 = ax1.twinx()
        
        if cond:
            # plotting conductivity
            cond = sns.lineplot(data=data['Cond'], x='ml', y='mS/cm', color='tab:orange', ax=ax2)
            ax2.set_ylim(0-cond_max*0.05, cond_max+cond_max*0.05)
            ax2.set_ylabel('Conductivity (mS/cm)', fontsize=18, fontname='Helvetica', color='tab:orange')
            ax2.yaxis.set_tick_params(labelsize=15)
            
        elif concB:
            # plotting concentration B
            concB = sns.lineplot(data=data['Conc B'],x='ml',y='%',color='green', ax=ax2)
            ax2.set_ylim(-5, 105)
            ax2.set_ylabel('Concentration B (%)', fontsize=18, fontname='Helvetica', color='green')
            ax2.yaxis.set_tick_params(labelsize=15)
            
    else:
        pass

    # plot fractions
    if fractions:
        
        # adjusting fraction labels
        fraction_volume = data['Fraction'].dropna().loc[1,'ml'] - data['Fraction'].dropna().loc[0,'ml']
        offset = 0.6 * fraction_volume
        text_y = uv_max * 0.03

        for row in data['Fraction'].dropna().iterrows():
            ml = row[1]['ml']
            fraction = row[1]['Fraction']

            # plotting lines and labels
            if min_x and max_x:
                if min_x < ml and ml < max_x:
                    ax1.axvline(ml, ymax=0.08, color='black', alpha=0.5)
                    ax1.text(x=ml+offset, y=text_y, s=fraction, rotation=90, fontsize=8, horizontalalignment='center')

            elif min_x:
                if min_x < ml:
                    ax1.axvline(ml, ymax=0.08, color='black', alpha=0.5)
                    ax1.text(x=ml+offset, y=text_y, s=fraction, rotation=90, fontsize=8, horizontalalignment='center')

            elif max_x:
                if ml < max_x:
                    ax1.axvline(ml, ymax=0.08, color='black', alpha=0.5)
                    ax1.text(x=ml+offset, y=text_y, s=fraction, rotation=90, fontsize=8, horizontalalignment='center')
    
    ax1.set_xlim([min_x, max_x])

    # setting plot title if sample name is provided
    if sample != '':
        ax1.set_title(sample, fontsize=18)
    
    return None


def fplc(data: pd.DataFrame,
         cond: bool=False,
         concB: bool=False,
         fractions: bool=False,
         min_x: float=None,
         max_x: float=None,
         ylim: list=[],
         elution: bool=False,
         height: int=8,
         width: int=18,
         save_svg: bool=False, 
         save_png: bool=False,
         out_file: str='plot',
         sample: str='',
         **save_kwargs
        ):
    """
    data: pd.DataFrame, data frame containing the data exported from Unicorn
    cond: bool, whether to plot conductivity
    concB: bool, whether to plot the concentration of B
    fractions: bool, whether to show the fractions
    min_x: float, start of x axis
    max_x: float, end of x axis
    ylim: list, manual limits for UV signal on y axis
    elution: bool, whether to show only elution phase (will overwrite min_x and max_x) 
    height: int, height of the plot
    width: int, width of the plot
    save_png: bool, whether to save figure as png
    save_svg: bool, whether to save figure as svg
    out_file: str, path to output file
    sample: str, sample name, will be set as figure title if provided
    
    Function to plot chromatograms from FPLC data. Need to import data first using 
    wetlabtools.plot.import_fplc().
    """
    
    # initializing plot
    fig, ax1 = plt.subplots(figsize=(width,height))
    fig.subplots_adjust(right=0.75)

    # plotting the chromatogram
    plot_subplots(data=data,
                  ax1=ax1,
                  cond=cond,
                  concB=concB,
                  fractions=fractions,
                  min_x=min_x,
                  max_x=max_x,
                  elution=elution,
                  sample=sample,
                  ylim=ylim
                  )

    # saving figure if path is provided
    if save_svg:
        plt.savefig(out_file + '.svg', format='svg')
        print(f'saved plot: {out_file}')
    
    if save_png:
        plt.savefig(out_file + '.png', format='png')
        print(f'saved plot: {out_file}')

    return fig


def fplc_summary(directory: str, save_figure: bool=False):
    """
    directory: str, path to the directory containing the chromatograms
    save_figure: bool, save the figure? Will be saved in the input directory

    Function to plot a panel containing His and SEC chromatograms of a protein purification.
    This function expects IMAC and SEC chromatograms for each sample in the input directory.
    Files should be named like this XXX_sample-name_His and XXX_sample-name_Sec. The script 
    will then automatically copy the sample name. If there is only IMAC or SEC data for one
    sample, the function should still work. The panel will be saved in the same directory.
    """

    # collect files
    files = [file for file in os.listdir(directory) if file.endswith('.csv')]

    # collect sample names
    samples = set([file.split('_')[-2] for file in files])

    # create subplots: 1 row for each sample, 2 columns for imac and sec
    fig, ax = plt.subplots(len(samples), 2, figsize=(20,5*len(samples)))

    # adjusting the padding around the subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=1.25, 
                        top=0.9, 
                        wspace=0.35, 
                        hspace=0.4)
        
    for i, sample in enumerate(samples):
        
        # setting axes
        if len(samples) == 1:
            ax_imac = ax[0]
            ax_sec = ax[1]
            ax_imac.set_title('IMAC', fontsize=24, fontname='Helvetica')
            ax_sec.set_title('SEC', fontsize=24, fontname='Helvetica')
        
        else:
            ax_imac = ax[i, 0]
            ax_sec = ax[i, 1]
            if i == 0:
                ax[i, 0].set_title('IMAC', fontsize=24, fontname='Helvetica')
                ax[i, 1].set_title('SEC', fontsize=24, fontname='Helvetica')
            
        # find data files
        try:
            s = sample + '_his'
            imac = [file for file in files if s.casefold() in file.casefold() and 'his'.casefold() in file.casefold()][0]
        except IndexError:
            print(f'did not find a IMAC chromatogram file for {sample}')
            imac = None
        
        try:
            s = sample + '_sec'
            sec =  [file for file in files if s.casefold() in file.casefold() and 'sec'.casefold() in file.casefold()][0]
        except IndexError:
            print(f'did not find a SEC chromatogram file for {sample}')
            sec = None

        # importing data and plotting
        if imac == None:
            # removing plot
            ax_imac.spines['top'].set_visible(False)
            ax_imac.spines['right'].set_visible(False)
            ax_imac.spines['bottom'].set_color('none')
            ax_imac.spines['left'].set_color('none')
            ax_imac.xaxis.set_ticks_position('none')
            ax_imac.yaxis.set_ticks_position('none')
            ax_imac.set_xticks([])
            ax_imac.set_yticks([])
            
        else:
            imac_data = import_fplc(os.path.join(directory, imac))
            plot_subplots(data=imac_data,
                          ax1=ax_imac,
                          elution=True,
                          cond=True,
                          concB=True,
                          fractions=True
                          )
            
            # adding sample description
            ax_imac.text(0.02, 0.9, sample, 
                        transform=ax_imac.transAxes, 
                        fontsize=16, 
                        fontname='Helvetica')
        
        if sec == None:
            ax_sec.spines['top'].set_visible(False)
            ax_sec.spines['right'].set_visible(False)
            ax_sec.spines['bottom'].set_color('none')
            ax_sec.spines['left'].set_color('none')
            ax_sec.xaxis.set_ticks_position('none')
            ax_sec.yaxis.set_ticks_position('none')
            ax_sec.set_xticks([])
            ax_sec.set_yticks([])
            
        else:
            sec_data = import_fplc(os.path.join(directory, sec))
            plot_subplots(data=sec_data,
                          ax1=ax_sec,
                          elution=False,
                          cond=False,
                          concB=False,
                          fractions=True
                          )
            
            # adding sample description
            ax_sec.text(0.02, 0.9, sample, 
                        transform=ax_sec.transAxes, 
                        fontsize=16, 
                        fontname='Helvetica')

    if save_figure:
        plt.savefig(os.path.join(directory, 'summary.png'), dpi=300, bbox_inches='tight')

    plt.show()
    return None