"""
Module to plot any kind of chromatogram from FPLC (IMAC, SEC, ...)
"""

import os
import pandas as pd

import bokeh.io
import bokeh.layouts
import bokeh.models
import bokeh.plotting

notebook_url = 'localhost:8888'
bokeh.io.output_notebook()



def import_fplc(sec_data: str):
    """
    :param: sec_data: str, path to the csv file with FPLC data
    Function to load FPLC data and fix the column headings. Will return a Data Frame.
    """
    
    # open csv file and fix header formatting
    # fixed csv is written to temporary file
    tmp_file = 'wetlabtools_tmp.csv'

    with open(sec_data, 'r', encoding='utf16') as read_file, open(tmp_file, 'w') as write_file:
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



def interactive_fplc(csv_path: str):
    """
    :param: csv_path: str, path to the csv file containing FPLC data
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
        frame_width=600,
        frame_height=300,
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