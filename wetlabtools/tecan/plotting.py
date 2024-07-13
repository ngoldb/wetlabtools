"""Module to plot tecan data"""

import colorcet
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def plot_96_well_plate(df, title: str='96 Well Plate', show_values: bool=True, value_col: str='black', show: bool=True):
    """
    df: DataFrame, data to plot
    title: str, title of the plot
    show_values: bool, whether to annotate wells with values
    value_col: str, color of the annotation text
    show: bool, whether to show the plot

    Function to plot values from a 96 well plate
    """

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a grid for the 96-well plate
    rows = 'ABCDEFGH'
    cols = range(1, 13)
    
    # Normalize the values to get a color map
    well_key = [key for key in df.keys() if key.casefold() in ['well', 'wells']][0]
    value_key = [key for key in df.keys() if key != well_key][0]
    norm = matplotlib.colors.Normalize(vmin=df[value_key].min(), vmax=df[value_key].max())
    cmap = colorcet.m_CET_L18

    # Create a set of all possible well positions
    all_wells = {f"{row}{col}" for row in rows for col in cols}

    # Create a set of wells present in the dataframe
    wells_with_values = set(df[well_key])

    # Draw the wells
    for well in all_wells:
        row_idx = rows.index(well[0])
        col_idx = int(well[1:]) - 1
        x = col_idx + 0.5
        y = row_idx + 0.5
        
        if well in wells_with_values:
            value = df[df[well_key] == well][value_key].values[0]
            facecolor = cmap(norm(value))
            edgecolor = 'black'
            label = f"{value:.2f}"
        else:
            facecolor = 'none'
            edgecolor = 'gray'
            label = ''
        
        # Create a circle at the calculated position
        circle = matplotlib.patches.Circle((x, y), 0.4, edgecolor=edgecolor, facecolor=facecolor, lw=1)
        ax.add_patch(circle)
        
        # Annotate the value in the center of the well if it exists
        if label and show_values:
            ax.text(x, y, label, ha='center', va='center', color=value_col, fontsize=8)

    # Set the limits, labels, and title
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.set_xticks(np.arange(0.5, 12.5))
    ax.set_yticks(np.arange(0.5, 8.5))
    ax.set_xticklabels(cols)
    ax.set_yticklabels(rows)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_title(title, pad=50)

    # Create color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label(value_key)

    # Adjust the plot to fit the circles
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.grid(False)
    
    if show:
        plt.show()

    return fig, ax