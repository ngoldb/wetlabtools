"""
This submodule contains layouts to present CD data
"""

import os
import logging
import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from wetlabtools.chirascan import cd_experiment

logging.getLogger().setLevel(logging.CRITICAL)

# svg font names
plt.rcParams['svg.fonttype'] = 'none'

# small helper
def extract_n_elements(lst, n):
    '''returns n evenly spaced elements from list'''
    if n > len(lst):
        n = len(lst)
    lst.sort()
    indices = np.linspace(0, len(lst) - 1, n, dtype=int)
    return [lst[i] for i in indices]


def single_wvl_melt(premelt_file: str, postmelt_file: str, melt_file: str, blank_file: str, sample_data: dict,
                    qc_metric: str='hv', wvl_lim: tuple=None, tmp_lim: tuple=None, ylim: tuple=None, show_ci: bool=False, 
                    title: bool=True, save_png: bool=False, save_svg: bool=False):
    """
    premelt_file: str, path to the csv file of pre melt spectra
    postmelt_file: str, path to the csv file of post melt spectra
    melt_file: str, path to the csv file of the melt
    blank_file: str, path to the csv file of the blank
    sample_data: dict, sample information 
    qc_metric: str, which metric to use for qc plots [hv, absorbance]
    wvl_lim: tuple, limits of the wavelength range to show for pre- and post melt spectra
    tmp_lim: tuple, limits of the temperature range to show for melting curve
    ylim: tuple, limits of cd signal (y axis).
    show_ci: bool, whether to show confidence interval based on repeated measurements
    title: bool, whether to display title on plot (sample id)
    save_png: bool, whether to save the figure as png
    save_svg: bool, whether to save the figure as svg

    Function to plot data from a single wavelength melt experiment
    """
    
    # load data
    pre_melt = cd_experiment(result_file=premelt_file, sample_data=sample_data)
    post_melt = cd_experiment(result_file=postmelt_file, sample_data=sample_data)
    melt = cd_experiment(result_file=melt_file, sample_data=sample_data)

    # subtract blank
    pre_melt.subtract_blank(blank_file)
    post_melt.subtract_blank(blank_file)
    melt.subtract_blank(blank_file)

    # convert to MREx10^3
    pre_melt.convert(unit='MREx103')
    post_melt.convert(unit='MREx103')
    melt.convert(unit='MREx103')

    # account for user input
    qc_metrics = ['HV', 'Absorbance']
    try:
        qc_metric = [metric for metric in qc_metrics if metric.casefold() == qc_metric.casefold()][0]
    except IndexError as err:
        print(f'quality control metric "{qc_metric}" not available from {qc_metrics}')

    # define colors
    colors = ['#4a7bb7', '#dd3d2d']

    save_dir = os.path.join(os.path.dirname(premelt_file), 'plots') 
    
    # create one plot for each cell
    for cell in pre_melt.get_cells().keys():
        pre_df = pre_melt.get_data(cell[-1])
        sample = pre_df['CircularDichroism']['sample_id'].unique()[0]
        post_df = post_melt.get_data(cell[-1])
        melt_df = melt.get_data(cell[-1])

        # create figure
        fig = plt.figure(layout='constrained', figsize=(5.5, 3), dpi=300)
        if title:
            fig.suptitle(sample, fontsize=9)
        subfigs = fig.subfigures(1, 2, wspace=0.07)

        # full spectra: pre and post melt
        subfigsnest0 = subfigs[0].subfigures(1, 1)
        axsnest0 = subfigsnest0.subplots(2, 1, sharex=True, height_ratios=[2, 0.5])
        labels = ['pre-melt', 'post-melt']

        for i, df in enumerate([pre_df, post_df]):
            cd_df = df['CircularDichroism']
            qc_df = df[qc_metric]

            if wvl_lim:
                cd_df = cd_df[cd_df['Wavelength'].between(*wvl_lim)]
                qc_df = qc_df[qc_df['Wavelength'].between(*wvl_lim)]
            
            if not show_ci:
                cd_df = cd_df.groupby(['Wavelength'], as_index=False)['value'].mean()
                qc_df = qc_df.groupby(['Wavelength'], as_index=False)['value'].mean()

            sns.lineplot(data=cd_df, x="Wavelength", y="value", ax=axsnest0[0], color=colors[i], label=labels[i]).axhline(0, ls='--', color="black", linewidth=0.5)
            sns.lineplot(data=qc_df, x="Wavelength", y="value", ax=axsnest0[1], color=colors[i])
        
        # melting
        subfigsnest1 = subfigs[1].subfigures(1, 1)
        axsnest1 = subfigsnest1.subplots(2, 1, sharex=True, height_ratios=[2, 0.5])
        
        cd_df = melt_df['CircularDichroism']
        qc_df = melt_df[qc_metric]
        wavelength = cd_df.Wavelength.unique()[0]

        if not show_ci:
            cd_df = cd_df.groupby(['Temperature', 'phase'], as_index=False)['value'].mean()
            qc_df = qc_df.groupby(['Temperature', 'phase'], as_index=False)['value'].mean()

        if tmp_lim:
            cd_df = cd_df[cd_df['Temperature'].between(*tmp_lim)]
            qc_df = qc_df[qc_df['Temperature'].between(*tmp_lim)]

        if 'phase' not in cd_df.columns:
            cd_df['phase'] = 'heat'
            palette = {'heat': '#dd3d2d'}
        else:
            palette = {'heat': '#dd3d2d', 'cool': '#4a7bb7'}

        sns.lineplot(data=cd_df, x="Temperature", y="value", ax=axsnest1[0], hue='phase', palette=palette).axhline(0, ls='--', color="black", linewidth=0.5)
        sns.lineplot(data=qc_df, x="Temperature", y="value", ax=axsnest1[1], hue='phase', palette=palette)
        
        axsnest1[1].legend().remove()

        # Styling plots
        if qc_metric == 'HV':
            qc_label = 'HT (V)'
        elif qc_metric == 'Absorbance':
            qc_label = 'Absorbance\n (a.u.)'

        ## labels
        axsnest1[1].set_xlabel('Temperature ($^\circ$C)\n\n', fontsize=7)
        axsnest0[1].set_xlabel('Wavelength ($\mathrm{nm}$)\n\n', fontsize=7)
        axsnest0[0].set_ylabel('$\mathrm{MRE}$\n  $\mathrm{(deg\ cm^{2}\ dmol^{-1}\ res^{-1}\ x10^{3}}$)', fontsize=7)
        axsnest1[0].set_ylabel(
            f'$\\mathrm{{MRE_{{{wavelength}}}}}$\n  $\\mathrm{{(deg\\ cm^{{2}}\\ dmol^{{-1}}\\ res^{{-1}}\\ x10^{{3}})}}$',
            fontsize=7
        )
        axsnest1[1].set_ylabel(qc_label, fontsize=7)
        axsnest0[1].set_ylabel(qc_label, fontsize=7)

        ## legend
        axsnest1[0].legend(fontsize=6, frameon=False)
        axsnest0[0].legend(fontsize=6, frameon=False)

        ## axis limits
        if qc_metric == 'HV':
            axsnest0[1].set_ylim(0, 1000)
            axsnest1[1].set_ylim(0, 1000)
            axsnest1[1].set_yticks([0, 500, 1000])
            axsnest0[1].set_yticks([0, 500, 1000])

        if ylim:
            axsnest0[0].set_ylim(*ylim)
            axsnest1[0].set_ylim(*ylim)
        
        ## xtick label fontsize  
        axsnest0[0].tick_params(axis='y', labelsize=7)
        axsnest1[0].tick_params(axis='y', labelsize=7)
        axsnest0[1].tick_params(axis='y', labelsize=7)
        axsnest1[1].tick_params(axis='y', labelsize=7)
        axsnest0[1].tick_params(axis='x', labelsize=7)
        axsnest1[1].tick_params(axis='x', labelsize=7)
        
        sns.despine()

        save_path = os.path.join(save_dir, sample)
        if save_png or save_svg:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        if save_png:
            save_file = save_path + '.png'
            plt.savefig(save_file, bbox_inches="tight")
            print(f'saving file to {save_file}')
        
        if save_svg:
            save_file = save_path + '.svg'
            plt.savefig(save_file, bbox_inches="tight")
            print(f'saving file to {save_file}')

        plt.show()
        plt.close('all')


# TODO
# - account for final measurement after melt
def full_spectrum_melt(
        melt_file: str, 
        blank_file: str=None, 
        sample_data: dict=None, 
        qc_metric: str='HV',
        wvl_lim: tuple=None, 
        tmp_lim: tuple=None, 
        ylim: tuple=None, 
        show_ci: bool=False, 
        title: bool=True, 
        colormap: str='vlag',
        melt_wavelength: int=222,
        n_spectra: int=999,
        legend: str='continuous',
        save_png: bool=False, 
        save_svg: bool=False
    ):
    """
    melt_file: str, path to the csv file of the melt
    blank_file: str, path to the csv file of the blank
    sample_data: dict, sample information 
    qc_metric: str, which metric to use for qc plots [hv, absorbance]
    wvl_lim: tuple, limits of the wavelength range to show for pre- and post melt spectra
    tmp_lim: tuple, limits of the temperature range to show for melting curve
    ylim: tuple, limits of cd signal (y axis).
    show_ci: bool, whether to show confidence interval based on repeated measurements
    title: bool, whether to display title on plot (sample id)
    colormap: str, name of the seaborn colormap to use (default: vlag)
    melt_wavelength: int, wavelength to plot for the melting curve
    n_spectra: int, number of spectra to plot (does not affect melting curve)
    legend: str, type of legend to generate ['discrete', 'continuous'] - False if no legend should be generated
    save_png: bool, whether to save the figure as png
    save_svg: bool, whether to save the figure as svg

    Function to plot data from a temperature melt experiment with full spectra recorded. 
    Currently does not support to plot final measurements after heat ramp
    """

    melt = cd_experiment(melt_file, sample_data=sample_data)
    melt.subtract_blank(blank_file)
    melt.convert('mrex103')

    # account for user input
    qc_metrics = ['HV', 'Absorbance']
    try:
        qc_metric = [metric for metric in qc_metrics if metric.casefold() == qc_metric.casefold()][0]
    except IndexError as err:
        print(f'quality control metric "{qc_metric}" not available from {qc_metrics}')
        
    for cell in melt.get_cells().keys():
        df = melt.get_data(cell[-1])
        sample = df['CircularDichroism']['sample_id'].unique()[0]

        # create figure
        fig = plt.figure(layout='constrained', figsize=(5.5, 3), dpi=300)
        if title:
            fig.suptitle(sample, fontsize=9)
        subfigs = fig.subfigures(1, 2, wspace=0.07)

        # full spectra plot (left)
        subfigsnest0 = subfigs[0].subfigures(1, 1)
        axsnest0 = subfigsnest0.subplots(2, 1, sharex=True, height_ratios=[2, 0.5])

        # setting up color palette
        cmap = sns.color_palette(colormap, as_cmap=True)

        cd_df = df['CircularDichroism']
        qc_df = df[qc_metric]

        if cd_df.Temperature.dtype != int:
            cd_df.loc[:, 'Temperature'] = cd_df.Temperature.astype(int)
        if qc_df.Temperature.dtype != int:
            qc_df.loc[:, 'Temperature'] = qc_df.Temperature.astype(int)

        # setting up user provided limits
        if wvl_lim:
            cd_df = cd_df[cd_df['Wavelength'].between(*wvl_lim)]
            qc_df = qc_df[qc_df['Wavelength'].between(*wvl_lim)]

        if tmp_lim:
            cd_df = cd_df[cd_df['Temperature'].between(*tmp_lim)]
            qc_df = qc_df[qc_df['Temperature'].between(*tmp_lim)]

        if not show_ci:
            cd_df = cd_df.groupby(['Wavelength', 'Temperature'], as_index=False)['value'].mean()
            qc_df = qc_df.groupby(['Wavelength', 'Temperature'], as_index=False)['value'].mean()

        # prepare data frame for melting plot
        melt_cd = cd_df[cd_df['Wavelength'] == melt_wavelength]
        melt_qc = qc_df[qc_df['Wavelength'] == melt_wavelength]

        # extract spectra if needed
        if n_spectra < len(cd_df.Temperature.unique()):
            temp_to_plot = extract_n_elements(cd_df.Temperature.unique(), n_spectra)
            cd_df = cd_df[cd_df['Temperature'].isin(temp_to_plot)]
            qc_df = qc_df[qc_df['Temperature'].isin(temp_to_plot)]

        sns.lineplot(data=cd_df, x="Wavelength", y="value", ax=axsnest0[0], hue='Temperature', legend='full', palette=cmap).axhline(0, ls='--', color="black", linewidth=0.5)
        sns.lineplot(data=qc_df, x="Wavelength", y="value", ax=axsnest0[1], hue='Temperature', legend=False, palette=cmap)

        # melting plot (right)
        subfigsnest1 = subfigs[1].subfigures(1, 1)
        axsnest1 = subfigsnest1.subplots(2, 1, sharex=True, height_ratios=[2, 0.5])

        sns.lineplot(data=melt_cd, x="Temperature", y="value", ax=axsnest1[0], color='tab:blue', legend=False).axhline(0, ls='--', color="black", linewidth=0.5)
        sns.lineplot(data=melt_qc, x="Temperature", y="value", ax=axsnest1[1], color='tab:blue', legend=False)

        axsnest1[1].legend().remove()

        # Styling plots
        if qc_metric == 'HV':
            qc_label = 'HT (V)'
        elif qc_metric == 'Absorbance':
            qc_label = 'Absorbance\n (a.u.)'

        ## labels
        axsnest1[1].set_xlabel('Temperature ($^\circ$C)\n\n', fontsize=7)
        axsnest0[1].set_xlabel('Wavelength ($\mathrm{nm}$)\n\n', fontsize=7)
        axsnest0[0].set_ylabel('$\mathrm{MRE}$\n  $\mathrm{(deg\ cm^{2}\ dmol^{-1}\ res^{-1}\ x10^{3}}$)', fontsize=7)
        axsnest1[0].set_ylabel(
            f'$\\mathrm{{MRE_{{{melt_wavelength}}}}}$\n  $\\mathrm{{(deg\\ cm^{{2}}\\ dmol^{{-1}}\\ res^{{-1}}\\ x10^{{3}})}}$',
            fontsize=7
        )
        axsnest1[1].set_ylabel(qc_label, fontsize=7)
        axsnest0[1].set_ylabel(qc_label, fontsize=7)

        ## axis limits
        if qc_metric == 'HV':
            axsnest0[1].set_ylim(0, 1000)
            axsnest1[1].set_ylim(0, 1000)
            axsnest1[1].set_yticks([0, 500, 1000])
            axsnest0[1].set_yticks([0, 500, 1000])

        if ylim:
            axsnest0[0].set_ylim(*ylim)
            axsnest1[0].set_ylim(*ylim)

        ## xtick label fontsize  
        axsnest0[0].tick_params(axis='y', labelsize=7)
        axsnest1[0].tick_params(axis='y', labelsize=7)
        axsnest0[1].tick_params(axis='y', labelsize=7)
        axsnest1[1].tick_params(axis='y', labelsize=7)
        axsnest0[1].tick_params(axis='x', labelsize=7)
        axsnest1[1].tick_params(axis='x', labelsize=7)

        sns.despine()

        if not legend:
            axsnest0[0].legend().remove()

        elif legend == 'discrete':
            axsnest0[0].legend(
                ncols=3, 
                frameon=False, 
                fontsize=2, 
                title_fontsize=4,
                title='Temperature (°C)', 
                markerscale=2
            )
        elif legend == 'continuous':
            axsnest0[0].legend().remove()
            cbar_ax = axsnest0[0].inset_axes([0.8, 0.1, 0.03, 0.3])
            norm = matplotlib.colors.Normalize(vmin=cd_df.Temperature.min(), vmax=cd_df.Temperature.max())
            cb = plt.colorbar(
                matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=cbar_ax, 
                orientation='vertical',
                drawedges=False,
                format="%.0f °C"
            )
            cb.ax.tick_params(
                labelsize=5, 
                length=2, 
                width=0.5,
                pad=1
            )
            cb.outline.set_linewidth(0.5)

            # Saving plots
            save_dir = os.path.join(os.path.dirname(melt_file), 'plots') 
            save_path = os.path.join(save_dir, sample)
            if save_png or save_svg:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

            if save_png:
                save_file = save_path + '.png'
                plt.savefig(save_file, bbox_inches="tight")
                print(f'saving file to {save_file}')
            
            if save_svg:
                save_file = save_path + '.svg'
                plt.savefig(save_file, bbox_inches="tight")
                print(f'saving file to {save_file}')

            plt.show()
            plt.close('all')