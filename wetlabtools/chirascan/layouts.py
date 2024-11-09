"""
This submodule contains layouts to present CD data
"""

import os
import seaborn as sns
import matplotlib.pyplot as plt

from wetlabtools.chirascan import cd_experiment

# svg font names
plt.rcParams['svg.fonttype'] = 'none'


def single_wvl_melt(premelt_file: str, postmelt_file: str, melt_file: str, blank_file: str, sample_data: dict,
                    qc_metric: str='hv', wvl_lim: tuple=None, tmp_lim: tuple=None, show_ci: bool=False, save_png: bool=False,
                    save_svg: bool=False):
    """
    premelt_file: str, path to the csv file of pre melt spectra
    postmelt_file: str, path to the csv file of post melt spectra
    melt_file: str, path to the csv file of the melt
    blank_file: str, path to the csv file of the blank
    sample_data: dict, sample information 
    qc_metric: str, which metric to use for qc plots [ht, absorbance]
    wvl_lim: tuple, limits of the wavelength range to show for pre- and post melt spectra
    tmp_lim: tuple, limits of the temperature range to show for melting curve
    show_ci: bool, whether to show confidence interval based on repeated measurements
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
        axsnest1[1].set_xlabel('Temperature ($^\circ$C)\n\n', fontsize=8)
        axsnest0[1].set_xlabel('Wavelength ($\mathrm{nm}$)\n\n', fontsize=8)
        axsnest1[0].set_ylabel('$\mathrm{MRE}$\n  $\mathrm{(deg\ cm^{2}\ dmol^{-1}\ res^{-1}\ x10^{3}}$)', fontsize=8)
        axsnest0[0].set_ylabel('$\mathrm{MRE}$\n  $\mathrm{(deg\ cm^{2}\ dmol^{-1}\ res^{-1}\ x10^{3}}$)', fontsize=8)
        axsnest1[1].set_ylabel(qc_label, fontsize=8)
        axsnest0[1].set_ylabel(qc_label, fontsize=8)

        ## legend
        axsnest1[0].legend(fontsize=6, frameon=False)
        axsnest0[0].legend(fontsize=6, frameon=False)

        ## axis limits
        if qc_metric == 'HV':
            axsnest0[1].set_ylim(0, 1000)
            axsnest1[1].set_ylim(0, 1000)
            axsnest1[1].set_yticks([0, 500, 1000])
            axsnest0[1].set_yticks([0, 500, 1000])
        
        ## xtick label fontsize  
        axsnest0[0].tick_params(axis='y', labelsize=8)
        axsnest1[0].tick_params(axis='y', labelsize=8)
        axsnest0[1].tick_params(axis='y', labelsize=8)
        axsnest1[1].tick_params(axis='y', labelsize=8)
        axsnest0[1].tick_params(axis='x', labelsize=8)
        axsnest1[1].tick_params(axis='x', labelsize=8)
        
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