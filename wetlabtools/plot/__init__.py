"""Subpackage for plotting experimental data"""

from wetlabtools.plot.CD import cd, load_CD_data
from wetlabtools.plot.MALS import secmals
from wetlabtools.plot.FPLC import interactive_fplc, import_fplc, fplc, fplc_summary
from wetlabtools.plot.SPR import kinetics, spr_affinity, load_affinity_data, load_affinity_fit, fit_sigmoid_function, multi_affinity, spr_summary

__author__ = "Nicolas Goldbach"
__email__ = "nicolas.goldbach@epfl.ch"
__version__ = "0.0.1"