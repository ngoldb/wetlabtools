"""
This module contains simple helper functions
"""

def normalize_percent_max(y: float, max_value: float, min_value: float):
    '''Function to no normalize data to a scale 0 - 100% of data range'''
    return (y - min_value) / (max_value - min_value)