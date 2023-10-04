"""
This module contains simple helper functions
"""

def normalize_percent_max(y: float, max_value: float, min_value: float):
    '''Function to normalize data to a scale 0 - 100% of data range'''
    return (y - min_value) / (max_value - min_value)


def rel_scale(y: float, max_value: float):
    '''Function to scale data to a axis 0 to 1 where the maximal value is 1.'''
    return y / max_value